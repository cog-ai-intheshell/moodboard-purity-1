"""OWLv2 open-vocabulary grounding adapter.

This adapter grounds object/symbol/texture labels against image regions. The
analyzer prepares the candidate taxonomy; this module only owns model loading,
inference and result shaping.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from PIL import Image

from src.moodboard.ai.registry import OPEN_VOCAB_DETECTOR_MODEL_ID, local_model_status
from src.moodboard.ai.variables.labels import clean_concept_label
from src.moodboard.core.cache import ML_MODEL_CACHE
from src.moodboard.core.schemas import VALID_ANALYSIS_DEPTHS, clamp_float, clamp_int


def grounding_candidates_for_entry(entry: dict[str, Any], taxonomy: dict[str, list[str]], limit: int) -> list[str]:
    """Build a compact list of candidate region labels for one image."""

    candidates: list[str] = []
    for field in ("objects", "symbols", "textures"):
        for value in entry.get(field, []) or []:
            clean = clean_concept_label(str(value), max_words=4)
            if clean and clean not in candidates:
                candidates.append(clean)
    for obs_type in ("object", "symbol"):
        for value in taxonomy.get(obs_type, [])[: limit * 2]:
            clean = clean_concept_label(str(value), max_words=4)
            if clean and clean not in candidates:
                candidates.append(clean)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    return candidates[:limit]


def try_owlv2_grounding(
    images: list[Image.Image],
    image_entries: list[dict[str, Any]],
    taxonomy: dict[str, list[str]],
    analysis_depth: str = "balanced",
) -> tuple[dict[int, list[dict[str, Any]]], str]:
    """Run local OWLv2 open-vocabulary detection when enabled and installed."""

    if os.environ.get("MOODBOARD_ENABLE_ML", "1").lower() in {"0", "false", "no"}:
        return {}, "disabled"
    if os.environ.get("MOODBOARD_ENABLE_OWLV2", "1").lower() in {"0", "false", "no"}:
        return {}, "disabled"
    if not images or not image_entries:
        return {}, "no-images"
    if not local_model_status(OPEN_VOCAB_DETECTOR_MODEL_ID)["available"]:
        return {}, f"{OPEN_VOCAB_DETECTOR_MODEL_ID} not-installed"
    try:
        import torch  # type: ignore
        from transformers import AutoProcessor, Owlv2ForObjectDetection  # type: ignore

        if "owlv2" in ML_MODEL_CACHE:
            processor, model, device = ML_MODEL_CACHE["owlv2"]
        else:
            requested_device = os.environ.get("MOODBOARD_OWLV2_DEVICE", "auto").lower()
            if requested_device in {"auto", "mps"} and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            processor = AutoProcessor.from_pretrained(OPEN_VOCAB_DETECTOR_MODEL_ID, local_files_only=True)
            model = Owlv2ForObjectDetection.from_pretrained(OPEN_VOCAB_DETECTOR_MODEL_ID, local_files_only=True)
            model.to(device)
            model.eval()
            ML_MODEL_CACHE["owlv2"] = (processor, model, device)

        depth = analysis_depth if analysis_depth in VALID_ANALYSIS_DEPTHS else "balanced"
        image_limit = {"fast": 4, "balanced": 8, "deep": 24}.get(depth, 8)
        image_limit = clamp_int(os.environ.get("MOODBOARD_OWLV2_IMAGE_LIMIT"), min(image_limit, len(images)), 1, 96)
        label_limit = {"fast": 12, "balanced": 24, "deep": 48}.get(depth, 24)
        label_limit = clamp_int(os.environ.get("MOODBOARD_OWLV2_LABEL_LIMIT"), label_limit, 4, 80)
        threshold = clamp_float(float(os.environ.get("MOODBOARD_OWLV2_THRESHOLD", "0.18")), 0.18, 0.02, 0.80)
        detections: dict[int, list[dict[str, Any]]] = {}

        for idx, image in enumerate(images[:image_limit]):
            labels = grounding_candidates_for_entry(image_entries[idx], taxonomy, label_limit)
            if not labels:
                continue
            text_queries = [[f"a photo of {label}" for label in labels]]
            inputs = processor(text=text_queries, images=image, return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]], device=device)
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)[0]
            image_items: list[dict[str, Any]] = []
            boxes = results.get("boxes", [])
            scores = results.get("scores", [])
            label_indices = results.get("labels", [])
            for box, score, label_idx in zip(boxes, scores, label_indices):
                label_index = int(label_idx.detach().cpu().item() if hasattr(label_idx, "detach") else label_idx)
                if not 0 <= label_index < len(labels):
                    continue
                clean = clean_concept_label(labels[label_index], max_words=4)
                if not clean:
                    continue
                box_values = box.detach().cpu().float().tolist() if hasattr(box, "detach") else list(box)
                score_value = float(score.detach().cpu().item() if hasattr(score, "detach") else score)
                image_items.append(
                    {
                        "label": clean,
                        "bbox": [round(float(value), 2) for value in box_values[:4]],
                        "confidence": round(clamp_float(score_value, 0.0, 0.0, 1.0), 4),
                        "task": "owlv2-open-vocabulary",
                    }
                )
            if image_items:
                image_items.sort(key=lambda item: item["confidence"], reverse=True)
                deduped: list[dict[str, Any]] = []
                seen: set[str] = set()
                for item in image_items:
                    if item["label"] in seen:
                        continue
                    seen.add(item["label"])
                    deduped.append(item)
                    if len(deduped) >= 16:
                        break
                detections[idx] = deduped
        return detections, f"{OPEN_VOCAB_DETECTOR_MODEL_ID} on {device}; {sum(len(v) for v in detections.values())} grounded regions"
    except Exception as exc:
        print(f"[WARN] OWLv2 grounding unavailable: {exc}", file=sys.stderr)
        return {}, f"owlv2-fallback: {exc}"


def ground(
    images: list[Any],
    entries: list[dict[str, Any]],
    taxonomy: dict[str, list[str]],
    depth: str = "balanced",
) -> tuple[dict[int, list[dict[str, Any]]], str]:
    """Registry-friendly alias for open-vocabulary grounding."""

    return try_owlv2_grounding(images, entries, taxonomy, depth)
