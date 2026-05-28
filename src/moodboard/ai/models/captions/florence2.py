"""Florence-2 caption and region adapter.

Florence-2 is the heavier local vision-language adapter. It provides detailed
captions and optional object/dense-region observations for balanced/deep runs.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from PIL import Image

from src.moodboard.ai.registry import FLORENCE_CAPTION_LIMIT, FLORENCE_MODEL_ID
from src.moodboard.core.cache import ML_MODEL_CACHE
from src.moodboard.core.schemas import VALID_ANALYSIS_DEPTHS, clamp_int


def try_florence_vision_tasks(
    images: list[Image.Image],
    analysis_depth: str = "balanced",
) -> tuple[dict[int, str], dict[int, list[dict[str, Any]]], str]:
    """Run Florence-2 captions and region tasks from the local model cache."""

    if os.environ.get("MOODBOARD_ENABLE_ML", "1").lower() in {"0", "false", "no"}:
        return {}, {}, "disabled"
    if os.environ.get("MOODBOARD_ENABLE_FLORENCE", "1").lower() in {"0", "false", "no"}:
        return {}, {}, "disabled"
    if not images:
        return {}, {}, "no-images"
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

        if "florence2" in ML_MODEL_CACHE:
            processor, model, device = ML_MODEL_CACHE["florence2"]
        else:
            requested_device = os.environ.get("MOODBOARD_FLORENCE_DEVICE", "cpu").lower()
            if requested_device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            processor = AutoProcessor.from_pretrained(
                FLORENCE_MODEL_ID,
                local_files_only=True,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_ID,
                local_files_only=True,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            model.to(device)
            model.eval()
            ML_MODEL_CACHE["florence2"] = (processor, model, device)

        def run_florence_task(image: Image.Image, task: str, text_input: str | None = None) -> Any:
            prompt = task if text_input is None else task + text_input
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=96,
                    num_beams=1,
                    do_sample=False,
                    use_cache=False,
                )
            raw_caption = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            if hasattr(processor, "post_process_generation"):
                try:
                    return processor.post_process_generation(raw_caption, task=task, image_size=image.size)
                except Exception:
                    return raw_caption
            return raw_caption

        captions: dict[int, str] = {}
        detections: dict[int, list[dict[str, Any]]] = {}
        depth = analysis_depth if analysis_depth in VALID_ANALYSIS_DEPTHS else "balanced"
        if depth == "fast":
            default_caption_limit = min(6, len(images))
            default_region_limit = 0
        elif depth == "deep":
            default_caption_limit = min(max(FLORENCE_CAPTION_LIMIT, len(images)), 96)
            default_region_limit = min(24, default_caption_limit)
        else:
            default_caption_limit = min(14, max(1, len(images)))
            default_region_limit = min(6, default_caption_limit)
        caption_limit = clamp_int(os.environ.get("MOODBOARD_FLORENCE_LIMIT"), default_caption_limit, 1, 96)
        region_limit = clamp_int(os.environ.get("MOODBOARD_FLORENCE_REGION_LIMIT"), default_region_limit, 0, caption_limit)
        for idx, image in enumerate(images[:caption_limit]):
            caption_payload = run_florence_task(image, "<MORE_DETAILED_CAPTION>")
            caption_text = ""
            if isinstance(caption_payload, dict):
                caption_text = str(caption_payload.get("<MORE_DETAILED_CAPTION>") or next(iter(caption_payload.values()), ""))
            else:
                caption_text = str(caption_payload)
            caption_text = re.sub(r"<[^>]+>", " ", caption_text)
            caption_text = re.sub(r"\s+", " ", caption_text).strip()
            if caption_text:
                captions[idx] = caption_text[:240]

            if idx >= region_limit:
                continue
            region_items: list[dict[str, Any]] = []
            for task in ("<OD>", "<DENSE_REGION_CAPTION>"):
                payload = run_florence_task(image, task)
                data = payload.get(task) if isinstance(payload, dict) else None
                if not isinstance(data, dict):
                    continue
                labels = data.get("labels", []) or []
                boxes = data.get("bboxes", []) or []
                for label, bbox in zip(labels, boxes):
                    clean = re.sub(r"\s+", " ", str(label)).strip(" .")
                    if not clean:
                        continue
                    region_items.append(
                        {
                            "label": clean[:80],
                            "bbox": [round(float(value), 2) for value in bbox[:4]] if isinstance(bbox, list) else None,
                            "task": task.strip("<>"),
                            "confidence": 0.82 if task == "<OD>" else 0.74,
                        }
                    )
            if region_items:
                detections[idx] = region_items[:24]
        return captions, detections, f"{FLORENCE_MODEL_ID} on {device}; {len(captions)} captions; {sum(len(v) for v in detections.values())} regions"
    except Exception as exc:
        print(f"[WARN] Florence-2 unavailable, captions skipped: {exc}", file=sys.stderr)
        return {}, {}, "caption-fallback"


def run(images: list[Any], depth: str = "balanced") -> tuple[dict[int, str], dict[int, list[dict[str, Any]]], str]:
    """Registry-friendly alias for Florence-2 vision-language tasks."""

    return try_florence_vision_tasks(images, depth)
