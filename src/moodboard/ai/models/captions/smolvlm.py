"""Fast SmolVLM caption adapter.

SmolVLM is used for the fast analysis path: it produces concise captions that
are later parsed into objects, symbols, textures, style, composition and mood.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from PIL import Image

from src.moodboard.ai.registry import FAST_CAPTION_MODEL_ID, local_model_status
from src.moodboard.core.cache import ML_MODEL_CACHE
from src.moodboard.core.schemas import VALID_ANALYSIS_DEPTHS, clamp_int


def try_fast_caption_model(
    images: list[Image.Image],
    analysis_depth: str = "fast",
) -> tuple[dict[int, str], str]:
    """Caption images with the locally cached fast VLM, if available."""

    if os.environ.get("MOODBOARD_ENABLE_ML", "1").lower() in {"0", "false", "no"}:
        return {}, "disabled"
    if os.environ.get("MOODBOARD_ENABLE_FAST_CAPTION", "1").lower() in {"0", "false", "no"}:
        return {}, "disabled"
    if not images:
        return {}, "no-images"
    if not local_model_status(FAST_CAPTION_MODEL_ID)["available"]:
        return {}, f"{FAST_CAPTION_MODEL_ID} not-installed"
    try:
        import torch  # type: ignore
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

        if "fast_caption" in ML_MODEL_CACHE:
            processor, model, device = ML_MODEL_CACHE["fast_caption"]
        else:
            requested_device = os.environ.get("MOODBOARD_FAST_CAPTION_DEVICE", "auto").lower()
            if requested_device in {"auto", "mps"} and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            processor = AutoProcessor.from_pretrained(FAST_CAPTION_MODEL_ID, local_files_only=True)
            model = AutoModelForImageTextToText.from_pretrained(
                FAST_CAPTION_MODEL_ID,
                local_files_only=True,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
            )
            model.to(device)
            model.eval()
            ML_MODEL_CACHE["fast_caption"] = (processor, model, device)

        depth = analysis_depth if analysis_depth in VALID_ANALYSIS_DEPTHS else "fast"
        default_limit = min(8 if depth == "fast" else 14, len(images))
        caption_limit = clamp_int(os.environ.get("MOODBOARD_FAST_CAPTION_LIMIT"), default_limit, 1, 96)
        captions: dict[int, str] = {}
        prompt = "Describe the visible objects, symbols, textures, style, composition and mood in one concise sentence."

        for idx, image in enumerate(images[:caption_limit]):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            if hasattr(processor, "apply_chat_template"):
                text = processor.apply_chat_template(messages, add_generation_prompt=True)
            else:
                text = prompt
            inputs = processor(text=text, images=image, return_tensors="pt")
            inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=72,
                    do_sample=False,
                    num_beams=1,
                )
            if "input_ids" in inputs and hasattr(generated_ids, "__getitem__"):
                prompt_len = inputs["input_ids"].shape[-1]
                generated_ids = generated_ids[:, prompt_len:]
            caption_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            caption_text = re.sub(r"\s+", " ", caption_text).strip()
            if caption_text:
                captions[idx] = caption_text[:260]
        return captions, f"{FAST_CAPTION_MODEL_ID} on {device}; {len(captions)} captions"
    except Exception as exc:
        print(f"[WARN] Fast caption model unavailable: {exc}", file=sys.stderr)
        return {}, f"fast-caption-fallback: {exc}"


def caption(images: list[Any], analysis_depth: str = "fast") -> tuple[dict[int, str], str]:
    """Registry-friendly alias for fast captioning."""

    return try_fast_caption_model(images, analysis_depth)
