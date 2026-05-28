"""SigLIP2 image/text embedding adapter.

This module owns the local SigLIP2 runtime: lazy loading, cache reuse, image
embeddings, text embeddings and attention-signature extraction. Keeping it here
lets the analyzer call a stable adapter while the server remains transport-only.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from PIL import Image

from src.moodboard.ai.models.attention.vit_attention import siglip_attention_signature
from src.moodboard.ai.registry import SIGLIP_MODEL_ID
from src.moodboard.core.cache import ML_MODEL_CACHE


def load_siglip_model() -> tuple[Any, Any, str, Any]:
    """Load SigLIP2 once from the local Hugging Face cache."""

    import torch  # type: ignore
    from transformers import AutoModel, AutoProcessor  # type: ignore

    if "siglip2" in ML_MODEL_CACHE:
        processor, model, device = ML_MODEL_CACHE["siglip2"]
    else:
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID, local_files_only=True, use_fast=False)
        model = AutoModel.from_pretrained(SIGLIP_MODEL_ID, local_files_only=True)
        model.to(device)
        model.eval()
        ML_MODEL_CACHE["siglip2"] = (processor, model, device)
    return processor, model, device, torch


def try_siglip_embeddings_with_attention(images: list[Image.Image]) -> tuple[list[list[float]] | None, list[dict[str, Any]], str]:
    """Embed images and return optional patch-token salience signatures."""

    if os.environ.get("MOODBOARD_ENABLE_ML", "1").lower() in {"0", "false", "no"}:
        return None, [], "disabled"
    try:
        processor, model, device, torch = load_siglip_model()
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(model, "vision_model"):
                outputs = model.vision_model(**inputs, return_dict=True)
                feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)
                attention = siglip_attention_signature(getattr(outputs, "last_hidden_state", None), torch)
            elif hasattr(model, "get_image_features"):
                feats = model.get_image_features(**inputs)
                attention = []
            else:
                outputs = model(**inputs)
                feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)
                attention = siglip_attention_signature(getattr(outputs, "last_hidden_state", None), torch)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.detach().cpu().float().tolist(), attention, f"{SIGLIP_MODEL_ID} on {device}; attention-signature={len(attention)}"
    except Exception as exc:
        print(f"[WARN] SigLIP2 unavailable, using heuristic embeddings: {exc}", file=sys.stderr)
        return None, [], "heuristic-fallback"


def try_siglip_embeddings(images: list[Image.Image]) -> tuple[list[list[float]] | None, str]:
    """Embed images without returning attention details."""

    vectors, _attention, status = try_siglip_embeddings_with_attention(images)
    return vectors, status


def try_siglip_text_embeddings(texts: list[str]) -> tuple[list[list[float]] | None, str]:
    """Embed text prompts with the same SigLIP2 text tower."""

    if os.environ.get("MOODBOARD_ENABLE_ML", "1").lower() in {"0", "false", "no"}:
        return None, "disabled"
    if not texts:
        return [], "no-text"
    try:
        processor, model, device, torch = load_siglip_model()
        inputs = processor(text=texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(model, "get_text_features"):
                feats = model.get_text_features(**inputs)
            else:
                outputs = model(**inputs)
                feats = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs.last_hidden_state.mean(dim=1)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.detach().cpu().float().tolist(), f"{SIGLIP_MODEL_ID} text on {device}"
    except Exception as exc:
        print(f"[WARN] SigLIP2 text embeddings unavailable: {exc}", file=sys.stderr)
        return None, "text-fallback"


def load() -> tuple[Any, Any, str, Any]:
    """Registry-friendly alias for the model loader."""

    return load_siglip_model()


def embed_images(images: list[Any]) -> tuple[list[list[float]] | None, list[dict[str, Any]], str]:
    """Registry-friendly alias for image embedding."""

    return try_siglip_embeddings_with_attention(images)


def embed_texts(texts: list[str]) -> tuple[list[list[float]] | None, str]:
    """Registry-friendly alias for text embedding."""

    return try_siglip_text_embeddings(texts)
