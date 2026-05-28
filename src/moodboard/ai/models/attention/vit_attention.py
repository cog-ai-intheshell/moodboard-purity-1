"""Attention and salience signatures for ViT-like image encoders.

The current MVP uses SigLIP2 patch-token activations as a lightweight attention
proxy. The output is intentionally structured like an embedding plus readable
stats so it can later be replaced by true attention rollout without changing
graph/purity consumers.
"""

from __future__ import annotations

import math
import sys
from typing import Any

from src.moodboard.core.schemas import clamp_float


def siglip_attention_signature(tokens: Any, torch_module: Any) -> list[dict[str, Any]]:
    """Build a normalized salience signature from SigLIP patch tokens."""

    if tokens is None:
        return []
    try:
        batch_size, token_count, _hidden = tokens.shape
        if token_count <= 0:
            return []
        token_strength = torch_module.linalg.norm(tokens.float(), dim=-1)
        centered = token_strength - token_strength.mean(dim=1, keepdim=True)
        scale = token_strength.std(dim=1, keepdim=True).clamp_min(1e-6)
        weights = torch_module.softmax(centered / (scale * 0.85), dim=-1)
        grid_size = int(round(math.sqrt(token_count)))
        if grid_size * grid_size == token_count:
            axis = torch_module.linspace(0.0, 1.0, grid_size, device=tokens.device)
            yy, xx = torch_module.meshgrid(axis, axis, indexing="ij")
            coords = torch_module.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        else:
            coords = torch_module.stack(
                [
                    torch_module.linspace(0.0, 1.0, token_count, device=tokens.device),
                    torch_module.zeros(token_count, device=tokens.device),
                ],
                dim=1,
            )
        distances = torch_module.cdist(coords, coords, p=2)
        signatures: list[dict[str, Any]] = []
        max_entropy = math.log(max(2, token_count))
        top_k = max(1, int(math.ceil(token_count * 0.10)))
        for idx in range(int(batch_size)):
            item_weights = weights[idx]
            center = (item_weights[:, None] * coords).sum(dim=0)
            delta = coords - center
            spread = torch_module.sqrt((item_weights[:, None] * delta.pow(2)).sum(dim=0).clamp_min(0.0))
            entropy = float(-(item_weights * torch_module.log(item_weights + 1e-9)).sum().detach().cpu().item()) / max_entropy
            concentration = float(torch_module.topk(item_weights, k=top_k).values.sum().detach().cpu().item())
            mean_distance = float((item_weights[:, None] * item_weights[None, :] * distances).sum().detach().cpu().item())
            signature = item_weights.detach().cpu().float().tolist()
            signatures.append(
                {
                    "embedding": [round(float(value), 7) for value in signature],
                    "stats": {
                        "source": "siglip2-patch-token-salience",
                        "patchSize": 16,
                        "tokenCount": int(token_count),
                        "gridSize": grid_size if grid_size * grid_size == token_count else None,
                        "entropy": round(clamp_float(entropy, 0.0, 0.0, 1.0), 4),
                        "concentration": round(clamp_float(concentration, 0.0, 0.0, 1.0), 4),
                        "centerX": round(float(center[0].detach().cpu().item()), 4),
                        "centerY": round(float(center[1].detach().cpu().item()), 4),
                        "spreadX": round(float(spread[0].detach().cpu().item()), 4),
                        "spreadY": round(float(spread[1].detach().cpu().item()), 4),
                        "meanAttentionDistance": round(clamp_float(mean_distance / math.sqrt(2.0), 0.0, 0.0, 1.0), 4),
                    },
                }
            )
        return signatures
    except Exception as exc:
        print(f"[WARN] SigLIP2 attention signature unavailable: {exc}", file=sys.stderr)
        return []


def siglip_patch_signature(tokens: Any, torch_module: Any) -> list[dict[str, Any]]:
    """Compatibility alias used by the modular AI adapter API."""

    return siglip_attention_signature(tokens, torch_module)
