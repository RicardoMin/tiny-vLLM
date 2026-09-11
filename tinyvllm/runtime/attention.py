from __future__ import annotations

import torch
import torch.nn.functional as F

from tinyvllm.contiguous_kv import ContiguousKVCache

class ReferenceAttention:
    """Ragged SDPA reference; CPU offsets avoid device-to-host reads."""

    def __init__(
        self,
        q_offsets: tuple[int, ...],
        context_lens_before: tuple[int, ...],
        caches: tuple[ContiguousKVCache, ...],
    ) -> None:
        self.q_offsets = q_offsets
        self.context_lens_before = context_lens_before
        self.caches = caches
        self.masks: list[torch.Tensor | None] = []

        # Build offset causal masks once per step, then reuse across layers.
        for index, cache in enumerate(caches):
            query_len = q_offsets[index + 1] - q_offsets[index]
            past_len = context_lens_before[index]
            mask = None
            if past_len and query_len > 1:
                query_positions = torch.arange(
                    past_len,
                    past_len + query_len,
                    device=cache.key.device,
                )
                key_positions = torch.arange(
                    past_len + query_len,
                    device=cache.key.device,
                )
                mask = (
                    key_positions.unsqueeze(0)
                    <= query_positions.unsqueeze(1)
                )[None, None]
            self.masks.append(mask)


    def __call__(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        output = torch.empty_like(q)
        for index, cache in enumerate(self.caches):
            start, end = self.q_offsets[index:index + 2]
            past_len = self.context_lens_before[index]
            cache.write(
                layer_idx,
                past_len,
                k[start:end],
                v[start:end],
            )
            keys, values = cache.read(
                layer_idx,
                past_len + end - start,
            )
            attended = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                keys.transpose(0, 1).unsqueeze(0),
                values.transpose(0, 1).unsqueeze(0),
                attn_mask=self.masks[index],
                dropout_p=0.0,
                is_causal=past_len == 0,
                scale=scale,
                enable_gqa=q.shape[1] != k.shape[1]
            )
            output[start:end].copy_(
                attended.squeeze(0).transpose(0, 1)
            )
        return output