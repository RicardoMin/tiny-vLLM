from __future__ import annotations

import torch
import torch.nn.functional as F

from tinyvllm.batch import PreparedBatch
from tinyvllm.paged_kv import PagedKVCache
from tinyvllm.runtime.contracts import ModelInput

class ReferenceAttention:
    """Paged storage plus per-request gather/SDPA, sharing one model forward"""

    def __init__(
        self,
        batch: PreparedBatch,
        model_input: ModelInput,
        cache: PagedKVCache,
    ) -> None:
        self.batch = batch
        self.kv = model_input.kv
        self.cache = cache
        self.masks: list[torch.Tensor | None] = []

        for before, after, query_len in zip(
            batch.context_lens_before, batch.context_lens_after, batch.query_lens,
        ):
            mask = None
            if before and query_len > 1:
                queries = torch.arange(before, after, device=cache.device)
                keys = torch.arange(after, device=cache.device)
                mask = (keys[None, :] <= queries[:, None])[None, None]
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
        self.cache.write(layer_idx, self.kv.slot_mapping, k, v)
        output = torch.empty_like(q)

        for index, slots in enumerate(self.kv.read_slots):
            start, end = self.batch.q_offsets[index: index + 2]
            keys, values = self.cache.read(layer_idx, slots)
            attended = F.scaled_dot_product_attention(
                q[start: end].transpose(0, 1).unsqueeze(0),
                keys.transpose(0, 1).unsqueeze(0),
                values.transpose(0, 1).unsqueeze(0),
                attn_mask=self.masks[index],
                dropout_p=0.0,
                is_causal=self.batch.context_lens_before[index] == 0,
                scale=scale,
                enable_gqa=q.shape[1] != k.shape[1],
            )
            output[start: end].copy_(attended.squeeze(0).transpose(0, 1))

        return output
