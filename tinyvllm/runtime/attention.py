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
            batch.context_lens_before, 
            batch.context_lens_after, 
            batch.query_lens,
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
        self._attend(layer_idx, q, output, scale=scale)

        return output


    def _attend(
        self,
        layer_idx: int,
        q: torch.Tensor,
        output: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        for index, slots in enumerate(self.kv.read_slots):
            if slots is None:
                continue

            start, end = self.batch.q_offsets[index:index + 2]
            keys, values = self.cache.read(layer_idx, slots)

            attended = F.scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                keys.transpose(0, 1).unsqueeze(0),
                values.transpose(0, 1).unsqueeze(0),
                attn_mask=self.masks[index],
                dropout_p=0.0,
                is_causal=self.batch.context_lens_before[index] == 0,
                scale=scale,
                enable_gqa=q.shape[1] != self.cache.num_kv_heads,
            )

            output[start:end].copy_(
                attended.squeeze(0).transpose(0, 1)
            )


class CudaDecodeAttention(ReferenceAttention):
    """One-token queries read pages directly; longer queries use reference SDPA."""

    def _attend(
        self,
        layer_idx: int,
        q: torch.Tensor,
        output: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        super()._attend(layer_idx, q, output, scale=scale)

        if self.kv.decode is not None:
            from tinyvllm.kernels.paged_decode import paged_decode_attention

            paged_decode_attention(
                q,
                self.cache,
                layer_idx,
                self.kv.decode,
                output,
                scale=scale,
            )


def select_attention_backend(name: str, model: torch.nn.Module):
    """Select once at construction; an explicitly requestes backend must work."""

    if name == "reference":
        return ReferenceAttention

    if name != "cuda":
        raise ValueError(f"unsupported attention backend: {name}")

    parameter = next(model.parameters())
    config = model.config

    if parameter.device.type != "cuda" or parameter.dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError(
            "cuda attention requires CUDA float16 or bfloat16"
        )

    if not 1 <= config.head_dim <= 256:
        raise ValueError(
            "cuda attention requires head_dim in [1, 256]"
        )

    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError(
            "query heads must be divisible by KV heads"
        )

    # Compile/load before allocating KV; CPU/reference users need to toolkit.
    from tinyvllm.kernels.paged_decode import load_paged_decode

    load_paged_decode()

    return CudaDecodeAttention