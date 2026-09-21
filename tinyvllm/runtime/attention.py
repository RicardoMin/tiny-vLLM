from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.backends.cuda import SDPAParams, can_use_flash_attention
from torch.nn.attention.bias import CausalBias, causal_lower_right

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
        self.masks = [
            self._make_mask(before, after, query_len)
            for before, after, query_len in zip(
                batch.context_lens_before,
                batch.context_lens_after,
                batch.query_lens,
            )
        ]


    def _make_mask(
        self,
        before: int,
        after: int,
        query_len: int,
    ) -> torch.Tensor | None:
        if before == 0 or query_len == 1:
            return None

        queries = torch.arange(
            before,
            after,
            device=self.cache.device,
        )
        keys = torch.arange(
            after,
            device=self.cache.device,
        )

        return (
            keys[None, :] <= queries[:, None]
        )[None, None]


    def _attention_mask(
        self,
        index: int,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.masks[index]
    

    def __call__(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        self._write_kv(layer_idx, k, v)

        output = torch.empty_like(q)
        self._attend(layer_idx, q, k, v, output, scale=scale)

        return output


    def _write_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        self.cache.write(
            layer_idx,
            self.kv.slot_mapping,
            k,
            v,
        )


    def _attend(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        for index, slots in enumerate(self.kv.read_slots):
            if slots is None and (
                self.batch.query_lens[index] == 1
                or self.batch.context_lens_before[index] > 0
            ):
                continue

            start, end = self.batch.q_offsets[index:index + 2]

            if slots is None:
                keys, values = k[start:end], v[start:end]
            else:
                keys, values = self.cache.read(
                    layer_idx,
                    slots,
                )

            query = q[start:end].transpose(0, 1).unsqueeze(0)
            keys = keys.transpose(0, 1).unsqueeze(0)
            values = values.transpose(0, 1).unsqueeze(0)

            attended = F.scaled_dot_product_attention(
                query,
                keys,
                values,
                attn_mask=self._attention_mask(
                    index,
                    query,
                    keys,
                    values,
                ),
                dropout_p=0.0,
                is_causal=self.batch.context_lens_before[index] == 0,
                scale=scale,
                enable_gqa=q.shape[1] != self.cache.num_kv_heads,
            )

            output[start:end].copy_(
                attended.squeeze(0).transpose(0, 1)
            )


class CudaAttention(ReferenceAttention):
    """Fused KV writes, paged decode and fused causal chunk attention."""

    def _make_mask(
        self,
        before: int,
        after: int,
        query_len: int,
    ) -> torch.Tensor | None:
        if before == 0 or query_len == 1:
            return None

        return causal_lower_right(query_len, after)

    def _attention_mask(
        self,
        index: int,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor | None:
        mask = self.masks[index]

        if isinstance(mask, CausalBias):
            params = SDPAParams(
                query,
                keys,
                values,
                None,
                0.0,
                False,
                query.shape[1] != keys.shape[1],
            )

            if not can_use_flash_attention(params):
                # Cache one explicit mask instead of materializing it per layer.
                mask = super()._make_mask(
                    self.batch.context_lens_before[index],
                    self.batch.context_lens_after[index],
                    self.batch.query_lens[index],
                )
                self.masks[index] = mask

        return mask


    def _write_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        from tinyvllm.kernels.kv_store import store_kv

        store_kv(
            self.cache,
            layer_idx,
            self.kv.slot_mapping,
            k,
            v,
        )

    def _attend(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        super()._attend(layer_idx, q, k, v, output, scale=scale)

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


class FlashInferAttention(CudaAttention):
    """Only historical multi-query rows use the runner-owned paged plan."""

    def __init__(self, batch, model_input, cache, prefill) -> None:
        super().__init__(batch, model_input, cache)
        self.prefill = prefill


    def _make_mask(self, before, after, query_len):
        return None


    def _attend(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        super()._attend(layer_idx, q, k, v, output, scale=scale)
        self.prefill.run(layer_idx, q, self.cache, output, scale=scale)


def select_attention_backend(name: str, model: torch.nn.Module):
    """Select once at construction; an explicitly requested backend must work."""

    if name == "reference":
        return ReferenceAttention

    if name not in ("cuda", "flashinfer"):
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

    # Compile/load before allocating KV; CPU/reference users need no toolkit.
    from tinyvllm.kernels.paged_decode import load_paged_decode

    load_paged_decode()

    return FlashInferAttention if name == "flashinfer" else CudaAttention