from __future__ import annotations

import torch
from torch import nn

from tinyvllm._validation import _check_positive_int
from tinyvllm.batch import PreparedBatch
from tinyvllm.contiguous_kv import ContiguousKVCache
from tinyvllm.runtime.attention import ReferenceAttention
from tinyvllm.runtime.contracts import ModelInput


class ModelRunner:
    """Own reference KV state and run one flat batch through the model"""

    def __init__(
        self,
        model: nn.Module,
        *,
        max_model_len: int | None = None,
    ) -> None:
        self.model = model.eval()
        parameter = next(self.model.parameters())
        self.device = parameter.device
        config = self.model.config
        if max_model_len is None:
            max_model_len = config.max_position_embeddings
        _check_positive_int("max_model_len", max_model_len)
        if max_model_len > config.max_position_embeddings:
            raise ValueError(
                "max_model_len exceeds model position capacity"
            )
        self.max_model_len = max_model_len
        self._cache_args = {
            "num_layers": config.num_hidden_layers,
            "max_seq_len": max_model_len,
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "device": self.device,
            "dtype": parameter.dtype,
        }
        self._caches: dict[int, ContiguousKVCache] = {}


    def _validate(self, batch: PreparedBatch) -> None:
        count = batch.num_items
        if count == 0 or len(batch.request_ids) != count:
            raise ValueError(
                "request_ids must align with nonempty batch items"
            )
        if len(set(batch.request_ids)) != count:
            raise ValueError(
                "a request may appear only once in one batch"
            )
        if (
            len(batch.q_offsets) != count + 1
            or batch.q_offsets[0] != 0
            or batch.q_offsets[-1] != batch.num_query_tokens
            or len(batch.context_lens_before) != count
            or len(batch.context_lens_after) != count
            or len(batch.positions) != batch.num_query_tokens
        ):
            raise ValueError(
                "batch token and context metadata must align"
            )
        for index, request_id in enumerate(batch.request_ids):
            start, end = batch.q_offsets[index:index + 2]
            before = batch.context_lens_before[index]
            after = batch.context_lens_after[index]
            if (
                end <= start
                or end - start != batch.query_lens[index]
                or before < 0
                or after != before + end - start
                or after > self.max_model_len
            ):
                raise ValueError("invalid query or context length")
            if batch.positions[start:end] != tuple(range(before, after)):
                raise ValueError(
                    "positions must be contiguous after the cache"
                )
            cache = self._caches.get(request_id)
            actual_before = 0 if cache is None else cache.length
            if actual_before != before:
                raise RuntimeError(
                    "cache length does not match planned context"
                )

        sample_indices = batch.sample_token_indices
        if (
            len(batch.sample_request_ids) != batch.num_samples
            or len(batch.sampling_params) != batch.num_samples
            or any(
                a >= b
                for a, b in zip(sample_indices, sample_indices[1:])
            )
        ):
            raise ValueError("sampling rows must be aligned and ordered")
        boundary_ids = dict(
            zip(
                (end - 1 for end in batch.q_offsets[1:]),
                batch.request_ids,
            )
        )
        for index, request_id in zip(
            sample_indices,
            batch.sample_request_ids,
        ):
            if index not in boundary_ids or boundary_ids[index] != request_id:
                raise ValueError(
                    "sampling rows do not match batch item boundaries"
                )


    @torch.inference_mode()
    def run(self, batch: PreparedBatch) -> torch.Tensor | None:
        self._validate(batch)
        for request_id in batch.request_ids:
            if request_id not in self._caches:
                self._caches[request_id] = ContiguousKVCache(
                    **self._cache_args
                )
        caches = tuple(
            self._caches[request_id]
            for request_id in batch.request_ids
        )
        model_input = ModelInput.from_prepared_batch(
            batch,
            device=self.device,
        )
        attention = ReferenceAttention(
            batch.q_offsets,
            batch.context_lens_before,
            caches,
        )
        hidden_states = self.model(
            model_input.input_ids,
            model_input.positions,
            attention=attention,
        )
        # All layers must write at the old length before committing progress.
        for cache, length in zip(caches, batch.context_lens_after):
            cache.advance(length)
        if batch.num_samples == 0:
            return None
        return self.model.compute_logits(
            hidden_states,
            model_input.sample_token_indices,
        )


    def release(self, request_id: int) -> None:
        self._caches.pop(request_id, None)

    def clear(self) -> None:
        self._caches.clear()