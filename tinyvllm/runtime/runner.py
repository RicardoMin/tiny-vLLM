from __future__ import annotations

import torch
from torch import nn

from tinyvllm._validation import _check_positive_int
from tinyvllm.batch import PreparedBatch
from tinyvllm.paged_kv import PagedKVCache
from tinyvllm.runtime.attention import select_attention_backend
from tinyvllm.runtime.contracts import ModelInput
from tinyvllm.runtime.inputs import DecodeInputs
from tinyvllm.runtime.graphs import DecodeGraphs


class ModelRunner:
    """Own one physical KV store; request ownership stays in BlockManager."""

    def __init__(
        self,
        model: nn.Module,
        *,
        block_size: int,
        num_blocks: int,
        max_model_len: int | None = None,
        max_num_seqs: int = 64,
        attention_backend: str = "cuda",
        enforce_eager: bool = False,
    ) -> None:
        if type(enforce_eager) is not bool:
            raise TypeError("enforce_eager must be bool")
        self.model = model.eval()
        self.attention_type = select_attention_backend(
            attention_backend,
            model,
        )
        self.paged_decode = attention_backend != "reference"
        self.prefill = None
        self.decode_inputs = None
        self.decode_graphs = None
        self.kv_cache = None

        parameter = next(self.model.parameters())
        self.device = parameter.device
        config = model.config

        if max_model_len is None:
            max_model_len = config.max_position_embeddings
        _check_positive_int("max_model_len", max_model_len)
        if max_model_len > config.max_position_embeddings:
            raise ValueError(
                "max_model_len exceeds model position capacity"
            )
        self.max_model_len = max_model_len

        if attention_backend == "flashinfer":
            from tinyvllm.runtime.prefill import PagedPrefill

            self.prefill = PagedPrefill(
                config,
                device=self.device,
                dtype=parameter.dtype,
                block_size=block_size,
            )

        try:
            if self.paged_decode:
                self.decode_inputs = DecodeInputs(
                    max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len,
                    block_size=block_size,
                    device=self.device,
                )
            
            self.kv_cache: PagedKVCache | None = PagedKVCache(
                num_layers=config.num_hidden_layers,
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                device=self.device,
                dtype=parameter.dtype,
            )
            
            if self.decode_inputs is not None and not enforce_eager:
                self.decode_graphs = DecodeGraphs(self.decode_inputs.stream)

                self.decode_graphs.capture(
                    model,
                    self.decode_inputs,
                    self.kv_cache,
                )
        except Exception as cause:
            try:
                self.clear()
            except Exception as cleanup_error:
                cause.add_note(f"runner cleanup: {cleanup_error!r}")

            raise


    def _validate(
        self,
        batch: PreparedBatch,
        cache: PagedKVCache,
    ) -> None:
        count = batch.num_items
        tokens = batch.num_query_tokens
        if count == 0 or len(batch.request_ids) != count:
            raise ValueError("request_ids must align with nonempty batch items")
        if len(set(batch.request_ids)) != count:
            raise ValueError("a request may appear only once in one batch")
        if (
            len(batch.q_offsets) != count + 1
            or batch.q_offsets[0] != 0
            or batch.q_offsets[-1] != tokens
            or len(batch.context_lens_before) != count
            or len(batch.context_lens_after) != count
            or len(batch.positions) != tokens
        ):
            raise ValueError("batch token and context metadata must align")
        if (
            batch.kv_block_size != cache.block_size
            or batch.kv_num_blocks != cache.num_blocks
            or len(batch.kv_block_table_offsets) != count + 1
            or batch.kv_block_table_offsets[0] != 0
            or batch.kv_block_table_offsets[-1] != len(batch.kv_block_table_ids)
            or len(batch.kv_slot_mapping) != tokens
        ):
            raise ValueError("paged KV geometry and metadata must align")
        block_ids = batch.kv_block_table_ids
        if any(type(i) is not int or not 0 <= i < cache.num_blocks for i in block_ids):
            raise ValueError("block ID is out of range")
        # No shared blocks or COW in P1, including sharing between requests.
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("block tables must own distinct physical blocks")
        if any(type(i) is not int or not 0 <= i < self.model.config.vocab_size for i in batch.input_ids):
            raise ValueError("input token ID is out of vocabulary range")

        for index in range(count):
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
                raise ValueError("positions must follow the planned KV prefix")
            first, last = batch.kv_block_table_offsets[index:index + 2]
            required = (after + cache.block_size - 1) // cache.block_size
            if first < 0 or last - first != required:
                raise ValueError("block table length does not match context")
            table = block_ids[first:last]
            expected = tuple(
                table[p // cache.block_size] * cache.block_size + p % cache.block_size
                for p in range(before, after)
            )
            if batch.kv_slot_mapping[start:end] != expected:
                raise ValueError("write slots do not match block tables and positions")

        samples = batch.sample_token_indices
        if (
            len(batch.sample_request_ids) != batch.num_samples
            or len(batch.sampling_params) != batch.num_samples
            or any(a >= b for a, b in zip(samples, samples[1:]))
        ):
            raise ValueError("sampling rows must be aligned and ordered")
        boundaries = dict(zip((end - 1 for end in batch.q_offsets[1:]), batch.request_ids))
        for index, request_id in zip(samples, batch.sample_request_ids):
            if index not in boundaries or boundaries[index] != request_id:
                raise ValueError("sampling rows do not match batch item boundaries")


    @torch.inference_mode()
    def run(self, batch: PreparedBatch) -> torch.Tensor | None:
        """Return borrowed logits; consume or clone before the next run."""
        cache = self.kv_cache
        if cache is None:
            raise RuntimeError("runner is closed")

        # All value/range checks happen on CPU, before transfers or KV writes.
        self._validate(batch, cache)
        model_input = (
            self.decode_inputs.prepare(batch)
            if self.decode_inputs is not None
            else None
        )
        if model_input is not None and self.decode_graphs is not None:
            logits = self.decode_graphs.replay(batch.num_items)

            if logits is not None:
                return logits
        if model_input is None:
            model_input = ModelInput.from_prepared_batch(
                batch, 
                device=self.device,
                paged_decode=self.paged_decode,
                paged_prefill=self.prefill is not None,
            )

        try:
            if self.prefill is None:
                attention  = self.attention_type(batch, model_input, cache)
            else:
                self.prefill.prepare(batch)
                attention = self.attention_type(
                    batch, model_input, cache, self.prefill,
                )

            hidden = self.model(
                model_input.input_ids,
                model_input.positions,
                attention=attention,
            )
        finally:
            if self.prefill is not None:
                self.prefill.finish()

        if batch.num_samples == 0:
            return None

        return self.model.compute_logits(
            hidden, model_input.sample_token_indices,
        )


    def clear(self) -> None:
        # Final resource disposal; ordinary completion only recycles blocks.
        try:
            if self.decode_graphs is not None:
                self.decode_graphs.close()
        finally:
            try:
                if self.prefill is not None:
                    self.prefill.close()
            finally:
                self.decode_graphs = None
                self.prefill = None
                self.decode_inputs = None
                self.kv_cache = None
                self.model = None