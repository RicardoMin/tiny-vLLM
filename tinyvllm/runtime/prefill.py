from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import torch

from tinyvllm.batch import PreparedBatch
from tinyvllm.paged_kv import PagedKVCache


# FA2: 32 MiB float workspace + 8 MiB int workspace + 128 KiB KV lengths.
PREFILL_DEVICE_BYTES = 40 * 1024**2 + 128 * 1024


class PagedPrefill:
    """Runner-owned FA2 workspace; one CPU plan reused by all model layers."""

    def __init__(self,
        config,
        *,
        device,
        dtype,
        block_size: int
    ) -> None:
        if config.head_dim != 128 or block_size != 16:
            raise ValueError(
                "flashinfer currently requires head_dim=128 and block_size=16"
            )

        try:
            installed = version("flashinfer-python")
        except PackageNotFoundError as exc:
            raise RuntimeError("install tinyvllm[flashinfer] to use this backend") from exc 

        if installed != "0.3.1":
            raise RuntimeError("install the pinned tinyvllm[flashinfer] dependencies")

        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        self.device = torch.device(device)

        if self.device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("flashinfer requires CUDA float16 or bfloat16")

        self.dtype = dtype
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = config.head_dim
        self.block_size = block_size
        self.scale = self.dim**-0.5

        self.stream = torch.cuda.current_stream(self.device)
        self.plan_done = torch.cuda.Event()
        self.planned = False
        self.active = False

        self.rows = self.query = self.output = None
        self.wrapper = self.workspace = None

        # Compile the specialization before the first measured model step.
        try:
            self.workspace = torch.empty(
                32 * 1024**2, device=self.device, dtype=torch.uint8,
            )
            self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.workspace, "NHD", backend="fa2",
            )
            self._plan(([0,2], [0,1], [0], [2]))
        except Exception as cause:
            try:
                self.close()
            except Exception as cleanup_error:
                cause.add_note(f"paged prefill cleanup: {cleanup_error!r}")
                
            raise


    def _plan(self, arrays) -> None:
        metadata = tuple(
            torch.tensor(values, dtype=torch.int32, device="cpu")
            for values in arrays
        )

        try:
            self.wrapper.plan(
                *metadata,
                self.heads,
                self.kv_heads,
                self.dim,
                self.block_size,
                causal=True,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                use_fp16_qk_reduction=False,
                sm_scale=self.scale,
            )
        finally:
            # The planner asynchronously uploads a reusable pinned CPU buffer.
            self.plan_done.record(self.stream)
            self.planned = True


    def prepare(self, batch: PreparedBatch) -> None:
        if self.wrapper is None:
            raise RuntimeError("paged prefill is closed")

        if self.active:
            raise RuntimeError("paged prefill step is already acitve")

        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("flashinfer runner must stay on its creation stream")

        if (
            batch.num_query_tokens >= 2**31
            or batch.kv_num_blocks >= 2**31
            or max(batch.context_lens_after) >= 2**31
            or len(batch.kv_block_table_ids) >= 2**31
            or batch.num_items > 32768
        ):
            raise ValueError("batch exceeds FlashInfer int32 or request capacity")

        rows, qo_offsets, block_ids, block_offsets, last_lens = [], [0], [], [0], []

        lengths = zip(
            batch.context_lens_before,
            batch.query_lens,
            batch.context_lens_after,
        )

        for index, (before, query_len, after) in enumerate(lengths):
            if before == 0 or query_len == 1:
                continue

            start, end = batch.q_offsets[index:index + 2]
            first, last = batch.kv_block_table_offsets[index:index + 2]

            rows.extend(range(start, end))
            qo_offsets.append(len(rows))
            block_ids.extend(batch.kv_block_table_ids[first:last])
            block_offsets.append(len(block_ids))
            last_lens.append((after - 1) % self.block_size + 1)

        self.active = True

        if not rows:
            return

        # CPU reuse needs H2D completion; same-stream ordering alone is insufficient.
        if self.planned:
            self.plan_done.synchronize()

        self._plan((qo_offsets, block_offsets, block_ids, last_lens))

        self.rows = torch.tensor(rows, device=self.device, dtype=torch.long)
        shape = (len(rows), self.heads, self.dim)

        self.query = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.output = torch.empty_like(self.query)


    def run(
        self,
        layer_idx: int,
        q: torch.Tensor,
        cache: PagedKVCache,
        output: torch.Tensor,
        *,
        scale: float,
    ) -> None:
        if not self.active:
            raise RuntimeError("paged prefill requires an active step")

        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("one model step must stay on its preparation stream")

        if self.rows is None:
            return

        if scale != self.scale:
            raise ValueError("attention scale differs from the prefill plan")

        shape = (cache.num_blocks, self.block_size, self.kv_heads, self.dim)

        torch.index_select(q, 0, self.rows, out=self.query)

        self.wrapper.run(
            self.query,
            (cache.key[layer_idx].view(shape), cache.value[layer_idx].view(shape)),
            out=self.output,
            enable_pdl=False,
        )

        output.index_copy_(0, self.rows, self.output)


    def finish(self) -> None:
        # These tensors are allocated and used exclusively on the bound stream.
        self.active = False
        self.rows = self.query = self.output = None


    def close(self) -> None:
        try:
            if self.wrapper is not None or self.workspace is not None:
                self.stream.synchronize()
        finally:
            self.finish()
            self.wrapper = self.workspace = None