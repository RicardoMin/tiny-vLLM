from __future__ import annotations

import torch

from tinyvllm._validation import _check_positive_int
from tinyvllm.batch import PreparedBatch
from tinyvllm.runtime.contracts import (
    ModelInput,
    PagedKVMetadata,
    SingleQueryMetadata,
)


class DecodeInputs:
    """Runner-owned storage; returned views expire at the next prepare call."""

    @staticmethod
    def device_bytes(
        *,
        max_num_seqs: int,
        max_model_len: int,
        block_size: int,
    ) -> int:
        pages = (max_model_len + block_size - 1) // block_size
        return 8 * (6 * max_num_seqs + 1 + max_num_seqs * pages)

        
    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_model_len: int,
        block_size: int,
        device: torch.device | str,
    ) -> None:
        for name, value in (
            ("max_num_seqs", max_num_seqs),
            ("max_model_len", max_model_len),
            ("block_size", block_size),
        ):
            _check_positive_int(name, value)
            
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("decode inputs require a CUDA device")

        self.capacity = max_num_seqs
        pages = (max_model_len + block_size - 1) // block_size
        self.table_capacity = max_num_seqs * pages
        sizes = (
            max_num_seqs,
            max_num_seqs,
            max_num_seqs,
            max_num_seqs + 1,
            max_num_seqs,
            self.table_capacity,
        )
        self.table_start = sum(sizes[:-1])
        self.stream = torch.cuda.current_stream(self.device)
        self.upload_done = torch.cuda.Event()
        self.uploaded = False
        
        self.host = torch.zeros(sum(sizes), dtype=torch.long, pin_memory=True)
        self.device_data = torch.empty(sum(sizes), dtype=torch.long, device=self.device)
        self.rows = torch.arange(max_num_seqs, dtype=torch.long, device=self.device)
        self.host_views = tuple(part.numpy() for part in self.host.split(sizes))
        self.device_views = self.device_data.split(sizes)
        
        
    def prepare(self, batch: PreparedBatch) -> ModelInput | None:
        """Reuse storage for validated, fully sampled single-query batches."""

        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("CUDA runner must stay on its creation stream")

        count = batch.num_items
        if not 1 <= count <= self.capacity:
            raise ValueError("batch exceeds decode input request capacity")

        if batch.num_query_tokens != count or batch.num_samples != count:
            return None
        
        table_size = len(batch.kv_block_table_ids)
        if table_size > self.table_capacity:
            raise ValueError("batch exceeds decode input block-table capacity")

        # GPU stream ordering cannot protect host memory from an early CPU write.
        if self.uploaded:
            self.upload_done.synchronize()

        # Reset stale rows from a previously larger batch before one H2D copy.
        tokens, positions, slots, offsets, lengths, _ = self.host_views
        
        tokens[count:] = 0
        positions[count:] = 0
        slots[count:] = -1
        offsets[count + 1:] = table_size
        lengths[count:] = 0
            
        values = (
            batch.input_ids,
            batch.positions,
            batch.kv_slot_mapping,
            batch.kv_block_table_offsets,
            batch.context_lens_after,
            batch.kv_block_table_ids,
        )
        for target, source in zip(self.host_views, values):
            target[:len(source)] = source
            
        size = self.table_start + table_size
        self.device_data[:size].copy_(self.host[:size], non_blocking=True)
        self.upload_done.record(self.stream)
        self.uploaded = True

        return self.view(count, table_size)
        

    def view(self, count: int, table_size: int | None = None) -> ModelInput:
        """Borrow fixed storage; graph capture uses the full block-table capacity."""

        tokens, positions, slots, offsets, lengths, blocks = self.device_views
        rows = self.rows[:count]
        return ModelInput(
            input_ids=tokens[:count],
            positions=positions[:count],
            sample_token_indices=rows,
            kv=PagedKVMetadata(
                slot_mapping=slots[:count],
                read_slots=(None,) * count,
                decode=SingleQueryMetadata(
                    query_rows=rows,
                    block_table_ids=blocks[:table_size],
                    block_table_offsets=offsets[:count + 1],
                    context_lens=lengths[:count],
                ),
            ),
        )