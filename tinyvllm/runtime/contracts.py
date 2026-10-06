from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from tinyvllm.batch import PreparedBatch


class AttentionBackend(Protocol):
    """Per-step attention over flat, normalized and rotated Q/K/V."""

    def __call__(
        self,
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Return attention output shaped [tokens, query_heads, head_dim]."""
        ...


@dataclass(frozen=True, slots=True)
class SingleQueryMetadata:
    """Single-query tables; context_len=0 denotes internal graph padding."""

    query_rows: torch.Tensor
    block_table_ids: torch.Tensor
    block_table_offsets: torch.Tensor
    context_lens: torch.Tensor


@dataclass(frozen=True, slots=True)
class PagedKVMetadata:
    """Only the selected attention path's read metadata is materialized."""

    slot_mapping: torch.Tensor
    read_slots: tuple[torch.Tensor | None, ...]
    decode: SingleQueryMetadata | None = None


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Device inputs shared by all layers of one flat model forward."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    sample_token_indices: torch.Tensor
    kv: PagedKVMetadata

    @classmethod
    def from_prepared_batch(
        cls,
        batch: PreparedBatch,
        *,
        device: torch.device | str,
        paged_decode: bool = False,
        paged_prefill: bool = False,
    ) -> 'ModelInput':
        """Lower an already validated CPU batch once, not once per layer."""
        if not batch.has_kv_metadata or batch.kv_block_size <= 0:
            raise ValueError("model input requires paged KV metadata")

        if paged_prefill and not paged_decode:
            raise ValueError("paged prefill requires CUDA decode metadata")

        kwargs = {
            'dtype': torch.long,
            'device': device,
        }
        read_slots = []
        query_rows, block_ids, table_offsets, context_lens = [], [], [0], []
        gather_tables = None

        if not paged_decode or ( 
            not paged_prefill
            and any(
                before > 0 and query_len > 1
                for before, query_len in zip(
                    batch.context_lens_before,
                    batch.query_lens,
                )
            )
        ):
            reference_tables = torch.tensor(
                batch.kv_block_table_ids,
                **kwargs,
            )

        for index, length in enumerate(batch.context_lens_after):
            first, last = batch.kv_block_table_offsets[index: index + 2]
            table = batch.kv_block_table_ids[first:last]

            if paged_decode and batch.query_lens[index] == 1:
                query_rows.append(batch.q_offsets[index])
                block_ids.extend(table)
                table_offsets.append(len(block_ids))
                context_lens.append(length)
                read_slots.append(None)
            elif paged_decode and (
                batch.context_lens_before[index] == 0 or paged_prefill
            ):
                # Fresh queries use current K/V; paged prefill reads blocks.
                read_slots.append(None)
            else:
                # Gather histor:28
                # ical K/V in logical order for SDPA.
                positions = torch.arange(length, **kwargs)
                physical = reference_tables[first:last].index_select(
                    0, positions // batch.kv_block_size,
                )
                read_slots.append(
                    physical * batch.kv_block_size + positions % batch.kv_block_size
                )

        decode = None
        if query_rows:
            decode = SingleQueryMetadata(
                query_rows=torch.tensor(query_rows, **kwargs),
                block_table_ids=torch.tensor(block_ids, **kwargs),
                block_table_offsets=torch.tensor(table_offsets, **kwargs),
                context_lens=torch.tensor(context_lens, **kwargs),
            )

        return cls(
            input_ids=torch.tensor(
                batch.input_ids,
                **kwargs,
            ),
            positions=torch.tensor(
                batch.positions,
                **kwargs,
            ),
            sample_token_indices=torch.tensor(
                batch.sample_token_indices,
                **kwargs,
            ),
            kv=PagedKVMetadata(
                slot_mapping=torch.tensor(batch.kv_slot_mapping, **kwargs),
                read_slots=tuple(read_slots),
                decode=decode,
            ),
        )
