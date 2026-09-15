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
class PagedKVMetadata:
    """Logical paged-KV metadata lowered for one device batch."""

    slot_mapping: torch.Tensor
    read_slots: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Only tensors consumed by the reference backend; offsets stay on CPU."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    sample_token_indices: torch.Tensor
    kv: PagedKVMetadata | None

    @classmethod
    def from_prepared_batch(
        cls,
        batch: PreparedBatch,
        *,
        device: torch.device | str,
    ) -> 'ModelInput':
        """Lower an already validated CPU batch once, not once per layer."""
        if not batch.has_kv_metadata or batch.kv_block_size <= 0:
            raise ValueError("model input requires paged KV metadata")
        kwargs = {
            'dtype': torch.long,
            'device': device,
        }
        block_ids = torch.tensor(batch.kv_block_table_ids, **kwargs)
        read_slots = []

        for index, length in enumerate(batch.context_lens_after):
            first, last = batch.kv_block_table_offsets[index: index + 2]
            positions = torch.arange(length, **kwargs)
            physical = block_ids[first:last].index_select(0, positions // batch.kv_block_size)
            read_slots.append(physical * batch.kv_block_size + positions % batch.kv_block_size)

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
            ),
        )
