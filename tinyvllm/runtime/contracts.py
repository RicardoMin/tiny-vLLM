from __future__ import annotations

from dataclasses import dataclass

import torch

from tinyvllm.batch import PreparedBatch


@dataclass(frozen=True, slots=True)
class PagedKVMetadata:
    """Logical paged-KV metadata lowered for one device batch."""

    page_table_ids: torch.Tensor
    page_table_offsets: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Device-side model inputs derived from one prepared batch."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    token_request_slots: torch.Tensor

    q_offsets: torch.Tensor
    context_lens_before: torch.Tensor
    query_lens: torch.Tensor
    context_lens_after: torch.Tensor

    sample_token_indices: torch.Tensor
    num_decode_items: int
    kv: PagedKVMetadata | None

    @classmethod
    def from_prepared_batch(
        cls,
        batch: PreparedBatch,
        *,
        device: torch.device | str,
    ) -> 'ModelInput':
        index_kwargs = {
            'dtype': torch.int32,
            'device': device,
        }

        kv: PagedKVMetadata | None = None
        if batch.has_kv_metadata:
            kv = PagedKVMetadata(
                page_table_ids=torch.tensor(
                    batch.kv_page_table_ids,
                    **index_kwargs,
                ),
                page_table_offsets=torch.tensor(
                    batch.kv_page_table_offsets,
                    **index_kwargs,
                ),
                slot_mapping=torch.tensor(
                    batch.kv_slot_mapping,
                    **index_kwargs,
                ),
            )

        return cls(
            input_ids=torch.tensor(
                batch.input_ids,
                dtype=torch.long,
                device=device,
            ),
            positions=torch.tensor(
                batch.positions,
                dtype=torch.long,
                device=device,
            ),
            token_request_slots=torch.tensor(
                batch.token_request_slots,
                **index_kwargs,
            ),
            q_offsets=torch.tensor(
                batch.q_offsets,
                **index_kwargs,
            ),
            context_lens_before=torch.tensor(
                batch.context_lens_before,
                **index_kwargs,
            ),
            query_lens=torch.tensor(
                batch.query_lens,
                **index_kwargs,
            ),
            context_lens_after=torch.tensor(
                batch.context_lens_after,
                **index_kwargs,
            ),
            sample_token_indices=torch.tensor(
                batch.sample_token_indices,
                dtype=torch.long,
                device=device,
            ),
            num_decode_items=batch.num_decode_items,
            kv=kv,
        )