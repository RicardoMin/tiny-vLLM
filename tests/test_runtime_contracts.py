from __future__ import annotations

import torch

from tinyvllm.batch import PreparedBatch, StepMode
from tinyvllm.request import SamplingParams
from tinyvllm.runtime import ModelInput


def prepared_batch(*, with_kv: bool) -> PreparedBatch:
    return PreparedBatch(
        plan_id=4,
        mode=StepMode.MIXED,
        num_decode_items=1,
        input_ids=(13, 21, 22),
        positions=(4, 0, 1),
        token_request_slots=(2, 0, 0),
        q_offsets=(0, 1, 3),
        context_lens_before=(4, 0),
        query_lens=(1, 2),
        context_lens_after=(5, 2),
        sample_token_indices=(0,),
        sample_request_ids=(99,),
        sampling_params=(SamplingParams(),),
        kv_block_table_ids=(7, 9) if with_kv else (),
        kv_block_table_offsets=(0, 1, 2) if with_kv else (),
        kv_slot_mapping=(19, 24, 25) if with_kv else (),
    )


def test_model_input_lowers_execution_metadata() -> None:
    model_input = ModelInput.from_prepared_batch(
        prepared_batch(with_kv=True),
        device='cpu',
    )

    assert torch.equal(
        model_input.input_ids,
        torch.tensor([13, 21, 22]),
    )
    assert torch.equal(
        model_input.positions,
        torch.tensor([4, 0, 1]),
    )
    assert torch.equal(
        model_input.q_offsets,
        torch.tensor([0, 1, 3], dtype=torch.int32),
    )
    assert torch.equal(
        model_input.sample_token_indices,
        torch.tensor([0]),
    )
    assert model_input.num_decode_items == 1
    assert model_input.token_request_slots.dtype == torch.int32
    assert model_input.sample_token_indices.dtype == torch.long

    assert model_input.kv is not None
    assert torch.equal(
        model_input.kv.block_table_ids,
        torch.tensor([7, 9], dtype=torch.int32),
    )
    assert torch.equal(
        model_input.kv.block_table_offsets,
        torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    assert torch.equal(
        model_input.kv.slot_mapping,
        torch.tensor([19, 24, 25], dtype=torch.int32),
    )


def test_model_input_omits_absent_paged_kv_metadata() -> None:
    model_input = ModelInput.from_prepared_batch(
        prepared_batch(with_kv=False),
        device='cpu',
    )

    assert model_input.kv is None
