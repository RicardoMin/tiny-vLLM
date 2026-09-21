"""CPU gates for optional paged prefill metadata, configuration and budgeting."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tinyvllm.batch import PreparedBatch, StepMode
from tinyvllm.bootstrap import _plan_kv_blocks, create_engine
from tinyvllm.config import EngineConfig
from tinyvllm.request import SamplingParams
from tinyvllm.runtime import prefill as prefill_module
from tinyvllm.runtime.attention import (
    CudaAttention,
    FlashInferAttention,
    ReferenceAttention,
    select_attention_backend,
)
from tinyvllm.runtime.contracts import ModelInput
from tinyvllm.runtime.prefill import PREFILL_DEVICE_BYTES, PagedPrefill
from test_bootstrap import mocked_cuda_loader


MiB = 1024**2
GiB = 1024**3


def _mixed_batch():
    # Decode, exact-page chunk, fresh prefill, crossing-page chunk, one-token prompt.
    return PreparedBatch(
        plan_id=0,
        mode=StepMode.MIXED,
        num_decode_items=1,
        input_ids=tuple(range(1, 11)),
        positions=(16, 14, 15, 0, 1, 2, 30, 31, 32, 0),
        token_request_slots=(0, 1, 1, 2, 2, 2, 3, 3, 3, 4),
        q_offsets=(0, 1, 3, 6, 9, 10),
        context_lens_before=(16, 14, 0, 30, 0),
        query_lens=(1, 2, 3, 3, 1),
        context_lens_after=(17, 16, 3, 33, 1),
        sample_token_indices=(0, 2, 5, 8, 9),
        sample_request_ids=(10, 20, 30, 40, 50),
        sampling_params=(SamplingParams(),) * 5,
        kv_block_size=16,
        kv_num_blocks=8,
        kv_block_table_ids=(4, 1, 6, 2, 7, 0, 5, 3),
        kv_block_table_offsets=(0, 2, 3, 4, 7, 8),
        kv_slot_mapping=(16, 110, 111, 32, 33, 34, 14, 15, 80, 48),
        request_ids=(10, 20, 30, 40, 50),
    )


def test_paged_prefill_lowering_omits_gathers_and_preserves_decode(monkeypatch):
    batch = _mixed_batch()
    original_tensor = torch.tensor

    def checked_tensor(data, *args, **kwargs):
        assert data is not batch.kv_block_table_ids, "copied reference read tables"
        return original_tensor(data, *args, **kwargs)

    def forbidden_arange(*args, **kwargs):
        pytest.fail("direct paged prefill materialized gather positions")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", checked_tensor)
        patch.setattr(torch, "arange", forbidden_arange)
        lowered = ModelInput.from_prepared_batch(
            batch, device="cpu", paged_decode=True, paged_prefill=True,
        )

    assert lowered.kv.read_slots == (None,) * 5
    assert lowered.kv.slot_mapping.tolist() == list(batch.kv_slot_mapping)
    assert lowered.input_ids.tolist() == list(batch.input_ids)
    assert lowered.positions.tolist() == list(batch.positions)
    assert lowered.sample_token_indices.tolist() == [0, 2, 5, 8, 9]

    decode = lowered.kv.decode
    assert decode is not None
    assert decode.query_rows.tolist() == [0, 9]
    assert decode.block_table_ids.tolist() == [4, 1, 3]
    assert decode.block_table_offsets.tolist() == [0, 2, 3]
    assert decode.context_lens.tolist() == [17, 1]
    for tensor in (
        decode.query_rows, decode.block_table_ids,
        decode.block_table_offsets, decode.context_lens,
        lowered.kv.slot_mapping,
    ):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.long


def test_paged_prefill_cannot_disable_the_single_query_path():
    with pytest.raises(ValueError, match="paged prefill requires"):
        ModelInput.from_prepared_batch(
            _mixed_batch(), device="cpu", paged_prefill=True,
        )


def test_existing_backends_keep_their_read_contracts():
    batch = _mixed_batch()
    logical = [
        list(range(64, 80)) + [16],
        list(range(96, 112)),
        [32, 33, 34],
        list(range(112, 128)) + list(range(16)) + [80],
        [48],
    ]
    reference = ModelInput.from_prepared_batch(batch, device="cpu")
    assert reference.kv.decode is None
    assert [slots.tolist() for slots in reference.kv.read_slots] == logical

    cuda = ModelInput.from_prepared_batch(batch, device="cpu", paged_decode=True)
    assert [None if slots is None else slots.tolist() for slots in cuda.kv.read_slots] == [
        None, logical[1], None, logical[3], None,
    ]


@pytest.fixture
def cpu_planner(monkeypatch):
    # Exercise the real packing method; replace only GPU construction and planning.
    planner = object.__new__(PagedPrefill)
    planner.__dict__.update(
        wrapper=object(), active=False, planned=False, stream=None,
        device=torch.device("cpu"), dtype=torch.float32,
        block_size=16, heads=4, dim=128,
        rows=None, query=None, output=None,
    )
    planner._plan = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    return planner


def test_history_chunk_plan_packs_exact_rows_pages_and_tail_lengths(cpu_planner):
    cpu_planner.prepare(_mixed_batch())

    cpu_planner._plan.assert_called_once_with((
        [0, 2, 5],
        [0, 1, 4],
        [6, 7, 0, 5],
        [16, 1],
    ))
    assert cpu_planner.rows.tolist() == [1, 2, 6, 7, 8]
    assert cpu_planner.rows.dtype == torch.long
    assert cpu_planner.query.shape == cpu_planner.output.shape == (5, 4, 128)
    assert cpu_planner.query.data_ptr() != cpu_planner.output.data_ptr()


def test_fresh_and_decode_step_needs_no_prefill_plan_or_scratch(cpu_planner):
    batch = replace(
        _mixed_batch(),
        positions=(16, 0, 1, 0, 1, 2, 0, 1, 2, 0),
        context_lens_before=(16, 0, 0, 0, 0),
        context_lens_after=(17, 2, 3, 3, 1),
        kv_block_table_ids=(4, 1, 6, 2, 7, 3),
        kv_block_table_offsets=(0, 2, 3, 4, 5, 6),
        kv_slot_mapping=(16, 96, 97, 32, 33, 34, 112, 113, 114, 48),
    )
    cpu_planner.prepare(batch)
    cpu_planner._plan.assert_not_called()
    assert cpu_planner.rows is cpu_planner.query is cpu_planner.output is None


@pytest.mark.parametrize(
    "changes",
    [
        {"kv_num_blocks": 2**31},
        {"context_lens_after": (17, 16, 3, 2**31, 1)},
        {"query_lens": (1,) * 32769},
    ],
    ids=["page-index-overflow", "context-index-overflow", "request-capacity"],
)
def test_plan_capacity_rejected_before_allocation(cpu_planner, changes, monkeypatch):
    def forbidden_tensor(*args, **kwargs):
        pytest.fail("invalid metadata reached tensor allocation")

    monkeypatch.setattr(torch, "tensor", forbidden_tensor)
    with pytest.raises(ValueError, match="int32 or request capacity"):
        cpu_planner.prepare(replace(_mixed_batch(), **changes))
    cpu_planner._plan.assert_not_called()
    assert not cpu_planner.active


@pytest.mark.parametrize("head_dim,block_size", [(64, 16), (128, 8)])
def test_unsupported_geometry_fails_before_dependency_import(
    head_dim, block_size, monkeypatch,
):
    def forbidden_version(*args, **kwargs):
        pytest.fail("unsupported geometry reached dependency loading")

    monkeypatch.setattr(prefill_module, "version", forbidden_version)
    with pytest.raises(ValueError, match="head_dim=128 and block_size=16"):
        PagedPrefill(
            SimpleNamespace(head_dim=head_dim),
            device="cpu", dtype=torch.bfloat16, block_size=block_size,
        )


def test_flashinfer_is_an_explicit_optional_backend(tmp_path):
    assert EngineConfig(model=tmp_path).attention_backend == "reference"
    for backend in ("reference", "cuda", "flashinfer"):
        assert EngineConfig(model=tmp_path, attention_backend=backend).attention_backend == backend
    with pytest.raises(ValueError, match="attention backend"):
        EngineConfig(model=tmp_path, attention_backend="auto")


def test_backend_selection_keeps_reference_lazy_and_selects_explicit_classes(monkeypatch):
    load = Mock()
    monkeypatch.setattr("tinyvllm.kernels.paged_decode.load_paged_decode", load)
    assert select_attention_backend("reference", None) is ReferenceAttention
    load.assert_not_called()

    # Device metadata is sufficient: constructing a CUDA tensor is unnecessary.
    parameter = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16)
    model = SimpleNamespace(
        parameters=lambda: iter((parameter,)),
        config=SimpleNamespace(
            head_dim=128, num_attention_heads=16, num_key_value_heads=8,
        ),
    )
    assert select_attention_backend("cuda", model) is CudaAttention
    assert select_attention_backend("flashinfer", model) is FlashInferAttention
    assert load.call_count == 2


@pytest.mark.parametrize("reserve", [0, 40 * MiB + 128 * 1024])
def test_reserved_workspace_preserves_explicit_kv_budget_and_usage_ceiling(reserve):
    assert PREFILL_DEVICE_BYTES == 40 * MiB + 128 * 1024
    arguments = dict(
        total_bytes=4 * GiB,
        free_bytes=2 * GiB + reserve,
        utilization=0.75,
        requested_bytes=GiB + 123,
        bytes_per_block=MiB,
        reserved_bytes=reserve,
    )
    # The 123-byte remainder does not allocate another block; reserve fits exactly.
    assert _plan_kv_blocks(**arguments) == 1024

    arguments["free_bytes"] -= 1
    with pytest.raises(ValueError, match="GPU usage ceiling"):
        _plan_kv_blocks(**arguments)


def test_workspace_reservation_is_opt_in_and_cannot_be_negative():
    arguments = dict(
        total_bytes=4 * GiB,
        free_bytes=2 * GiB,
        utilization=0.75,
        requested_bytes=GiB,
        bytes_per_block=MiB,
    )
    assert _plan_kv_blocks(**arguments) == 1024
    with pytest.raises(ValueError, match="GPU usage ceiling"):
        _plan_kv_blocks(**arguments, reserved_bytes=PREFILL_DEVICE_BYTES)
    with pytest.raises(ValueError, match="nonnegative"):
        _plan_kv_blocks(**arguments, reserved_bytes=-1)


@pytest.mark.parametrize("backend", ["reference", "cuda", "flashinfer"])
def test_bootstrap_budgets_fixed_workspace_and_compact_qo_before_kv_allocation(
    backend, tmp_path, mocked_cuda_loader,
):
    model, _, executor_factory, snapshot = mocked_cuda_loader
    model.config.num_attention_heads = 16
    config = EngineConfig(
        model=tmp_path,
        attention_backend=backend,
        dtype="float16",
        max_num_batched_tokens=512,
        prefill_chunk_size=256,
        gpu_memory_utilization=0.75,
        kv_cache_memory_bytes=2 * MiB,
    )
    # Qwen3: one KV block is 1.75 MiB; 512 rows of Q and O require 4 MiB.
    kv_bytes = 1792 * 1024
    reserve = 44 * MiB + 128 * 1024 if backend == "flashinfer" else 0
    snapshot.return_value = (2 * GiB + kv_bytes + reserve, 8 * GiB)

    engine = create_engine(config)
    assert engine.scheduler.block_manager.pool.num_blocks == 1
    assert executor_factory.call_args.kwargs["attention_backend"] == backend

    # One byte below the required headroom must fail before allocating an executor.
    executor_factory.reset_mock()
    snapshot.return_value = (2 * GiB + kv_bytes + reserve - 1, 8 * GiB)
    with pytest.raises(ValueError, match="GPU usage ceiling"):
        create_engine(config)
    executor_factory.assert_not_called()
