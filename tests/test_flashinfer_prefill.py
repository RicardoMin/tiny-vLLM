"""Opt-in paged-prefill numerics, planning, and resource lifetime gates."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from tinyvllm.batch import PreparedBatch, StepMode
from tinyvllm.runtime.prefill import PagedPrefill
from attention_helpers import _inputs, _oracle


pytestmark = pytest.mark.flashinfer


def _batch(befores=(63, 31, 0, 18), lengths=(1, 17, 5, 3)):
    """Decode, two continuing chunks, and fresh rows in scattered pages."""
    block_size = 16
    afters = tuple(before + length for before, length in zip(befores, lengths))
    counts = tuple((after + block_size - 1) // block_size for after in afters)
    num_blocks = sum(counts) + 3
    physical = list(range(num_blocks))[::2][::-1] + list(range(num_blocks))[1::2]
    tables, positions, slots = [], [], []
    offsets, table_offsets = [0], [0]

    for before, after, count in zip(befores, afters, counts):
        table = physical[len(tables):len(tables) + count]
        tables.extend(table)
        table_offsets.append(len(tables))
        positions.extend(range(before, after))
        slots.extend(
            table[position // block_size] * block_size + position % block_size
            for position in range(before, after)
        )
        offsets.append(offsets[-1] + after - before)

    decode_count = int(befores[0] > 0 and lengths[0] == 1)
    return PreparedBatch(
        plan_id=0,
        mode=StepMode.MIXED if decode_count else StepMode.PREFILL_ONLY,
        num_decode_items=decode_count,
        input_ids=tuple(range(1, sum(lengths) + 1)),
        positions=tuple(positions),
        q_offsets=tuple(offsets),
        context_lens_before=tuple(befores),
        query_lens=tuple(lengths),
        context_lens_after=afters,
        sample_token_indices=(),
        sample_request_ids=(),
        sampling_params=(),
        kv_block_size=block_size,
        kv_num_blocks=num_blocks,
        kv_block_table_ids=tuple(tables),
        kv_block_table_offsets=tuple(table_offsets),
        kv_slot_mapping=tuple(slots),
        request_ids=tuple(10 + index for index in range(len(lengths))),
    )


def _selected_rows(batch):
    return [
        row
        for index, (before, length) in enumerate(zip(
            batch.context_lens_before, batch.query_lens,
        ))
        if before > 0 and length > 1
        for row in range(*batch.q_offsets[index:index + 2])
    ]


def _state(dtype=torch.bfloat16):
    # Explicit opt-in must fail when CUDA or pinned dependencies are unavailable.
    assert torch.cuda.is_available(), "FlashInfer gate requires CUDA"
    config = SimpleNamespace(
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )
    return PagedPrefill(
        config,
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=dtype,
        block_size=16,
    )


def _case(batch, dtype=torch.bfloat16):
    attention, query, key, value, keys, values = _inputs(
        batch,
        device="cuda",
        dtype=dtype,
        heads=16,
        kv_heads=8,
        head_dim=128,
    )
    cache = attention.cache
    slots = torch.tensor(batch.kv_slot_mapping, device="cuda", dtype=torch.long)
    for layer in range(2):
        cache.key[layer].index_copy_(0, slots, key)
        cache.value[layer].index_copy_(0, slots, value)
    expected = _oracle(batch, query, keys, values, 128**-0.5)
    return query, cache, expected


def _check_selected(batch, output, expected, *, dtype, sentinel=-19):
    selected = torch.tensor(_selected_rows(batch), dtype=torch.long)
    actual = output.float().cpu()
    untouched = torch.ones(batch.num_query_tokens, dtype=torch.bool)
    untouched[selected] = False
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(
        actual[selected], expected[selected], atol=tolerance, rtol=tolerance,
    )
    assert torch.all(actual[untouched] == sentinel)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_scattered_pages_match_fp32_oracle_and_preserve_causality(dtype):
    batch = _batch()
    query, cache, expected = _case(batch, dtype)
    key_before = cache.key.clone()
    value_before = cache.value.clone()
    state = _state(dtype)
    output = torch.full_like(query, -19)

    try:
        state.prepare(batch)
        state.run(0, query, cache, output, scale=128**-0.5)
        _check_selected(batch, output, expected, dtype=dtype)
        torch.testing.assert_close(cache.key, key_before, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(cache.value, value_before, atol=0, rtol=0, equal_nan=True)

        # Last token of one chunk is invisible to its earlier queries and peers.
        changed_row = batch.q_offsets[2] - 1
        cache.value[0, batch.kv_slot_mapping[changed_row]].add_(8)
        changed = torch.full_like(query, -19)
        state.run(0, query, cache, changed, scale=128**-0.5)
        untouched = torch.arange(batch.num_query_tokens, device="cuda") != changed_row
        torch.testing.assert_close(changed[untouched], output[untouched], atol=0, rtol=0)
        assert not torch.equal(changed[changed_row], output[changed_row])
    finally:
        state.close()


@torch.inference_mode()
def test_one_plan_per_step_reuses_compact_buffers_across_layers(monkeypatch):
    batch = _batch()
    query, cache, expected = _case(batch)
    state = _state()
    calls = []
    wrapper_type = type(state.wrapper)
    original_plan = wrapper_type.plan

    def record_plan(wrapper, *args, **kwargs):
        arrays = args[:4]
        assert all(tensor.device.type == "cpu" for tensor in arrays)
        assert all(tensor.dtype == torch.int32 for tensor in arrays)
        calls.append(tuple(tensor.tolist() for tensor in arrays))
        return original_plan(wrapper, *args, **kwargs)

    monkeypatch.setattr(wrapper_type, "plan", record_plan)
    try:
        state.prepare(batch)
        assert len(calls) == 1
        assert calls[0][0] == [0, 17, 20]
        assert calls[0][3] == [16, 5]
        pointers = (state.query.data_ptr(), state.output.data_ptr())
        step_refs = tuple(weakref.ref(tensor) for tensor in (
            state.rows, state.query, state.output,
        ))

        for layer in range(2):
            output = torch.full_like(query, -19)
            state.run(layer, query, cache, output, scale=128**-0.5)
            _check_selected(batch, output, expected, dtype=query.dtype)
            assert (state.query.data_ptr(), state.output.data_ptr()) == pointers
        assert len(calls) == 1

        state.finish()
        assert not state.active
        assert state.rows is state.query is state.output is None
        assert all(reference() is None for reference in step_refs)
    finally:
        state.close()


@torch.inference_mode()
def test_no_sample_steps_replan_and_skip_empty_history_without_stale_rows(monkeypatch):
    batches = (
        _batch(),
        _batch((0, 0), (4, 1)),
        _batch((1025, 17, 0), (9, 15, 3)),
    )
    # Build oracles first; do not pull outputs to CPU between successive plans.
    cases = tuple(_case(batch) for batch in batches)
    state = _state()
    wrapper_type = type(state.wrapper)
    original_plan = wrapper_type.plan
    plan_count = 0

    def count_plan(wrapper, *args, **kwargs):
        nonlocal plan_count
        plan_count += 1
        return original_plan(wrapper, *args, **kwargs)

    monkeypatch.setattr(wrapper_type, "plan", count_plan)
    outputs = []
    try:
        for batch, (query, cache, _) in zip(batches, cases):
            assert batch.num_samples == 0
            state.prepare(batch)
            if not _selected_rows(batch):
                assert state.rows is state.query is state.output is None
            output = torch.full_like(query, -19)
            state.run(0, query, cache, output, scale=128**-0.5)
            outputs.append(output)
            state.finish()
            assert not state.active
            assert state.rows is state.query is state.output is None

        assert plan_count == 2
        for batch, output, (_, _, expected) in zip(batches, outputs, cases):
            _check_selected(batch, output, expected, dtype=output.dtype)
    finally:
        state.close()


@torch.inference_mode()
def test_prepare_rejects_stream_change_before_activating_step():
    state = _state()
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            with pytest.raises(RuntimeError, match="stream"):
                state.prepare(_batch())
        assert not state.active
        assert state.rows is state.query is state.output is None

        # Rejected preparation must leave the original stream usable.
        state.prepare(_batch())
        assert state.active
        state.finish()
    finally:
        state.close()


@pytest.mark.parametrize("stage", ["plan", "run"])
@torch.inference_mode()
def test_failed_plan_or_run_can_release_all_owned_resources(stage, monkeypatch):
    batch = _batch()
    query, cache, _ = _case(batch)
    output = torch.full_like(query, -19)
    state = _state()
    references = [weakref.ref(state.wrapper), weakref.ref(state.workspace)]
    wrapper_type = type(state.wrapper)
    original = getattr(wrapper_type, stage)

    def fail_after_dispatch(wrapper, *args, **kwargs):
        original(wrapper, *args, **kwargs)
        raise RuntimeError("injected FlashInfer failure")

    monkeypatch.setattr(wrapper_type, stage, fail_after_dispatch)

    def execute_and_catch():
        # The helper returns without retaining an exception traceback or wrapper.
        try:
            state.prepare(batch)
            references.extend(weakref.ref(tensor) for tensor in (
                state.rows, state.query, state.output,
            ))
            state.run(0, query, cache, output, scale=128**-0.5)
        except RuntimeError as exc:
            assert "injected FlashInfer failure" in str(exc)
        else:
            pytest.fail("injected failure was not raised")
        finally:
            state.finish()

    try:
        execute_and_catch()
        state.close()
        gc.collect()
        assert not state.active
        assert state.rows is state.query is state.output is None
        assert state.wrapper is state.workspace is None
        assert all(reference() is None for reference in references)
    finally:
        state.close()


@torch.inference_mode()
def test_close_releases_references_even_when_stream_cleanup_raises():
    state = _state()
    state.prepare(_batch())
    references = tuple(weakref.ref(value) for value in (
        state.wrapper, state.workspace, state.rows, state.query, state.output,
    ))
    # Drain real work before simulating an error from synchronization itself.
    state.stream.synchronize()

    class FailingStream:
        def synchronize(self):
            raise RuntimeError("injected cleanup failure")

    state.stream = FailingStream()
    with pytest.raises(RuntimeError, match="injected cleanup failure"):
        state.close()
    gc.collect()
    assert not state.active
    assert state.rows is state.query is state.output is None
    assert state.wrapper is state.workspace is None
    assert all(reference() is None for reference in references)
    state.close()
