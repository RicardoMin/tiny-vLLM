from dataclasses import replace

import pytest
import torch

from tinyvllm.batch import BatchBuilder, StepPlan, WorkItem
from tinyvllm.cache import BlockAppend, BlockManager
from tinyvllm.request import RequestState, SamplingParams, WorkClass
from tinyvllm.runtime.contracts import ModelInput
from tinyvllm.runtime.runner import ModelRunner
from paged_helpers import logical_kv, models, prepare, scenario


def test_paged_mixed_reordered_and_reused_blocks_match_hf_hidden_logits_and_kv():
    model, hf = models()
    manager = BlockManager(block_size=2, num_blocks=12)
    runner = ModelRunner(model, block_size=2, num_blocks=12, max_model_len=32)
    runner.kv_cache.key.fill_(float('nan'))
    runner.kv_cache.value.fill_(float('nan'))
    captured, calls = [], {}
    handles = [model.model.register_forward_hook(lambda m, a, out: captured.append(out.detach().clone()))]
    for name, module in model.named_modules():
        if name.endswith(('qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj')):
            calls[name] = []
            handles.append(module.register_forward_pre_hook(lambda m, a, name=name: calls[name].append(a[0].shape[0])))
    records_list = list(scenario())
    try:
        with torch.inference_mode():
            for step, records in enumerate(records_list):
                txn, batch = prepare(manager, step, records)
                txn.commit_before_dispatch()
                captured.clear()
                for rows in calls.values():
                    rows.clear()
                actual = runner.run(batch)
                assert len(captured) == 1
                assert all(rows == [batch.num_query_tokens] for rows in calls.values())
                expected_hidden, expected_logits = [], []
                for index, (rid, seq, _, sample, _) in enumerate(records):
                    before, after = batch.context_lens_before[index], batch.context_lens_after[index]
                    assert manager.valid_token_count(rid) == before
                    result = hf(torch.tensor([seq[:after]]), use_cache=True, output_hidden_states=True)
                    expected_hidden.append(result.hidden_states[-1][0, before:after])
                    if sample:
                        expected_logits.append(result.logits[0, -1])
                    keys, values = logical_kv(runner, manager, rid, after)
                    for layer, (expected_k, expected_v) in enumerate(result.past_key_values):
                        torch.testing.assert_close(keys[layer], expected_k[0].transpose(0, 1), atol=2e-5, rtol=2e-5)
                        torch.testing.assert_close(values[layer], expected_v[0].transpose(0, 1), atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(captured[0], torch.cat(expected_hidden), atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(actual, torch.stack(expected_logits), atol=2e-5, rtol=2e-5)
                txn.complete_writes()
                if step == 2:
                    table = manager.block_table(10)
                    assert any(b.block_id != a.block_id + 1 for a, b in zip(table, table[1:]))
                if step == 3:
                    old = manager.block_table(22)
                    manager.release_request(22)
                    reused = ((44, (21, 22, 23), 3, True, WorkClass.PREFILL),)
                    records_list.append(reused)
                if step == 4:
                    fresh = manager.block_table(44)
                    assert {r.block_id for r in fresh} <= {r.block_id for r in old}
                    assert min(r.generation for r in fresh) > min(r.generation for r in old)
    finally:
        for handle in handles:
            handle.remove()


def test_partial_prefill_skips_logits_and_runner_does_not_commit_manager(monkeypatch):
    model, _ = models()
    manager = BlockManager(block_size=2, num_blocks=4)
    runner = ModelRunner(model, block_size=2, num_blocks=4)
    txn, batch = prepare(manager, 0, ((1, (1, 2, 3), 2, False, WorkClass.PREFILL),))
    txn.commit_before_dispatch()
    monkeypatch.setattr(model, 'compute_logits', lambda *a: pytest.fail('partial prefill computed logits'))
    assert runner.run(batch) is None
    assert manager.valid_token_count(1) == 0
    txn.complete_writes()
    assert manager.valid_token_count(1) == 2


@pytest.mark.parametrize('updates', [
    {'kv_block_size': 4}, {'kv_num_blocks': 9},
    {'kv_block_table_ids': (-1, 1, 2)}, {'kv_block_table_ids': (0, 1, 8)},
    {'kv_block_table_ids': (0, 1, 1)}, {'kv_block_table_offsets': (0, 1, 3)},
    {'kv_block_table_offsets': ()}, {'kv_slot_mapping': (0, 0, 2, 4, 5)},
    {'kv_slot_mapping': (0, 1, 3, 4, 5)}, {'kv_slot_mapping': (0, 1, 16, 4, 5)},
    {'q_offsets': (0, 0, 5)}, {'positions': (0, 1, 8, 0, 1)},
    {'request_ids': (10, 10)}, {'sample_token_indices': (1,)},
    {'input_ids': (1, 2, 41, 3, 4)},
])
def test_invalid_cpu_metadata_rejected_before_lowering_and_writes(updates, monkeypatch):
    model, _ = models()
    manager = BlockManager(block_size=2, num_blocks=8)
    runner = ModelRunner(model, block_size=2, num_blocks=8)
    txn, batch = prepare(manager, 0, scenario()[0])
    runner.kv_cache.key.fill_(17)
    runner.kv_cache.value.fill_(23)
    monkeypatch.setattr(ModelInput, 'from_prepared_batch', lambda *a, **kw: pytest.fail('invalid batch reached device lowering'))
    with pytest.raises(ValueError):
        runner.run(replace(batch, **updates))
    assert (runner.kv_cache.key == 17).all()
    assert (runner.kv_cache.value == 23).all()
    txn.rollback()
    assert manager.free_block_count == 8


def test_builder_rejects_same_plan_id_with_wrong_append_interval():
    manager = BlockManager(block_size=2, num_blocks=4)
    txn = manager.begin_plan(plan_id=0, appends=(BlockAppend(1, 0, 2),))
    request = RequestState(1, 1, [1, 2, 3], SamplingParams(), 0)
    for start, before in ((1, 1), (0, 1)):
        plan = StepPlan(0, (WorkItem(1, 0, WorkClass.PREFILL, start, 2, before, False),))
        with pytest.raises(ValueError):
            BatchBuilder(max_model_len=32).prepare(plan, (request,), kv=txn)
    txn.rollback()


def test_lowered_indices_are_int64_and_history_is_in_logical_order():
    model, _ = models()
    manager = BlockManager(block_size=2, num_blocks=8)
    txn, batch = prepare(manager, 0, scenario()[0])
    runner = ModelRunner(model, block_size=2, num_blocks=8)
    runner._validate(batch, runner.kv_cache)
    data = ModelInput.from_prepared_batch(batch, device='cpu')
    assert data.kv.slot_mapping.dtype == torch.long
    assert data.kv.slot_mapping.tolist() == list(batch.kv_slot_mapping)
    expected = []
    for rid, length in zip(batch.request_ids, batch.context_lens_after):
        table = txn.staged_block_table(rid)
        expected.append([table[p // 2].block_id * 2 + p % 2 for p in range(length)])
    assert [x.tolist() for x in data.kv.read_slots] == expected
    assert all(x.dtype == torch.long for x in data.kv.read_slots)
    txn.rollback()


def test_builder_rejects_extra_unexecuted_request_in_transaction():
    manager = BlockManager(block_size=2, num_blocks=4)
    txn = manager.begin_plan(plan_id=0, appends=(BlockAppend(1, 0, 1), BlockAppend(2, 0, 1)))
    request = RequestState(1, 1, [1], SamplingParams(), 0)
    plan = StepPlan(0, (WorkItem(1, 0, WorkClass.PREFILL, 0, 1, 0, True),))
    with pytest.raises(ValueError, match="all plan items"):
        BatchBuilder(max_model_len=32).prepare(plan, (request,), kv=txn)
    txn.rollback()
    assert manager.valid_token_count(1) == manager.valid_token_count(2) == 0
