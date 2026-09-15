import pytest
import torch

from tinyvllm.engine import Engine, GenerateRequest
from tinyvllm.executor import EagerExecutor
from tinyvllm.request import SamplingParams
from tinyvllm.scheduler import Scheduler
from paged_helpers import models


def engine_for(model, *, num_blocks=8):
    scheduler = Scheduler(
        max_active_requests=3, max_step_requests=3, max_scheduled_tokens=4,
        prefill_chunk_size=2, max_model_len=16,
        prefill_reserve_tokens=1, prefill_reserve_requests=1,
        kv_block_size=2, kv_num_blocks=num_blocks,
    )
    executor = EagerExecutor(model, block_size=scheduler.kv_block_size,
                             num_blocks=scheduler.block_manager.pool.num_blocks,
                             max_model_len=scheduler.max_model_len)
    return Engine(scheduler, executor, clock=lambda: 0)


@pytest.mark.parametrize('count', [1, 3])
def test_engine_paged_generation_matches_hf_and_returns_blocks(count):
    model, hf = models()
    engine = engine_for(model, num_blocks=16)
    prompts = ((1, 4, 7), (11, 12), (2, 3, 5, 6, 8))[:count]
    output = engine.generate(tuple(GenerateRequest(p, SamplingParams(max_new_tokens=3)) for p in prompts))
    with torch.inference_mode():
        for prompt, actual in zip(prompts, output):
            sequence = list(prompt)
            for _ in range(3):
                sequence.append(hf(torch.tensor([sequence])).logits[0, -1].argmax().item())
            assert actual.output_token_ids == tuple(sequence[len(prompt):])
    assert engine.scheduler.block_manager.free_block_count == 16
    assert not engine.has_unfinished()


def test_cancellation_recycles_pool_and_preserves_storage_allocation():
    model, _ = models()
    engine = engine_for(model)
    store = engine.executor.runner.kv_cache
    key_ptr, value_ptr = store.key.data_ptr(), store.value.data_ptr()
    rid = engine.submit((1, 2), SamplingParams(max_new_tokens=8))
    engine.step(now_ns=0)
    used = engine.scheduler.block_manager.block_table(rid)
    assert engine.cancel(rid)
    assert engine.step(now_ns=1).completion_events
    assert engine.scheduler.block_manager.free_block_count == 8
    new = engine.submit((3, 4), SamplingParams(max_new_tokens=8), arrival_ns=1)
    engine.step(now_ns=1)
    fresh = engine.scheduler.block_manager.block_table(new)
    assert fresh[0].block_id == used[0].block_id
    assert fresh[0].generation > used[0].generation
    assert (store.key.data_ptr(), store.value.data_ptr()) == (key_ptr, value_ptr)


@pytest.mark.parametrize('point', ['attention', 'last_layer', 'logits', 'sampler'])
def test_partial_device_failure_fail_stops_and_releases_store(point, monkeypatch):
    model, _ = models()
    engine = engine_for(model)
    engine.submit((1, 2), SamplingParams(max_new_tokens=8))
    engine.step(now_ns=0)
    def fail(*args, **kwargs):
        raise RuntimeError('injected device-stage failure')
    if point == 'attention':
        from tinyvllm.runtime.attention import ReferenceAttention
        original = ReferenceAttention.__call__
        def fail_after_write(self, *args, **kwargs):
            original(self, *args, **kwargs)
            fail()
        monkeypatch.setattr(ReferenceAttention, '__call__', fail_after_write)
    elif point == 'last_layer':
        monkeypatch.setattr(model.model.layers[-1], 'forward', fail)
    elif point == 'logits':
        monkeypatch.setattr(model, 'compute_logits', fail)
    else:
        monkeypatch.setattr(engine.executor.sampler, 'sample', fail)
    with pytest.raises(RuntimeError, match='executor failed'):
        engine.step(now_ns=1)
    assert engine.executor.runner.kv_cache is None
    assert engine.scheduler.block_manager.free_block_count == 8
    with pytest.raises(RuntimeError, match='engine is failed'):
        engine.step(now_ns=2)
