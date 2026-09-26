import gc
import os
import weakref
from pathlib import Path

import pytest
import torch

from tinyvllm import Engine, EngineConfig, GenerateRequest, SamplingParams


@pytest.mark.cuda
@pytest.mark.model
def test_real_bootstrap_generation_and_kv_release():
    config = EngineConfig(
        model=Path(os.environ['TINYVLLM_MODEL']),
        dtype='bfloat16',
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=128,
        prefill_chunk_size=64,
        prefill_reserve_tokens=16,
        prefill_reserve_requests=1,
    )
    engine = Engine.from_config(config)
    cache = engine.executor.runner.kv_cache
    print('dtype:', cache.dtype, 'blocks:', cache.num_blocks,
          'KV bytes:', 2 * cache.key.numel() * cache.key.element_size())
    allocated_bytes = 2 * cache.key.numel() * cache.key.element_size()
    block_bytes = allocated_bytes // cache.num_blocks
    assert allocated_bytes <= config.kv_cache_memory_bytes
    assert config.kv_cache_memory_bytes - allocated_bytes < block_bytes
    assert cache.dtype is torch.bfloat16
    outputs = engine.generate(tuple(
        GenerateRequest(
            tuple(range(1, prompt_len + 1)),
            SamplingParams(max_new_tokens=8, ignore_eos=True),
        )
        for prompt_len in (1, 80, 130)
    ))
    assert [len(output.output_token_ids) for output in outputs] == [8, 8, 8]
    assert engine.scheduler.block_manager.free_block_count == cache.num_blocks
    assert not engine.has_unfinished()
    print('generation:', [len(output.output_token_ids) for output in outputs],
          'all blocks returned:', engine.scheduler.block_manager.free_block_count)

    cache_ref = weakref.ref(cache)
    key_ref = weakref.ref(cache.key)
    kv_bytes = 2 * cache.key.numel() * cache.key.element_size()
    del cache

    def fail(*args, **kwargs):
        raise RuntimeError('P2-A injected layer failure')

    engine.executor.runner.model.model.layers[-1].forward = fail
    engine.submit((1, 2), SamplingParams(max_new_tokens=1))

    def execute_and_catch():
        try:
            engine.step()
        except RuntimeError as exc:
            assert 'executor failed' in str(exc)
        else:
            raise AssertionError('injected failure was not raised')

    execute_and_catch()
    gc.collect()
    torch.cuda.synchronize()
    assert cache_ref() is None
    assert key_ref() is None
    assert not engine.has_unfinished()
    print('fail-stop actual KV object release: PASS;', kv_bytes, 'bytes no longer retained')
