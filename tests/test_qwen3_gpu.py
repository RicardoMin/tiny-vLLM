"""Opt-in real checkpoint test. Set TINYVLLM_MODEL to the local model directory."""
import os

import pytest
import torch

from tinyvllm.cache import BlockManager
from tinyvllm.models.qwen3 import Qwen3ForCausalLM
from tinyvllm.runtime.runner import ModelRunner
from tinyvllm.request import WorkClass
from paged_helpers import DenseAttentionOracle, prepare, scenario, logical_kv


MODEL_PATH = os.environ.get('TINYVLLM_MODEL')


@pytest.mark.skipif(not MODEL_PATH or not torch.cuda.is_available(), reason='TINYVLLM_MODEL and CUDA required')
def test_real_qwen3_bf16_paged_matches_dense_same_batch():
    model = Qwen3ForCausalLM.from_pretrained(MODEL_PATH, device='cuda', dtype=torch.bfloat16)
    manager = BlockManager(block_size=2, num_blocks=16)
    runner = ModelRunner(model, block_size=2, num_blocks=16, max_model_len=32)
    oracle = DenseAttentionOracle()
    batches = list(scenario())
    captured = []
    handle = model.model.register_forward_hook(lambda m, a, out: captured.append(out.clone()))
    samples = 0
    try:
        with torch.inference_mode():
            for step, records in enumerate(batches):
                txn, batch = prepare(manager, step, records)
                txn.commit_before_dispatch()
                captured.clear()
                actual = runner.run(batch)
                paged_hidden = captured[-1]
                expected_hidden = model(torch.tensor(batch.input_ids, device='cuda'),
                                        torch.tensor(batch.positions, device='cuda'),
                                        attention=oracle.for_batch(batch))
                torch.testing.assert_close(paged_hidden, expected_hidden, atol=0, rtol=0)
                expected = model.compute_logits(expected_hidden, torch.tensor(batch.sample_token_indices, device='cuda'))
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for i, rid in enumerate(batch.request_ids):
                    key, value = logical_kv(runner, manager, rid, batch.context_lens_after[i])
                    for layer in range(model.config.num_hidden_layers):
                        expected_k, expected_v = oracle.rows[(rid, layer)]
                        torch.testing.assert_close(key[layer], expected_k, atol=0, rtol=0)
                        torch.testing.assert_close(value[layer], expected_v, atol=0, rtol=0)
                samples += batch.num_samples
                txn.complete_writes()
                if step == 3:
                    old_blocks = {r.block_id for r in manager.block_table(22)}
                    manager.release_request(22)
                    oracle.release(22)
                    batches.append(((44, (21, 22, 23), 3, True, WorkClass.PREFILL),))
                if step == 4:
                    assert {r.block_id for r in manager.block_table(44)} <= old_blocks
        print(f'Real BF16 paged vs dense same batch: {len(batches)} steps, {samples} sample rows; hidden/logits/all-layer KV exact')
    finally:
        handle.remove()
