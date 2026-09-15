"""Independent HF oracle and CPU plan fixtures for paged integration tests."""
import torch
from transformers import Qwen3Config as HFConfig, Qwen3ForCausalLM as HFModel

from tinyvllm.batch import BatchBuilder, StepPlan, WorkItem
from tinyvllm.cache import BlockAppend, BlockManager
from tinyvllm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from tinyvllm.request import RequestState, SamplingParams, WorkClass
from tinyvllm.runtime.runner import ModelRunner


def models(dtype=torch.float32, device='cpu'):
    torch.manual_seed(123)
    config = HFConfig(
        vocab_size=41, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=6, max_position_embeddings=32, tie_word_embeddings=True,
        attn_implementation='sdpa',
    )
    hf = HFModel(config).to(device=device, dtype=dtype).eval()
    tiny = Qwen3ForCausalLM(Qwen3Config.from_hf(config), device=device, dtype=dtype).eval()
    for name, weight in hf.state_dict().items():
        tiny.load_weight(name, weight)
    return tiny, hf


def prepare(manager, plan_id, records, *, max_model_len=32):
    """Records: (request_id, complete_sequence, query_len, should_sample, work_class)."""
    requests, items, appends = [], [], []
    for slot, (rid, sequence, query_len, sample, work_class) in enumerate(records):
        before = manager.valid_token_count(rid)
        requests.append(RequestState(rid, rid, list(sequence), SamplingParams(), 0))
        items.append(WorkItem(rid, slot, work_class, before, query_len, before, sample))
        appends.append(BlockAppend(rid, before, query_len))
    plan = StepPlan(plan_id, tuple(items))
    txn = manager.begin_plan(plan_id=plan_id, appends=tuple(appends))
    assert txn is not None
    batch = BatchBuilder(max_model_len=max_model_len).prepare(plan, tuple(requests), kv=txn)
    return txn, batch


def scenario():
    P, D = WorkClass.PREFILL, WorkClass.DECODE
    return (
        ((10, (1, 4, 7), 3, True, P), (22, (2, 3, 5, 8, 9), 2, False, P)),
        ((10, (1, 4, 7, 6), 1, True, D), (22, (2, 3, 5, 8, 9), 2, False, P), (33, (11, 12), 2, True, P)),
        ((33, (11, 12, 13), 1, True, D), (10, (1, 4, 7, 6, 14), 1, True, D), (22, (2, 3, 5, 8, 9), 1, True, P)),
        ((10, (1, 4, 7, 6, 14, 15), 1, True, D), (33, (11, 12, 13, 16), 1, True, D)),
    )


def logical_kv(runner, manager, request_id, length):
    table = manager.block_table(request_id)
    slots = torch.tensor([
        table[p // manager.block_size].block_id * manager.block_size + p % manager.block_size
        for p in range(length)
    ], device=runner.device)
    cache = runner.kv_cache
    return cache.key.index_select(1, slots), cache.value.index_select(1, slots)

class DenseAttentionOracle:
    """Test-only append-by-concatenation; no block IDs, slots or production cache."""
    def __init__(self):
        self.rows = {}

    def for_batch(self, batch):
        import torch.nn.functional as F
        def attend(layer, q, k, v, *, scale):
            outputs = []
            for i, rid in enumerate(batch.request_ids):
                start, end = batch.q_offsets[i:i + 2]
                before = batch.context_lens_before[i]
                new_k, new_v = k[start:end], v[start:end]
                if before:
                    old_k, old_v = self.rows[(rid, layer)]
                    assert old_k.shape[0] == before
                    new_k, new_v = torch.cat((old_k, new_k)), torch.cat((old_v, new_v))
                new_k, new_v = new_k.contiguous(), new_v.contiguous()
                self.rows[(rid, layer)] = new_k, new_v
                # Match SDPA compute mode/strides so this comparison isolates storage.
                # The separate HF FP32 test supplies an independent attention oracle.
                allowed = torch.arange(new_k.shape[0], device=q.device)[None, :] <= torch.arange(before, before + end - start, device=q.device)[:, None]
                out = F.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1)[None], new_k.transpose(0, 1)[None],
                    new_v.transpose(0, 1)[None], attn_mask=allowed if before and end - start > 1 else None,
                    is_causal=before == 0, scale=scale, enable_gqa=q.shape[1] != k.shape[1],
                )
                outputs.append(out[0].transpose(0, 1))
            return torch.cat(outputs)
        return attend

    def release(self, rid):
        for key in tuple(self.rows):
            if key[0] == rid:
                del self.rows[key]
