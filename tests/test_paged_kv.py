import pytest
import torch

from tinyvllm.paged_kv import PagedKVCache


def store(device='cpu'):
    return PagedKVCache(num_layers=2, num_blocks=4, block_size=2,
                        num_kv_heads=1, head_dim=2, device=device, dtype=torch.float32)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))])
def test_noncontiguous_write_read_and_layer_isolation(device):
    cache = store(device)
    cache.key.fill_(float('nan'))
    cache.value.fill_(float('nan'))
    slots = torch.tensor([6, 1, 4], device=device)
    key = torch.arange(6, device=device, dtype=torch.float32).reshape(3, 1, 2)
    value = key + 10
    cache.write(1, slots, key, value)
    actual_k, actual_v = cache.read(1, slots.flip(0))
    torch.testing.assert_close(actual_k, key.flip(0))
    torch.testing.assert_close(actual_v, value.flip(0))
    assert cache.key[0].isnan().all()
    assert cache.key[1, 0].isnan().all()
    assert cache.device == cache.key.device


@pytest.mark.parametrize('kind', ['layer', 'shape', 'index_dtype', 'kv_dtype'])
def test_storage_rejects_structural_errors_before_copy(kind):
    cache = store()
    cache.key.zero_()
    slots = torch.tensor([0])
    rows = torch.ones(1, 1, 2)
    if kind == 'index_dtype':
        slots = slots.int()
    if kind == 'shape':
        rows = torch.ones(2, 1, 2)
    if kind == 'kv_dtype':
        rows = rows.double()
    with pytest.raises(ValueError):
        cache.write(2 if kind == 'layer' else 0, slots, rows, rows)
    assert (cache.key == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_storage_operations_do_not_read_gpu_scalars():
    cache = store('cuda')
    slots = torch.tensor([1, 5], device='cuda')
    rows = torch.ones(2, 1, 2, device='cuda')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        cache.write(0, slots, rows, rows)
        cache.read(0, slots)
    assert 'aten::_local_scalar_dense' not in {item.key for item in profile.key_averages()}
