"""Opt-in bit-exact KV scatter, layout, mutation and stream contracts."""

from __future__ import annotations


import pytest
import torch

from tinyvllm.kernels.kv_store import store_kv
from tinyvllm.kernels.paged_decode import load_paged_decode
from tinyvllm.paged_kv import PagedKVCache


pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module", autouse=True)
def _load_extension():
    load_paged_decode()


def _cache(dtype=torch.bfloat16, heads=2, dim=16):
    cache = PagedKVCache(
        num_layers=2,
        num_blocks=8,
        block_size=2,
        num_kv_heads=heads,
        head_dim=dim,
        device="cuda",
        dtype=dtype,
    )
    cache.key.fill_(-37)
    cache.value.fill_(-41)
    return cache


def _offset_copy(tensor):
    storage = torch.empty(
        tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype,
    )
    result = storage[1:].view(tensor.shape)
    result.copy_(tensor)
    assert result.data_ptr() % 16 == 2
    return result


def _fill_bits(tensor, seed):
    # Include signed zero, infinities and NaN payloads: this operator copies
    # storage and must not round, normalize or otherwise perform arithmetic.
    bits = (torch.arange(tensor.numel(), dtype=torch.int32) * 73 + seed).to(
        torch.int16,
    )
    special = torch.tensor([0, -32768, 31744, 32257, 32640, 32705], dtype=torch.int16)
    count = min(bits.numel(), special.numel())
    bits[:count] = special[:count]
    tensor.copy_(bits.view(tensor.dtype).reshape(tensor.shape))


def _assert_bits_equal(actual, expected):
    assert torch.equal(
        actual.cpu().contiguous().view(torch.int16),
        expected.cpu().contiguous().view(torch.int16),
    )


@pytest.mark.parametrize(
    "dtype,tokens,heads,dim,layout",
    [
        (torch.bfloat16, 1, 8, 128, "packed"),
        (torch.float16, 5, 8, 128, "packed"),
        (torch.bfloat16, 7, 3, 129, "dim_stride"),
        (torch.float16, 4, 2, 80, "head_stride"),
        (torch.bfloat16, 3, 8, 128, "offset"),
        (torch.float16, 3, 3, 129, "contiguous"),
        (torch.bfloat16, 0, 2, 16, "contiguous"),
    ],
    ids=[
        "bf16-single-token",
        "fp16-packed-qkv-v",
        "bf16-dim-stride-odd-dim",
        "fp16-head-stride",
        "bf16-misaligned-storage",
        "fp16-odd-dim",
        "bf16-empty",
    ],
)
@torch.inference_mode()
def test_store_matches_index_copy_bit_for_bit(dtype, tokens, heads, dim, layout):
    cache = _cache(dtype, heads, dim)
    shape = (tokens, heads, dim)
    key = torch.empty(shape, device="cuda", dtype=dtype)
    value = torch.empty_like(key)

    if layout == "packed":
        width = heads * dim
        projection = torch.empty((tokens, 4 * width), device="cuda", dtype=dtype)
        value = projection[:, 3 * width:].view(shape)
        # A singleton token dimension may get a canonical stride from view().
        if tokens > 1:
            assert value.stride(0) == 4 * width
            assert key.stride(0) != value.stride(0)
    elif layout == "dim_stride":
        key = torch.empty((tokens, heads, dim * 2), device="cuda", dtype=dtype)[..., ::2]
        value = torch.empty((tokens, dim, heads), device="cuda", dtype=dtype).transpose(1, 2)
        assert key.stride() != value.stride()
    elif layout == "head_stride":
        value = torch.empty((tokens, heads * 2, dim), device="cuda", dtype=dtype)[:, ::2]
        assert value.stride(1) == dim * 2
    elif layout == "offset":
        key, value = _offset_copy(key), _offset_copy(value)
        cache.key, cache.value = _offset_copy(cache.key), _offset_copy(cache.value)

    _fill_bits(key, 17)
    _fill_bits(value, 1901)
    slot_ids = [13, 2, 11, 0, 7, 4, 15][:tokens]
    slots = torch.tensor(slot_ids, device="cuda", dtype=torch.int64)
    expected_key, expected_value = cache.key.cpu(), cache.value.cpu()
    expected_key[1].index_copy_(0, slots.cpu(), key.cpu())
    expected_value[1].index_copy_(0, slots.cpu(), value.cpu())

    store_kv(cache, 1, slots, key, value)

    # Comparing the entire allocations checks untouched slots and layer 0 too.
    _assert_bits_equal(cache.key, expected_key)
    _assert_bits_equal(cache.value, expected_value)


@torch.inference_mode()
def test_store_obeys_current_stream_and_event_dependencies():
    cache = _cache()
    key = torch.full((1, 2, 16), -1, device="cuda", dtype=cache.dtype)
    value = torch.full_like(key, -2)
    slots = torch.tensor([1], device="cuda", dtype=torch.int64)
    store_kv(cache, 1, slots, key, value)
    cache.key.fill_(-37)
    cache.value.fill_(-41)
    torch.cuda.synchronize()

    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    ready = torch.cuda.Event()
    with torch.cuda.stream(producer):
        # A mistaken default-stream launch sees stale inputs and stale slots.
        torch.cuda._sleep(50_000_000)
        key.fill_(3)
        value.fill_(7)
        slots.fill_(5)
        store_kv(cache, 1, slots, key, value)
        ready.record()

    with torch.cuda.stream(consumer):
        consumer.wait_event(ready)
        observed_key, observed_value = cache.key.clone(), cache.value.clone()
    consumer.synchronize()

    expected_key = torch.full(cache.key.shape, -37, dtype=cache.dtype)
    expected_value = torch.full(cache.value.shape, -41, dtype=cache.dtype)
    expected_key[1, 5].fill_(3)
    expected_value[1, 5].fill_(7)
    _assert_bits_equal(observed_key, expected_key)
    _assert_bits_equal(observed_value, expected_value)


@pytest.mark.parametrize(
    "invalid",
    [
        "source_rank", "value_shape", "source_dtype", "value_dtype",
        "slots_dtype", "slots_device", "slots_stride", "slots_length",
        "cache_stride", "source_alias", "strided_source_alias", "cache_alias",
    ],
)
@torch.inference_mode()
def test_store_rejects_structural_errors_before_writes(invalid):
    cache = _cache()
    key = torch.ones((2, 2, 16), device="cuda", dtype=cache.dtype)
    value = torch.full_like(key, 2)
    slots = torch.tensor([1, 5], device="cuda", dtype=torch.int64)

    if invalid == "source_rank":
        key = key[0]
    elif invalid == "value_shape":
        value = value[:1]
    elif invalid == "source_dtype":
        key = key.float()
    elif invalid == "value_dtype":
        value = value.half()
    elif invalid == "slots_dtype":
        slots = slots.int()
    elif invalid == "slots_device":
        slots = slots.cpu()
    elif invalid == "slots_stride":
        slots = torch.tensor([1, 99, 5, 99], device="cuda", dtype=torch.int64)[::2]
    elif invalid == "slots_length":
        slots = slots[:1]
    elif invalid == "cache_stride":
        cache.key = torch.full((2, 16, 2, 32), -37, device="cuda", dtype=cache.dtype)[..., ::2]
    elif invalid == "source_alias":
        key = cache.key[1, :2]
    elif invalid == "strided_source_alias":
        key = cache.key[1, ::2][:2]
        slots = torch.tensor([2, 0], device="cuda", dtype=torch.int64)
    elif invalid == "cache_alias":
        cache.value = cache.key

    before_key, before_value = cache.key.cpu(), cache.value.cpu()
    with pytest.raises((ValueError, RuntimeError)):
        store_kv(cache, 1, slots, key, value)
    _assert_bits_equal(cache.key, before_key)
    _assert_bits_equal(cache.value, before_value)


@torch.inference_mode()
def test_store_does_not_read_gpu_scalars():
    cache = _cache()
    key = torch.ones((2, 2, 16), device="cuda", dtype=cache.dtype)
    value = torch.full_like(key, 2)
    slots = torch.tensor([1, 5], device="cuda", dtype=torch.int64)
    store_kv(cache, 1, slots, key, value)
    torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        store_kv(cache, 1, slots, key, value)
    operations = {event.key for event in profile.key_averages()}
    assert "aten::_local_scalar_dense" not in operations
    assert "aten::item" not in operations
