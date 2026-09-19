"""Paged-decode contracts and an opt-in independent CUDA numerical gate."""

import math
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tinyvllm.batch import PreparedBatch, StepMode
from tinyvllm.config import EngineConfig
from tinyvllm.paged_kv import PagedKVCache
from tinyvllm.request import SamplingParams
from tinyvllm.runtime.contracts import ModelInput, SingleQueryMetadata
from tinyvllm.runtime.runner import ModelRunner


def _batch(query_lens=(1, 3, 1), context_lens=(17, 5, 1)):
    """The last item is a one-token prefill, not a decode-class request."""
    block_size = 16
    block_tables = ((4, 1), (6,), (3,))
    q_offsets = [0]
    table_offsets = [0]
    positions = []
    slots = []
    token_request_slots = []

    for index, (query_len, after, table) in enumerate(
        zip(query_lens, context_lens, block_tables)
    ):
        before = after - query_len
        positions.extend(range(before, after))
        slots.extend(
            table[position // block_size] * block_size + position % block_size
            for position in range(before, after)
        )
        token_request_slots.extend([index] * query_len)
        q_offsets.append(q_offsets[-1] + query_len)
        table_offsets.append(table_offsets[-1] + len(table))

    return PreparedBatch(
        plan_id=0,
        mode=StepMode.MIXED if query_lens[0] == 1 else StepMode.PREFILL_ONLY,
        num_decode_items=int(query_lens[0] == 1),
        input_ids=tuple(range(1, q_offsets[-1] + 1)),
        positions=tuple(positions),
        token_request_slots=tuple(token_request_slots),
        q_offsets=tuple(q_offsets),
        context_lens_before=tuple(
            after - query_len
            for after, query_len in zip(context_lens, query_lens)
        ),
        query_lens=query_lens,
        context_lens_after=context_lens,
        sample_token_indices=tuple(end - 1 for end in q_offsets[1:]),
        sample_request_ids=(10, 20, 30),
        sampling_params=(SamplingParams(),) * 3,
        kv_block_size=block_size,
        kv_num_blocks=8,
        kv_block_table_ids=tuple(block for table in block_tables for block in table),
        kv_block_table_offsets=tuple(table_offsets),
        kv_slot_mapping=tuple(slots),
        request_ids=(10, 20, 30),
    )


def test_single_query_metadata_selects_decode_and_one_token_prefill():
    batch = _batch()
    data = ModelInput.from_prepared_batch(batch, device="cpu", paged_decode=True)
    metadata = data.kv.decode

    assert metadata is not None
    assert metadata.query_rows.tolist() == [0, 4]
    assert metadata.block_table_ids.tolist() == [4, 1, 3]
    assert metadata.block_table_offsets.tolist() == [0, 2, 3]
    assert metadata.context_lens.tolist() == [17, 1]
    assert data.kv.read_slots[0] is None
    assert data.kv.read_slots[2] is None
    assert data.kv.read_slots[1].tolist() == list(range(96, 101))
    assert data.kv.slot_mapping.tolist() == list(batch.kv_slot_mapping)
    assert data.positions.tolist() == list(batch.positions)

    for tensor in (
        metadata.query_rows,
        metadata.block_table_ids,
        metadata.block_table_offsets,
        metadata.context_lens,
        data.kv.slot_mapping,
        data.kv.read_slots[1],
    ):
        assert tensor.dtype == torch.int64
        assert tensor.device.type == "cpu"


def test_reference_lowering_preserves_all_history_and_has_no_decode_metadata():
    data = ModelInput.from_prepared_batch(_batch(), device="cpu")

    assert data.kv.decode is None
    assert [slots.tolist() for slots in data.kv.read_slots] == [
        list(range(64, 80)) + [16],
        list(range(96, 101)),
        [48],
    ]


def test_multi_query_only_batch_has_no_single_query_metadata():
    batch = _batch(query_lens=(2, 3, 2), context_lens=(17, 5, 2))
    data = ModelInput.from_prepared_batch(batch, device="cpu", paged_decode=True)

    assert data.kv.decode is None
    assert all(slots is not None for slots in data.kv.read_slots)
    assert [slots.numel() for slots in data.kv.read_slots] == [17, 5, 2]


def test_attention_backend_config_is_explicit_and_reference_is_default(tmp_path):
    assert EngineConfig(model=tmp_path).attention_backend == "reference"
    assert EngineConfig(model=tmp_path, attention_backend="cuda").attention_backend == "cuda"

    with pytest.raises(ValueError, match="attention backend"):
        EngineConfig(model=tmp_path, attention_backend="automatic")


@pytest.mark.parametrize("backend", ["cuda", "automatic"])
def test_runner_rejects_unsupported_backend_before_allocating_cache(backend, monkeypatch):
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.zeros(1, dtype=torch.bfloat16)))
    model.config = SimpleNamespace(
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=2048,
    )

    def unexpected_allocation(**kwargs):
        pytest.fail("invalid backend configuration reached KV allocation")

    monkeypatch.setattr("tinyvllm.runtime.runner.PagedKVCache", unexpected_allocation)

    with pytest.raises(ValueError):
        ModelRunner(
            model,
            block_size=16,
            num_blocks=4,
            attention_backend=backend,
        )


CUDA_ENABLED = (
    os.environ.get("TINYVLLM_RUN_CUDA") == "1"
    and torch.cuda.is_available()
)


@pytest.mark.skipif(not CUDA_ENABLED, reason="requires TINYVLLM_RUN_CUDA=1 and CUDA")
@pytest.mark.parametrize(
    "dtype,lengths,num_heads,num_kv_heads,head_dim,strided,qk_magnitude,block_size,offset_storage",
    [
        (torch.bfloat16, (1,), 2, 2, 32, False, 1, 16, False),
        (torch.float16, (15, 16, 17), 8, 2, 64, True, 1, 16, False),
        (torch.bfloat16, (65, 257), 6, 2, 128, True, 1, 16, False),
        (torch.float16, (1025,), 4, 1, 256, False, 1, 16, False),
        (torch.bfloat16, (129,), 4, 2, 6, True, 16, 16, False),
        (torch.bfloat16, (2, 3, 4, 65), 4, 2, 80, False, 1, 3, False),
        (torch.float16, (129,), 4, 2, 129, False, 1, 16, False),
        (torch.float16, (17,), 4, 2, 128, False, 1, 16, True),
    ],
    ids=[
        "bf16-single",
        "fp16-block-boundaries",
        "bf16-gqa-strided",
        "fp16-long-context",
        "bf16-large-scores-padded-dim",
        "bf16-non-power-of-two-pages-padded-dim80",
        "fp16-padded-dim129",
        "fp16-contiguous-offset-storage",
    ],
)
@torch.inference_mode()
def test_paged_decode_matches_independent_fp32_dense_attention(
    dtype,
    lengths,
    num_heads,
    num_kv_heads,
    head_dim,
    strided,
    qk_magnitude,
    block_size,
    offset_storage,
):
    from tinyvllm.kernels.paged_decode import paged_decode_attention

    generator = torch.Generator().manual_seed(90210)
    counts = [(length + block_size - 1) // block_size for length in lengths]
    num_blocks = sum(counts) + 3
    physical_ids = torch.randperm(num_blocks, generator=generator).tolist()[:sum(counts)]
    table_offsets = [0]
    for count in counts:
        table_offsets.append(table_offsets[-1] + count)

    # Extra rows model multi-token prefill in a mixed batch. Reverse the selected
    # rows so neither compact request order nor contiguous Q storage is assumed.
    total_rows = 2 * len(lengths) + 1
    query_rows = list(range(1, total_rows, 2))[::-1]
    if strided:
        q_cpu = torch.randn(
            (total_rows, head_dim, num_heads), generator=generator
        ).to(dtype).transpose(1, 2)
    else:
        q_cpu = torch.randn(
            (total_rows, num_heads, head_dim), generator=generator
        ).to(dtype)
    q_cpu.mul_(qk_magnitude)
    q = q_cpu.to(device="cuda")
    assert q.is_contiguous() == (not strided)

    def offset_copy(tensor):
        # A contiguous tensor need not start at an aligned allocation boundary.
        # Keep its public shape/stride contract, but offset its address by 2 bytes.
        storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
        view = storage[1:].view(tensor.shape)
        view.copy_(tensor)
        assert view.is_contiguous()
        assert view.storage_offset() == 1
        assert view.data_ptr() % 16 == 2
        return view

    if offset_storage:
        q = offset_copy(q)

    cache = PagedKVCache(
        num_layers=2,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cuda",
        dtype=dtype,
    )
    # Every invalid tail, unallocated block, and other layer remains poisoned.
    cache.key.fill_(float("nan"))
    cache.value.fill_(float("nan"))
    if offset_storage:
        cache.key = offset_copy(cache.key)
        cache.value = offset_copy(cache.value)
    layer_idx = 1
    scale = 0.7 / math.sqrt(head_dim)
    head_map = torch.arange(num_heads) // (num_heads // num_kv_heads)
    expected_rows = []

    for index, (length, query_row) in enumerate(zip(lengths, query_rows)):
        begin, end = table_offsets[index:index + 2]
        table = physical_ids[begin:end]
        slots = torch.tensor(
            [
                table[position // block_size] * block_size + position % block_size
                for position in range(length)
            ],
            dtype=torch.int64,
            device="cuda",
        )
        key = torch.randn(
            (length, num_kv_heads, head_dim), generator=generator
        ).to(dtype)
        key.mul_(qk_magnitude)
        if qk_magnitude > 1:
            # Maxima at opposite ends test rescaling in either traversal order.
            group_size = num_heads // num_kv_heads
            key[0, 0].copy_(q_cpu[query_row, 0] * 8)
            key[-1, 1].copy_(q_cpu[query_row, group_size] * 8)
        value = torch.randn(
            (length, num_kv_heads, head_dim), generator=generator
        ).to(dtype)
        cache.key[layer_idx].index_copy_(0, slots, key.to("cuda"))
        cache.value[layer_idx].index_copy_(0, slots, value.to("cuda"))

        # CPU FP32 math deliberately avoids SDPA, the production slot reader,
        # and CUDA TF32. The inputs are quantized to the tested dtype first.
        dense_key = key.float().index_select(1, head_map)
        dense_value = value.float().index_select(1, head_map)
        scores = torch.einsum("hd,thd->ht", q_cpu[query_row].float(), dense_key) * scale
        if qk_magnitude > 1:
            assert scores.max() > 100  # Direct FP32 exp would overflow.
            assert scores[0, 0] > scores[0, 1:].max()
            assert (scores[:, -1] > scores[:, :-1].amax(dim=1)).any()
        probabilities = scores.softmax(dim=-1)
        expected_rows.append(torch.einsum("ht,thd->hd", probabilities, dense_value))

    metadata = SingleQueryMetadata(
        query_rows=torch.tensor(query_rows, dtype=torch.int64, device="cuda"),
        block_table_ids=torch.tensor(physical_ids, dtype=torch.int64, device="cuda"),
        block_table_offsets=torch.tensor(table_offsets, dtype=torch.int64, device="cuda"),
        context_lens=torch.tensor(lengths, dtype=torch.int64, device="cuda"),
    )
    output = torch.empty_like(q)
    output.fill_(123)
    if offset_storage:
        output = offset_copy(output)

    paged_decode_attention(q, cache, layer_idx, metadata, output, scale=scale)
    torch.cuda.synchronize()

    actual = output.cpu()
    expected = torch.stack(expected_rows)
    assert torch.isfinite(actual[query_rows]).all()
    # BF16 has fewer mantissa bits; reduction ordering and the final output cast
    # differ from FP32. These fixed limits must not grow after observing failures.
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(
        actual[query_rows].float(),
        expected,
        atol=tolerance,
        rtol=tolerance,
    )
    untouched = [row for row in range(total_rows) if row not in query_rows]
    assert torch.equal(actual[untouched], torch.full_like(actual[untouched], 123))


@pytest.mark.skipif(not CUDA_ENABLED, reason="requires TINYVLLM_RUN_CUDA=1 and CUDA")
@torch.inference_mode()
def test_paged_decode_obeys_current_stream_and_event_dependencies():
    from tinyvllm.kernels.paged_decode import paged_decode_attention

    dtype = torch.bfloat16
    q = torch.full((1, 2, 32), -1, device="cuda", dtype=dtype)
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=1,
        block_size=16,
        num_kv_heads=1,
        head_dim=32,
        device="cuda",
        dtype=dtype,
    )
    cache.key.zero_()
    cache.value.fill_(-7)
    metadata = SingleQueryMetadata(
        query_rows=torch.tensor([0], device="cuda", dtype=torch.int64),
        block_table_ids=torch.tensor([0], device="cuda", dtype=torch.int64),
        block_table_offsets=torch.tensor([0, 1], device="cuda", dtype=torch.int64),
        context_lens=torch.tensor([2], device="cuda", dtype=torch.int64),
    )
    output = torch.empty_like(q)
    scale = 1 / math.sqrt(32)

    # Compile and initialize the extension before arranging the stream race.
    paged_decode_attention(q, cache, 0, metadata, output, scale=scale)
    new_query = torch.ones_like(q)
    new_keys = torch.stack((-torch.ones(32), torch.ones(32))).to(
        device="cuda", dtype=dtype,
    ).unsqueeze(1)
    new_values = torch.stack((torch.full((32,), 2), torch.full((32,), 6))).to(
        device="cuda", dtype=dtype,
    ).unsqueeze(1)
    output.fill_(111)
    torch.cuda.synchronize()

    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    ready = torch.cuda.Event()
    with torch.cuda.stream(producer):
        # A mistaken default-stream launch sees the poisoned old inputs while
        # this nonblocking stream is delayed. No device sync may bridge the gap.
        torch.cuda._sleep(50_000_000)
        q.copy_(new_query)
        cache.key[0, :2].copy_(new_keys)
        cache.value[0, :2].copy_(new_values)
        paged_decode_attention(q, cache, 0, metadata, output, scale=scale)
        ready.record()

    with torch.cuda.stream(consumer):
        consumer.wait_event(ready)
        observed = output.clone()
    consumer.synchronize()

    scores = torch.tensor([-math.sqrt(32), math.sqrt(32)])
    expected = (scores.softmax(dim=0) * torch.tensor([2.0, 6.0])).sum().item()
    torch.testing.assert_close(
        observed.cpu().float(),
        torch.full(q.shape, expected),
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(not CUDA_ENABLED, reason="requires TINYVLLM_RUN_CUDA=1 and CUDA")
@pytest.mark.parametrize(
    "invalid",
    [
        "query_rank",
        "query_dtype",
        "output_shape",
        "output_dtype",
        "metadata_dtype",
        "metadata_device",
        "metadata_length",
        "metadata_stride",
    ],
)
@torch.inference_mode()
def test_paged_decode_rejects_invalid_tensor_contracts(invalid):
    from tinyvllm.kernels.paged_decode import paged_decode_attention

    q = torch.zeros((1, 4, 32), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(q)
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=1,
        block_size=16,
        num_kv_heads=2,
        head_dim=32,
        device="cuda",
        dtype=torch.bfloat16,
    )
    metadata = SingleQueryMetadata(
        query_rows=torch.tensor([0], device="cuda", dtype=torch.int64),
        block_table_ids=torch.tensor([0], device="cuda", dtype=torch.int64),
        block_table_offsets=torch.tensor([0, 1], device="cuda", dtype=torch.int64),
        context_lens=torch.tensor([1], device="cuda", dtype=torch.int64),
    )
    if invalid == "query_rank":
        q = q[0]
    elif invalid == "query_dtype":
        q = q.float()
    elif invalid == "output_shape":
        output = torch.empty((2, 4, 32), device="cuda", dtype=q.dtype)
    elif invalid == "output_dtype":
        output = output.float()
    elif invalid == "metadata_dtype":
        metadata = replace(metadata, query_rows=metadata.query_rows.int())
    elif invalid == "metadata_device":
        metadata = replace(metadata, query_rows=metadata.query_rows.cpu())
    elif invalid == "metadata_length":
        metadata = replace(metadata, block_table_offsets=metadata.block_table_offsets[:1])
    elif invalid == "metadata_stride":
        offsets = torch.tensor([0, 99, 1, 99], device="cuda", dtype=torch.int64)[::2]
        assert not offsets.is_contiguous()
        metadata = replace(metadata, block_table_offsets=offsets)

    # Tensor sizes/dtypes/devices/strides are host metadata. Rejecting them must
    # not require downloading index values or adding a GPU synchronization.
    with pytest.raises((ValueError, RuntimeError)):
        paged_decode_attention(q, cache, 0, metadata, output, scale=1 / math.sqrt(32))
