from __future__ import annotations

import pytest
import torch

from tinyvllm.cache import BlockAppend, BlockManager, BlockRef
from tinyvllm.paged_kv import PagedKVCache


def make_store() -> PagedKVCache:
    return PagedKVCache(
        num_layers=2,
        num_blocks=4,
        block_size=2,
        num_kv_heads=1,
        head_dim=2,
        device="cpu",
        dtype=torch.float32,
    )


def test_logical_slots_preserve_noncontiguous_block_order() -> None:
    store = make_store()
    refs = (BlockRef(3, 1), BlockRef(0, 1), BlockRef(2, 1))
    slots = store.logical_slots(refs, start=1, end=5)
    assert slots.tolist() == [7, 0, 1, 4]


def test_empty_range_needs_no_page() -> None:
    store = make_store()

    slots = store.logical_slots(
        (),
        start=5,
        end=5,
    )

    assert slots.numel() == 0


def test_write_read_preserves_order_and_layers() -> None:
    manager = BlockManager(
        block_size=2,
        num_blocks=4,
    )

    txn = manager.begin_plan(
        plan_id=0,
        appends=(
            BlockAppend(
                request_id=5,
                token_start=0,
                query_len=5,
            ),
        ),
    )

    assert txn is not None

    refs = txn.staged_block_table(5)
    store = make_store()

    slots = store.logical_slots(
        refs,
        start=0,
        end=5,
    )

    key = torch.arange(
        10,
        dtype=torch.float32,
    ).reshape(5, 1, 2)

    value = key + 100

    store.write(
        0,
        slots,
        key,
        value,
    )

    store.write(
        1,
        slots,
        key + 1,
        value + 1,
    )

    actual_key, actual_value = store.read(
        0,
        slots,
    )

    assert torch.equal(actual_key, key)
    assert torch.equal(actual_value, value)

    layer_key, layer_value = store.read(
        1,
        slots,
    )

    assert torch.equal(layer_key, key + 1)
    assert torch.equal(layer_value, value + 1)


def test_partial_page_append_and_page_reuse() -> None:
    manager = BlockManager(
        block_size=2,
        num_blocks=2,
    )

    first = manager.begin_plan(
        plan_id=0,
        appends=(
            BlockAppend(
                request_id=1,
                token_start=0,
                query_len=1,
            ),
        ),
    )

    assert first is not None

    first.commit_before_dispatch()
    refs = first.staged_block_table(1)

    store = make_store()
    old_slots = store.logical_slots(
        refs,
        start=0,
        end=1,
    )

    old = torch.tensor(
        [[[3.0, 4.0]]],
    )

    store.write(
        0,
        old_slots,
        old,
        old,
    )

    first.complete_writes()
    manager.release_request(1)

    second = manager.begin_plan(
        plan_id=1,
        appends=(
            BlockAppend(
                request_id=2,
                token_start=0,
                query_len=1,
            ),
        ),
    )

    assert second is not None

    new_refs = second.staged_block_table(2)

    assert new_refs[0].block_id == refs[0].block_id
    assert new_refs[0].generation != refs[0].generation

    new_slots = store.logical_slots(
        new_refs,
        start=0,
        end=1,
    )

    new = torch.tensor(
        [[[9.0, 10.0]]],
    )

    store.write(
        0,
        new_slots,
        new,
        new,
    )

    actual, _ = store.read(
        0,
        new_slots,
    )

    assert torch.equal(actual, new)


def test_rejects_short_table_and_bad_rows() -> None:
    manager = BlockManager(
        block_size=2,
        num_blocks=2,
    )

    txn = manager.begin_plan(
        plan_id=0,
        appends=(
            BlockAppend(
                request_id=1,
                token_start=0,
                query_len=1,
            ),
        ),
    )

    assert txn is not None

    store = make_store()

    with pytest.raises(ValueError, match="too short"):
        store.logical_slots(
            txn.staged_block_table(1),
            start=0,
            end=3,
        )

    slots = torch.zeros(
        1,
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="do not match"):
        store.write(
            0,
            slots,
            torch.zeros(2, 1, 2),
            torch.zeros(2, 1, 2),
        )

    with pytest.raises(ValueError, match="int64"):
        store.read(
            0,
            torch.zeros(
                1,
                dtype=torch.int32,
            ),
        )

def test_rejects_reversed_logical_range() -> None:
    store = make_store()

    with pytest.raises(ValueError, match="end must not"):
        store.logical_slots(
            (),
            start=2,
            end=1,
        )


@pytest.mark.parametrize("slot", (-1, 8))
def test_rejects_out_of_range_slots(slot: int) -> None:
    store = make_store()

    with pytest.raises(ValueError, match="out of range"):
        store.read(
            0,
            torch.tensor([slot], dtype=torch.long),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)
def test_unindexed_cuda_device_uses_current_device() -> None:
    store = PagedKVCache(
        num_layers=1,
        num_blocks=1,
        block_size=1,
        num_kv_heads=1,
        head_dim=2,
        device="cuda",
        dtype=torch.float32,
    )

    expected = torch.device(
        "cuda",
        torch.cuda.current_device(),
    )
    assert store.device == expected
    assert store.key.device == expected
