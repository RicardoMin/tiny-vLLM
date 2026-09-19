from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tinyvllm.bootstrap import _plan_kv_blocks, create_engine
from tinyvllm.config import EngineConfig
from tinyvllm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM


GiB = 1024**3
MiB = 1024**2


def test_kv_budget_rounds_down_without_exceeding_explicit_limit():
    # Qwen3-0.6B: K and V, 28 layers, 16 tokens, 8 heads, 128 dims, BF16.
    block_bytes = 2 * 28 * 16 * 8 * 128 * 2
    blocks = _plan_kv_blocks(
        total_bytes=8 * GiB,
        free_bytes=6 * GiB,
        utilization=0.9,
        requested_bytes=2 * GiB,
        bytes_per_block=block_bytes,
    )

    assert blocks == 1170
    assert blocks * block_bytes <= 2 * GiB
    assert (blocks + 1) * block_bytes > 2 * GiB


def test_device_cap_counts_memory_already_used_by_other_processes():
    with pytest.raises(ValueError, match="after model loading"):
        _plan_kv_blocks(
            total_bytes=8 * GiB,
            free_bytes=3 * GiB,
            utilization=0.75,
            requested_bytes=2 * GiB,
            bytes_per_block=MiB,
        )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"free_bytes": -1}, "snapshot"),
        ({"free_bytes": 9 * GiB}, "snapshot"),
        ({"requested_bytes": 1}, "one complete block"),
        ({"utilization": float("nan")}, "utilization"),
    ],
)
def test_invalid_or_unusable_budget_is_rejected(changes, message):
    arguments = dict(
        total_bytes=8 * GiB,
        free_bytes=6 * GiB,
        utilization=0.9,
        requested_bytes=2 * GiB,
        bytes_per_block=MiB,
    )
    arguments.update(changes)

    with pytest.raises(ValueError, match=message):
        _plan_kv_blocks(**arguments)


@pytest.fixture
def mocked_cuda_loader(monkeypatch):
    model_config = SimpleNamespace(
        max_position_embeddings=4096,
        num_hidden_layers=28,
        num_key_value_heads=8,
        head_dim=128,
        eos_token_id=[151643, 151645],
    )
    model = SimpleNamespace(config=model_config)
    loader = Mock(return_value=model)
    executor_factory = Mock()
    memory_snapshot = Mock(return_value=(6 * GiB, 8 * GiB))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(torch.cuda, "mem_get_info", memory_snapshot)
    monkeypatch.setattr(
        Qwen3Config, "from_model_dir", Mock(return_value=model_config)
    )
    monkeypatch.setattr(Qwen3ForCausalLM, "from_pretrained", loader)
    monkeypatch.setattr("tinyvllm.executor.Executor", executor_factory)

    return model, loader, executor_factory, memory_snapshot


def test_bootstrap_aligns_model_scheduler_and_executor(
    tmp_path, mocked_cuda_loader
):
    model, loader, executor_factory, snapshot = mocked_cuda_loader
    config = EngineConfig(
        model=tmp_path,
        dtype="float16",
        attention_backend="cuda",
        kv_block_size=8,
        kv_cache_memory_bytes=MiB,
    )

    def memory_after_loading(device):
        loader.assert_called_once()
        return 6 * GiB, 8 * GiB

    snapshot.side_effect = memory_after_loading
    engine = create_engine(config)

    loader.assert_called_once_with(
        config.model,
        device=torch.device("cuda:1"),
        dtype=torch.float16,
    )
    snapshot.assert_called_once_with(torch.device("cuda:1"))
    executor_factory.assert_called_once_with(
        model,
        block_size=8,
        num_blocks=1,
        max_model_len=config.max_model_len,
        attention_backend="cuda",
    )
    assert engine.executor is executor_factory.return_value
    assert engine.scheduler.kv_block_size == 8
    assert engine.scheduler.block_manager.pool.num_blocks == 1
    assert engine.scheduler.eos_token_ids == {151643, 151645}
    assert engine.scheduler.max_model_len == config.max_model_len
    assert engine.scheduler.max_scheduled_tokens == config.max_num_batched_tokens


def test_unsupported_length_rejected_before_weights_or_kv_allocation(
    tmp_path, mocked_cuda_loader
):
    _, loader, executor_factory, snapshot = mocked_cuda_loader

    with pytest.raises(ValueError, match="position capacity"):
        create_engine(EngineConfig(model=tmp_path, max_model_len=8192))

    loader.assert_not_called()
    executor_factory.assert_not_called()
    snapshot.assert_not_called()


def test_usage_cap_rejects_before_kv_allocation(tmp_path, mocked_cuda_loader):
    _, _, executor_factory, snapshot = mocked_cuda_loader
    snapshot.return_value = (3 * GiB, 8 * GiB)

    with pytest.raises(ValueError, match="GPU usage ceiling"):
        create_engine(
            EngineConfig(model=tmp_path, gpu_memory_utilization=0.75)
        )

    executor_factory.assert_not_called()


def test_no_cuda_reports_clear_bootstrap_error(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA device"):
        create_engine(EngineConfig(model=tmp_path))
