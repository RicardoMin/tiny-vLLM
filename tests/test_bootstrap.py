import pytest
import torch

from tinyvllm.bootstrap import _plan_kv_blocks, create_engine
from tinyvllm.config import EngineConfig


GiB = 1024**3
MiB = 1024**2


def test_budget_must_hold_at_least_one_complete_block():
    with pytest.raises(ValueError, match="one complete block"):
        _plan_kv_blocks(
            total_bytes=8 * GiB,
            free_bytes=6 * GiB,
            utilization=0.9,
            requested_bytes=1,
            bytes_per_block=MiB,
        )


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
