"""Explicit optional test gates and shared construction fixtures.

Default pytest is CPU-only. CUDA needs TINYVLLM_RUN_CUDA=1;
FlashInfer also needs TINYVLLM_RUN_FLASHINFER=1; local checkpoint
tests need TINYVLLM_MODEL. An opted-in missing dependency is a failure,
not silently skipped.
"""

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    for name, description in (
        ("cuda", "requires explicit CUDA opt-in and an available CUDA device"),
        ("model", "requires local model assets via TINYVLLM_MODEL"),
        ("flashinfer", "requires CUDA and explicit FlashInfer opt-in"),
    ):
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_collection_modifyitems(items):
    cuda_available = None
    for item in items:
        needs_flashinfer = item.get_closest_marker("flashinfer") is not None
        needs_cuda = needs_flashinfer or item.get_closest_marker("cuda") is not None
        reasons = []
        if item.get_closest_marker("model") and not os.environ.get("TINYVLLM_MODEL"):
            reasons.append("set TINYVLLM_MODEL to local model assets")
        if needs_flashinfer and os.environ.get("TINYVLLM_RUN_FLASHINFER") != "1":
            reasons.append("set TINYVLLM_RUN_FLASHINFER=1")
        if needs_cuda:
            if os.environ.get("TINYVLLM_RUN_CUDA") != "1":
                reasons.append("set TINYVLLM_RUN_CUDA=1")
            else:
                if cuda_available is None:
                    import torch

                    cuda_available = torch.cuda.is_available()
                if not cuda_available:
                    raise pytest.UsageError(
                        'TINYVLLM_RUN_CUDA=1 but CUDA is unavailable; fix the environment or remove opt-in'
                    )
        if reasons:
            item.add_marker(pytest.mark.skip(reason="; ".join(reasons)))


@pytest.fixture
def model_path():
    return Path(os.environ["TINYVLLM_MODEL"])


@pytest.fixture
def small_model_config(model_path, backend):
    from tinyvllm.config import EngineConfig

    return EngineConfig(
        model=model_path,
        attention_backend=backend,
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        prefill_chunk_size=16,
        prefill_reserve_tokens=0,
        prefill_reserve_requests=0,
        kv_cache_memory_bytes=64 * 1024**2,
    )


@pytest.fixture
def mocked_cuda_loader(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    from tinyvllm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    GiB = 1024**3
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
