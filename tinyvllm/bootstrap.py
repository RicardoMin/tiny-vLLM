from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tinyvllm.config import EngineConfig

if TYPE_CHECKING:
    from tinyvllm.engine import Engine


def _plan_kv_blocks(
    *,
    total_bytes: int,
    free_bytes: int,
    utilization: float,
    requested_bytes: int,
    bytes_per_block: int,
    reserved_bytes: int = 0,
) -> int:
    """Fit an explicit KV allocation under the device usage ceiling."""

    if total_bytes <= 0 or not 0 <= free_bytes <= total_bytes:
        raise ValueError("invalid CUDA memory snapshot")

    if bytes_per_block <= 0 or requested_bytes <= 0:
        raise ValueError("KV byte sizes must be positive")

    if not 0.0 < utilization < 1.0:
        raise ValueError(
            "GPU memory utilization must be between 0 and 1"
        )

    if reserved_bytes < 0:
        raise ValueError("reserved bytes must be nonnegative")

    num_blocks = requested_bytes // bytes_per_block
    if num_blocks == 0:
        raise ValueError(
            "KV budget cannot hold one complete block"
        )

    allocated_bytes = num_blocks * bytes_per_block
    used_bytes = total_bytes - free_bytes

    available_bytes = min(
        free_bytes,
        int(total_bytes * utilization) - used_bytes,
    ) - reserved_bytes

    if allocated_bytes > available_bytes:
        raise ValueError(
            f"KV allocation needs {allocated_bytes} bytes, but only "
            f"{max(0, available_bytes)} bytes remain below the GPU usage "
            f"ceiling after model loading and reserving {reserved_bytes} backend bytes."
            "Lower kv_cache_memory_bytes "
            "or free device memory."
        )

    return num_blocks


def create_engine(config: EngineConfig) -> Engine:
    """Load Qwen3 on the current CUDA device and allocate an explicit KV pool.
    
    The utilization ceiling includes existing device usage. The byte budget
    does not estimate activation peaks; validate workload limits separately.
    """

    from tinyvllm.engine import Engine
    from tinyvllm.executor import Executor
    from tinyvllm.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from tinyvllm.scheduler import Scheduler

    if not torch.cuda.is_available():
        raise RuntimeError(
            "create_engine requires an available CUDA device"
        )

    model_config = Qwen3Config.from_model_dir(config.model)

    if config.max_model_len > model_config.max_position_embeddings:
        raise ValueError(
            "max_model_len exceeds model position capacity"
        )

    device = torch.device(
        "cuda",
        torch.cuda.current_device(),
    )
    dtype = getattr(torch, config.dtype)

    model = Qwen3ForCausalLM.from_pretrained(
        config.model,
        device=device,
        dtype=dtype,
    )

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    bytes_per_block = (
        2
        * model_config.num_hidden_layers
        * config.kv_block_size
        * model_config.num_key_value_heads
        * model_config.head_dim
        * (torch.finfo(dtype).bits // 8)
    )

    reserved_bytes = 0

    if config.attention_backend == "flashinfer":
        from tinyvllm.runtime.prefill import PREFILL_DEVICE_BYTES

        compact_bytes = (
            2 * config.max_num_batched_tokens
            * model_config.num_attention_heads * model_config.head_dim
            * (torch.finfo(dtype).bits // 8)
        )

        reserved_bytes = PREFILL_DEVICE_BYTES + compact_bytes

    num_blocks = _plan_kv_blocks(
        total_bytes=total_bytes,
        free_bytes=free_bytes,
        utilization=config.gpu_memory_utilization,
        requested_bytes=config.kv_cache_memory_bytes,
        bytes_per_block=bytes_per_block,
        reserved_bytes=reserved_bytes,
    )

    scheduler = Scheduler(
        max_active_requests=config.max_num_seqs,
        max_step_requests=config.max_num_seqs,
        max_scheduled_tokens=config.max_num_batched_tokens,
        prefill_chunk_size=config.prefill_chunk_size,
        max_model_len=config.max_model_len,
        prefill_reserve_tokens=config.prefill_reserve_tokens,
        prefill_reserve_requests=config.prefill_reserve_requests,
        kv_block_size=config.kv_block_size,
        kv_num_blocks=num_blocks,
        eos_token_ids=model_config.eos_token_id,
    )

    executor = Executor(
        model,
        block_size=config.kv_block_size,
        num_blocks=num_blocks,
        max_model_len=config.max_model_len,
        attention_backend=config.attention_backend,
    )

    return Engine(scheduler, executor)