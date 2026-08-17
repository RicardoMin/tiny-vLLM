from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tinyvllm._validation import _check_positive_int


DType = Literal["bfloat16", "float16"]


@dataclass(frozen=True, slots=True)
class EngineConfig:
    model: Path
    dtype: DType = "bfloat16"
    max_model_len: int = 4096
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 4096
    gpu_memory_utilization: float = 0.9

    prefill_chunk_size: int = 512
    prefill_reserve_tokens: int = 128
    prefill_reserve_requests: int = 1

    def __post_init__(self) -> None:
        model = Path(self.model).expanduser().resolve()
        object.__setattr__(self, "model", model)

        if not model.is_dir():
            raise ValueError(f"mdoel directory does not exist: {model}")
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError(f"unsupported dtype: {self.dtype}")

        for name in (
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "prefill_chunk_size",
        ):
            _check_positive_int(name, getattr(self, name))

        if not 0.0 < self.gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory utilization must be between 0 and 1")

        tokens = self.prefill_reserve_tokens
        requests = self.prefill_reserve_requests

        if (tokens == 0) != (requests == 0):
            raise ValueError("prefill reservations must both be zero or positive")
        if not 0 <= tokens < self.max_num_batched_tokens:
            raise ValueError("prefill token reservation leaves no decode capacity")
        if not 0 <= requests < self.max_num_seqs:
            raise ValueError("prefill request reservation leaves no decode capacity")
        if tokens < requests:
            raise ValueError("each reserved prefill request needs one token")
        if self.prefill_chunk_size > self.max_num_batched_tokens:
            raise ValueError("prefill chunk exceeds the step token budget")
