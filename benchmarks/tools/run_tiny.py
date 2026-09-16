from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.tools.collect_env import (
    git_identity,
    model_manifest,
    packages,
    sha256_file,
)
from benchmarks.tools.measurement import ExecutionProbe, replay
from benchmarks.tools.validate_workload import strict_json, validate
from tinyvllm.config import EngineConfig
from tinyvllm.engine import Engine


def main():
    parser = argparse.ArgumentParser(
        description="Measure the local tiny-vLLM engine"
    )

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--kv-cache-mib", type=int, default=512)

    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")

    if args.output.exists():
        parser.error("output already exists; choose a new run directory")

    workload = validate(args.workload)
    manifest = strict_json(args.workload.read_bytes())
    records_path = args.workload.parent / manifest["records_file"]

    records = [
        strict_json(line)
        for line in records_path.read_bytes().splitlines()
    ]

    config = EngineConfig(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        prefill_reserve_tokens=0,
        prefill_reserve_requests=0,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024**2,
    )

    repo = Path(__file__).resolve().parents[2]

    report = {
        "schema_version": "tinyvllm.benchmark.v1",
        "status": "running",
        "protocol": {
            "arrival": "open_loop_due_submit_at_step_boundaries",
            "token_clock": "host_monotonic_after_step_return",
            "percentile": "nearest_rank",
            "output_counting": "all_emitted_tokens_including_eos",
            "timed_region": "runner_start_to_last_completion_including_idle",
            "warmup_replays": args.warmup,
            "repeats": args.repeats,
            "seed_per_request": args.seed,
            "prefill_reservation": "disabled",
        },
        "identity": {
            "repo": git_identity(repo),
            "python": platform.python_version(),
            "packages": packages(),
            "model": model_manifest(config.model),
            "manifest_sha256": sha256_file(args.workload),
        },
        "config": {
            **asdict(config),
            "model": str(config.model),
        },
        "workload": workload,
        "runs": [],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        report["phase"] = "bootstrap"

        engine = Engine.from_config(config)
        executor = engine.executor
        cache = executor.runner.kv_cache
        device = executor.runner.device

        if manifest["vocab_size"] != executor.capabilities.vocab_size:
            raise ValueError("workload vocabulary does not match model")

        if set(manifest["eos_token_ids"]) != set(engine.scheduler.eos_token_ids):
            raise ValueError("workload EOS IDs do not match model")

        pool = engine.scheduler.block_manager.pool
        probe = ExecutionProbe(executor, pool)
        engine.executor = probe

        kv_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (cache.key, cache.value)
        )

        report["device"] = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": (
                torch.cuda.get_device_properties(device).total_memory
            ),
            "compute_capability": torch.cuda.get_device_capability(device),
            "torch_cuda": torch.version.cuda,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "sdpa_enabled": {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
            },
        }

        report["kv_pool"] = {
            "allocated_bytes": kv_bytes,
            "num_blocks": pool.num_blocks,
            "block_size": cache.block_size,
        }

        report["phase"] = "warmup"

        for index in range(args.warmup):
            report["phase_index"] = index
            replay(engine, records, seed=args.seed)

        report["phase"] = "measurement"

        for repeat in range(args.repeats):
            report["phase_index"] = repeat

            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            baseline_reserved = torch.cuda.memory_reserved(device)

            torch.cuda.reset_peak_memory_stats(device)
            probe.reset()

            result = replay(engine, records, seed=args.seed)

            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)

            if pool.free_block_count != pool.num_blocks:
                raise RuntimeError("KV blocks remain owned after replay")

            result["repeat"] = repeat
            result["steps_by_mode"] = dict(probe.steps_by_mode)
            result["memory"] = {
                "baseline_allocated_bytes": baseline,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": peak,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "extra_peak_allocated_bytes": peak - baseline,
                "peak_used_kv_blocks": probe.peak_used_blocks,
                "peak_used_kv_bytes": (
                    probe.peak_used_blocks * (kv_bytes // pool.num_blocks)
                ),
            }

            report["runs"].append(result)

        report["status"] = "ok"
        report["phase"] = "complete"

    except BaseException as error:
        report["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        report["error"] = f"{type(error).__name__}: {error}"

        raise

    finally:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        temporary.replace(args.output)

    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "runs": [
            {
                key: value
                for key, value in run.items()
                if key != "requests"
            }
            for run in report["runs"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()