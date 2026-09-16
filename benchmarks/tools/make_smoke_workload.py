from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


WORKLOAD_ID = "smoke-synth-qwen3-v1"
REQUEST_COUNT = 8
PROMPT_TOKENS = 16
MAX_NEW_TOKENS = 8
SEED = 0


def canonical_line(value: dict) -> bytes:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return (text + "\n").encode()


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible workloads"
    )
    parser.add_argument(
        "--profile-mixed",
        action="store_true",
        help="Generate varied prompt lengths for profiling mixed execution",
    )
    args = parser.parse_args()

    workload_id = (
        "profile-mixed-qwen3-v1"
        if args.profile_mixed
        else WORKLOAD_ID
    )
    suite = "profile" if args.profile_mixed else "smoke"
    prompt_lengths = (
        (16, 128, 256, 512)
        if args.profile_mixed
        else (PROMPT_TOKENS,) * REQUEST_COUNT
    )
    max_new_tokens = 16 if args.profile_mixed else MAX_NEW_TOKENS

    output_dir = Path("benchmarks/workloads")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    records = []

    for ordinal, prompt_tokens in enumerate(prompt_lengths):
        records.append({
            "request_id": f"req-{ordinal:06d}",
            "ordinal": ordinal,
            "arrival_ns": 0,
            "prompt_token_ids": [
                rng.randrange(10_000)
                for _ in range(prompt_tokens)
            ],
            "max_new_tokens": max_new_tokens,
            "temperature": 0.6,
            "ignore_eos": True,
        })

    records_bytes = b"".join(
        canonical_line(record)
        for record in records
    )

    records_name = f"{workload_id}.jsonl"
    records_hash = hashlib.sha256(records_bytes).hexdigest()

    manifest = {
        "schema_version": "tinyvllm.workload.v1",
        "workload_id": workload_id,
        "suite": suite,
        "model_id": "Qwen/Qwen3-0.6B",
        "vocab_size": 151936,
        "eos_token_ids": [151645],
        "records_file": records_name,
        "records_sha256": records_hash,
        "record_count": len(prompt_lengths),
        "totals": {
            "prompt_tokens": sum(prompt_lengths),
            "requested_output_tokens": len(prompt_lengths) * max_new_tokens,
        },
        "sampling": {
            "temperature": 0.6,
            "ignore_eos": True,
        },
        "arrival": {
            "clock": "monotonic_ns_relative",
            "origin": "runner_start",
            "sorted": True,
        },
        "source": {
            "kind": "synthetic_uniform",
            "seed": SEED,
            "token_min_inclusive": 0,
            "token_max_exclusive": 10_000,
        },
    }

    atomic_write(output_dir / records_name, records_bytes)
    atomic_write(
        output_dir / f"{workload_id}.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )

    print(json.dumps({
        "workload_id": workload_id,
        "records_sha256": records_hash,
        **manifest["totals"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()