from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

GPU_FIELDS = ("index", "uuid", "name", "memory.used", "memory.total", "utilization.gpu")
APP_FIELDS = ("gpu_uuid", "pid", "process_name", "used_memory")

def query(kind: str, fields: tuple[str, ...]) -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{kind}={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        env={**os.environ, "LC_ALL" : "C"},
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    output = result.stdout.strip()
    if not output or output.startswith("No running processes found"):
        return []

    rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
    if any(len(row) != len(fields) for row in rows):
        raise RuntimeError(f"unexpected nvidia-smi output: {output!r}")
    return [dict(zip(fields, row, strict=True)) for row in rows]


def take_sample(physical_gpu: int) -> dict:
    row = next(
        (row for row in query("gpu", GPU_FIELDS) if int(row["index"]) == physical_gpu),
        None,
    )
    if row is None:
        raise RuntimeError(f"GPU {physical_gpu} was not reported")

    apps = [
        app
        for app in query("compute-apps", APP_FIELDS)
        if app["gpu_uuid"] == row["uuid"]
    ]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "index": int(row["index"]),
            "uuid": row["uuid"],
            "name": row["name"],
            "memory_used_mib": int(row["memory.used"]),
            "memory_total_mib": int(row["memory.total"]),
            "utilization_pct": int(row["utilization.gpu"]),
        },
        "compute_apps": apps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument("--max-memory-mib", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("set CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if not selected or "," in selected:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select exactly one GPU")
    if args.samples < 1 or args.interval_ms < 0 or args.max_memory_mib < 0:
        raise RuntimeError("invalid preflight policy")

    samples = []
    failures = []

    for index in range(args.samples):
        sample = take_sample(args.physical_gpu)
        gpu = sample["gpu"]

        if selected not in (str(args.physical_gpu), gpu["uuid"]):
            raise RuntimeError("CUDA_VISIBLE_DEVICES does not match --physical-gpu")
        if gpu["memory_used_mib"] > args.max_memory_mib:
            failures.append(f"sample {index}: memory={gpu['memory_used_mib']} MiB")
        if gpu["utilization_pct"] != 0:
            failures.append(f"sample {index}: utilization={gpu['utilization_pct']}%")
        if sample["compute_apps"]:
            failures.append(f"sample {index}: compute_apps={len(sample['compute_apps'])}")

        samples.append(sample)
        if index + 1 < args.samples:
            time.sleep(args.interval_ms / 1000)

    payload = {
        "schema_version": "tinyvllm.gpu-preflight.v1",
        "policy": vars(args) | {"output": str(args.output)},
        "samples": samples,
        "gate": {"passed": not failures, "failure_reasons": failures},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)

    print(json.dumps(payload["gate"], sort_keys=True))
    return 0 if payload["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())