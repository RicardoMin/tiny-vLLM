import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.tools.validate_workload import canonical
from benchmarks.tools.collect_env import sha256_file


@pytest.mark.cuda
@pytest.mark.model
def test_real_benchmark_cli_mixed_replay_and_memory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "benchmarks/workloads/smoke-synth-qwen3-v1.json").read_text())
    records = [
        {
            "request_id": f"req-{index:06d}",
            "ordinal": index,
            "arrival_ns": index * 2_000_000,
            "prompt_token_ids": list(range(1, length + 1)),
            "max_new_tokens": 8,
            "temperature": 0.6,
            "ignore_eos": True,
        }
        for index, length in enumerate((1, 80, 130))
    ]
    records_path = tmp_path / "mixed.jsonl"
    records_path.write_bytes(b"".join(canonical(record) for record in records))
    manifest.update(
        workload_id="p2b-gpu-test",
        records_file=records_path.name,
        records_sha256=sha256_file(records_path),
        record_count=len(records),
        totals={"prompt_tokens": 211, "requested_output_tokens": 24},
    )
    manifest_path = tmp_path / "mixed.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "result.json"

    result = subprocess.run(
        [
            sys.executable, "-m", "benchmarks.tools.run_tiny",
            "--model", os.environ["TINYVLLM_MODEL"],
            "--workload", str(manifest_path),
            "--output", str(output),
            "--warmup", "1", "--repeats", "2",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "ok"
    assert report["phase"] == "complete"
    assert len(report["identity"]["repo"]["worktree_sha256"]) == 64
    assert report["kv_pool"]["allocated_bytes"] <= 512 * 1024**2

    for run in report["runs"]:
        assert run["completed_requests"] == 3
        assert run["output_tokens"] == 24
        assert run["itl_ms"]["count"] == 21
        assert run["tpot_ms"]["count"] == 3
        assert run["steps_by_mode"].get("MIXED", 0) > 0
        memory = run["memory"]
        assert memory["peak_allocated_bytes"] >= memory["baseline_allocated_bytes"]
        assert memory["peak_reserved_bytes"] >= memory["peak_allocated_bytes"]
        assert 0 < memory["peak_used_kv_blocks"] <= report["kv_pool"]["num_blocks"]
        assert memory["extra_peak_allocated_bytes"] == memory["peak_allocated_bytes"] - memory["baseline_allocated_bytes"]

        for row in run["requests"]:
            assert len(row["token_ids"]) == len(row["token_times_ns"]) == 8
            assert row["submitted_ns"] >= row["arrival_ns"]
            assert row["token_times_ns"] == sorted(row["token_times_ns"])
            assert row["finished_ns"] >= row["token_times_ns"][-1]
            expected = (row["token_times_ns"][-1] - row["token_times_ns"][0]) / 7 / 1e6
            assert row["tpot_ms"] == expected

    print(json.dumps({"status": report["status"], "runs": [
        {"output_tokens": run["output_tokens"], "steps": run["steps_by_mode"], "memory": run["memory"]}
        for run in report["runs"]
    ]}))
