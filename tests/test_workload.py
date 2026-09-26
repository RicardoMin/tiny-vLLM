"""Reproducible workload generation and validated replay input contracts."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

from benchmarks.tools.make_smoke_workload import main
from benchmarks.tools.measurement import ExecutionProbe, replay
from tinyvllm.engine import Engine
from engine_helpers import CpuFakeExecutor
from tinyvllm.scheduler import Scheduler
from benchmarks.tools.validate_workload import validate


def generate(tmp_path, monkeypatch, *args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_smoke_workload", *args])
    main()
    return tmp_path / "benchmarks/workloads"


@pytest.mark.parametrize("profile", [False, True])
def test_generated_workload_matches_checked_in_manifest(tmp_path, monkeypatch, profile):
    generated = generate(tmp_path, monkeypatch, *(["--profile-mixed"] if profile else []))
    name = "profile-mixed-qwen3-v1" if profile else "smoke-synth-qwen3-v1"
    reference = Path(__file__).resolve().parents[1] / "benchmarks/workloads"
    for extension in ("json", "jsonl"):
        filename = f"{name}.{extension}"
        assert (generated / filename).read_bytes() == (reference / filename).read_bytes()
    assert validate(generated / f"{name}.json")["record_count"] == (4 if profile else 8)


def test_profile_workload_exercises_all_step_modes(tmp_path, monkeypatch):
    generated = generate(tmp_path, monkeypatch, "--profile-mixed")
    records = [
        json.loads(line)
        for line in (generated / "profile-mixed-qwen3-v1.jsonl")
        .read_bytes().splitlines()
    ]
    scheduler = Scheduler(
        max_active_requests=4,
        max_step_requests=4,
        max_scheduled_tokens=128,
        prefill_chunk_size=64,
        max_model_len=1024,
        prefill_reserve_tokens=0,
        prefill_reserve_requests=0,
        kv_block_size=16,
        kv_num_blocks=128,
        eos_token_ids=(151645,),
    )
    pool = scheduler.block_manager.pool
    probe = ExecutionProbe(CpuFakeExecutor(token_id=7), pool)
    engine = Engine(scheduler, probe)

    result = replay(engine, records)

    assert result["completed_requests"] == 4
    assert result["output_tokens"] == 64
    assert all(row["token_ids"] == [7] * 16 for row in result["requests"])
    assert set(probe.steps_by_mode) == {"PREFILL_ONLY", "MIXED", "DECODE_ONLY"}
    assert all(count > 0 for count in probe.steps_by_mode.values())
    assert pool.free_block_count == pool.num_blocks
    assert not engine.has_unfinished()
    assert not scheduler.request_ids


@pytest.mark.parametrize("arrival_ns", [-1, 0])
def test_workload_arrival_lower_bound(tmp_path, arrival_ns):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "benchmarks/workloads/smoke-synth-qwen3-v1.json").read_text())
    records = [json.loads(line) for line in (root / "benchmarks/workloads" / manifest["records_file"]).read_text().splitlines()]
    records[0]["arrival_ns"] = arrival_ns
    data = b"".join((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode() for record in records)
    (tmp_path / manifest["records_file"]).write_bytes(data)
    manifest["records_sha256"] = hashlib.sha256(data).hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    if arrival_ns < 0:
        with pytest.raises(ValueError, match="invalid arrival"):
            validate(path)
    else:
        assert validate(path)["record_count"] == 8
