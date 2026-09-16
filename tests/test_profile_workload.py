import hashlib
import json
from pathlib import Path
import sys

from benchmarks.tools.make_smoke_workload import main
from benchmarks.tools.measurement import ExecutionProbe, replay
from tinyvllm.engine import Engine
from tinyvllm.executor import CpuFakeExecutor
from tinyvllm.scheduler import Scheduler
from benchmarks.tools.validate_workload import validate


def generate(tmp_path, monkeypatch, *args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_smoke_workload", *args])
    main()
    return tmp_path / "benchmarks/workloads"


def test_default_workload_remains_byte_identical(tmp_path, monkeypatch):
    generated = generate(tmp_path, monkeypatch)
    reference = Path(__file__).resolve().parents[1] / "benchmarks/workloads"
    hashes = {
        "json": "f29af634fca5c55b7ed4ce4b128d28c3a173d8b7f96b25791cc516f37d784c57",
        "jsonl": "b6490978bae086647778be7124b2050030562d7cb6c5cafb49b4dde622c0dc9a",
    }

    for extension, expected in hashes.items():
        filename = f"smoke-synth-qwen3-v1.{extension}"
        data = (generated / filename).read_bytes()
        assert data == (reference / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected


def test_profile_workload_contract(tmp_path, monkeypatch):
    generated = generate(tmp_path, monkeypatch, "--profile-mixed")
    manifest_path = generated / "profile-mixed-qwen3-v1.json"
    summary = validate(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    records = [
        json.loads(line)
        for line in (generated / manifest["records_file"]).read_bytes().splitlines()
    ]

    assert summary["record_count"] == 4
    assert manifest["workload_id"] == "profile-mixed-qwen3-v1"
    assert manifest["suite"] == "profile"
    assert manifest["totals"] == {
        "prompt_tokens": 912,
        "requested_output_tokens": 64,
    }
    assert [len(record["prompt_token_ids"]) for record in records] == [
        16, 128, 256, 512,
    ]
    assert all(record["arrival_ns"] == 0 for record in records)
    assert all(record["max_new_tokens"] == 16 for record in records)
    assert all(record["temperature"] == 0.6 for record in records)
    assert all(record["ignore_eos"] is True for record in records)


def test_profile_workload_is_reproducible(tmp_path, monkeypatch):
    generated = generate(tmp_path, monkeypatch, "--profile-mixed")
    first = {path.name: path.read_bytes() for path in generated.iterdir()}

    generate(tmp_path, monkeypatch, "--profile-mixed")
    second = {path.name: path.read_bytes() for path in generated.iterdir()}

    assert first == second
    reference = Path(__file__).resolve().parents[1] / "benchmarks/workloads"
    assert all(data == (reference / name).read_bytes() for name, data in first.items())


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
