import json
from pathlib import Path
import sys

import pytest

from benchmarks.tools import run_tiny
from benchmarks.tools.validate_workload import validate


def test_checked_in_workload_matches_its_manifest():
    root = Path(__file__).resolve().parents[1]
    result = validate(root / "benchmarks/workloads/smoke-synth-qwen3-v1.json")
    assert result["record_count"] == 8


@pytest.mark.parametrize("failure, status", [(RuntimeError("load failed"), "failed"), (KeyboardInterrupt(), "interrupted")])
def test_bootstrap_failure_is_reported_and_reraised(tmp_path, monkeypatch, failure, status):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "run_tiny", "--model", str(tmp_path),
        "--workload", str(root / "benchmarks/workloads/smoke-synth-qwen3-v1.json"),
        "--output", str(output),
        "--attention-backend", "cuda",
    ])
    monkeypatch.setattr(run_tiny, "git_identity", lambda path: {})
    monkeypatch.setattr(run_tiny, "model_manifest", lambda path: {})
    monkeypatch.setattr(run_tiny, "packages", lambda: [])

    def fail(config):
        assert config.attention_backend == "cuda"
        raise failure

    monkeypatch.setattr(run_tiny.Engine, "from_config", fail)
    with pytest.raises(type(failure)):
        run_tiny.main()
    report = json.loads(output.read_text())
    assert report["status"] == status
    assert report["phase"] == "bootstrap"
    assert report["runs"] == []
    assert report["config"]["attention_backend"] == "cuda"
