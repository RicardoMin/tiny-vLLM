import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.tools import run_tiny


@pytest.mark.parametrize(
    "body_error,cleanup_error",
    [
        (None, None),
        (ValueError("replay failed"), None),
        (KeyboardInterrupt(), None),
        (None, RuntimeError("cleanup failed")),
        (ValueError("replay failed"), RuntimeError("cleanup failed")),
    ],
)
def test_engine_session_closes_once_and_preserves_original_error(
    monkeypatch, body_error, cleanup_error,
):
    engine = SimpleNamespace(close=Mock(side_effect=cleanup_error))
    monkeypatch.setattr(run_tiny.Engine, "from_config", lambda config: engine)

    def run():
        with run_tiny._engine_session(object()) as actual:
            assert actual is engine
            if body_error is not None:
                raise body_error

    expected = body_error if body_error is not None else cleanup_error
    if expected is None:
        run()
    else:
        with pytest.raises(type(expected)) as caught:
            run()
        assert caught.value is expected
        if body_error is not None and cleanup_error is not None:
            assert any("cleanup failed" in note for note in caught.value.__notes__)
    engine.close.assert_called_once_with()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_manifest_mismatch_closes_loaded_engine_and_keeps_failure_report(
    tmp_path, monkeypatch, cleanup_fails,
):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "run_tiny", "--model", str(tmp_path),
        "--workload", str(root / "benchmarks/workloads/smoke-synth-qwen3-v1.json"),
        "--output", str(output),
    ])
    monkeypatch.setattr(run_tiny, "git_identity", lambda path: {})
    monkeypatch.setattr(run_tiny, "model_manifest", lambda path: {})
    monkeypatch.setattr(run_tiny, "packages", lambda: [])
    engine = SimpleNamespace(
        executor=SimpleNamespace(
            runner=SimpleNamespace(kv_cache=None, device="cpu"),
            capabilities=SimpleNamespace(vocab_size=-1),
        ),
        close=Mock(side_effect=RuntimeError("cleanup failed") if cleanup_fails else None),
    )
    monkeypatch.setattr(run_tiny.Engine, "from_config", lambda config: engine)

    with pytest.raises(ValueError, match="vocabulary") as caught:
        run_tiny.main()
    engine.close.assert_called_once_with()
    if cleanup_fails:
        assert any("cleanup failed" in note for note in caught.value.__notes__)
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["phase"] == "bootstrap"
    assert "vocabulary" in report["error"]


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
