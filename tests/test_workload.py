"""The benchmark clock starts at zero and arrivals must be non-negative."""

import hashlib
import json
from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize("arrival_ns", [-1, 0])
def test_workload_arrival_lower_bound(tmp_path, arrival_ns):
    root = Path(__file__).resolve().parents[1]
    validate = runpy.run_path(str(root / "benchmarks/tools/validate_workload.py"))["validate"]
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
