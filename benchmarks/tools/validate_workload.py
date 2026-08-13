from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


RECORD_KEYS = {
    "arrival_ns",
    "ignore_eos",
    "max_new_tokens",
    "ordinal",
    "prompt_token_ids",
    "request_id",
    "temperature",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_json(data: bytes):
    def unique(items):
        result = {}
        for key, value in items:
            check(key not in result, f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(
        data.decode("utf-8"),
        object_pairs_hook=unique,
        parse_constant=reject_constant,
    )


def canonical(record: dict) -> bytes:
    text = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return (text + "\n").encode()


def validate(manifest_path: Path) -> dict:
    manifest = strict_json(manifest_path.read_bytes())

    check(
        manifest["schema_version"] == "tinyvllm.workload.v1",
        "bad schema",
    )
    check(
        type(manifest["vocab_size"]) is int
        and manifest["vocab_size"] > 0,
        "bad vocabulary size",
    )
    check(
        type(manifest["record_count"]) is int
        and manifest["record_count"] > 0,
        "bad record count",
    )
    check(
        manifest["arrival"] == {
            "clock": "monotonic_ns_relative",
            "origin": "runner_start",
            "sorted": True,
        },
        "bad arrival contract",
    )

    eos_ids = manifest["eos_token_ids"]
    check(
        type(eos_ids) is list
        and eos_ids
        and len(eos_ids) == len(set(eos_ids))
        and all(
            type(token) is int
            and 0 <= token < manifest["vocab_size"]
            for token in eos_ids
        ),
        "bad EOS token IDs",
    )

    source = manifest["source"]
    token_min = source["token_min_inclusive"]
    token_max = source["token_max_exclusive"]

    check(
        source["kind"] == "synthetic_uniform"
        and type(token_min) is int
        and type(token_max) is int
        and 0 <= token_min < token_max <= manifest["vocab_size"],
        "bad token source",
    )

    records_name = manifest["records_file"]
    check(
        type(records_name) is str
        and Path(records_name).name == records_name,
        "bad records filename",
    )

    records_bytes = (manifest_path.parent / records_name).read_bytes()

    check(records_bytes.endswith(b"\n"), "missing final LF")
    check(
        not records_bytes.startswith(b"\xef\xbb\xbf"),
        "UTF-8 BOM is forbidden",
    )

    digest = hashlib.sha256(records_bytes).hexdigest()
    check(
        digest == manifest["records_sha256"],
        "records SHA-256 mismatch",
    )

    records = []
    previous_arrival = -1

    for ordinal, line in enumerate(
        records_bytes.splitlines(keepends=True)
    ):
        check(
            line != b"\n" and line.endswith(b"\n"),
            f"record {ordinal}: bad line",
        )

        record = strict_json(line[:-1])

        check(
            line == canonical(record),
            f"record {ordinal}: non-canonical JSON",
        )
        check(
            type(record) is dict and set(record) == RECORD_KEYS,
            f"record {ordinal}: wrong fields",
        )
        check(
            record["request_id"] == f"req-{ordinal:06d}",
            f"record {ordinal}: bad request ID",
        )
        check(
            type(record["ordinal"]) is int
            and record["ordinal"] == ordinal,
            f"record {ordinal}: ordinal mismatch",
        )
        check(
            type(record["arrival_ns"]) is int
            and record["arrival_ns"] >= previous_arrival,
            f"record {ordinal}: invalid arrival",
        )
        check(
            type(record["max_new_tokens"]) is int
            and record["max_new_tokens"] > 0,
            f"record {ordinal}: invalid output length",
        )
        check(
            type(record["ignore_eos"]) is bool,
            f"record {ordinal}: ignore_eos must be bool",
        )

        temperature = record["temperature"]
        check(
            type(temperature) in (int, float)
            and math.isfinite(temperature)
            and temperature > 1e-10,
            f"record {ordinal}: invalid temperature",
        )

        prompt = record["prompt_token_ids"]
        check(
            type(prompt) is list and prompt,
            f"record {ordinal}: empty prompt",
        )
        check(
            all(
                type(token) is int
                and token_min <= token < token_max
                for token in prompt
            ),
            f"record {ordinal}: invalid token",
        )
        check(
            temperature == manifest["sampling"]["temperature"],
            f"record {ordinal}: temperature mismatch",
        )
        check(
            record["ignore_eos"]
            is manifest["sampling"]["ignore_eos"],
            f"record {ordinal}: EOS policy mismatch",
        )

        previous_arrival = record["arrival_ns"]
        records.append(record)

    totals = {
        "prompt_tokens": sum(
            len(record["prompt_token_ids"])
            for record in records
        ),
        "requested_output_tokens": sum(
            record["max_new_tokens"]
            for record in records
        ),
    }

    check(
        len(records) == manifest["record_count"],
        "record count mismatch",
    )
    check(totals == manifest["totals"], "token totals mismatch")

    return {
        "workload_id": manifest["workload_id"],
        "records_sha256": digest,
        "record_count": len(records),
        **totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    try:
        print(json.dumps(validate(args.manifest), sort_keys=True))
        return 0
    except (
        OSError,
        UnicodeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(f"workload error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())