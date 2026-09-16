from __future__ import annotations

import math
import time
from collections import Counter

from benchmarks.tools.profiling import nvtx_range

from tinyvllm.request import RequestStatus, SamplingParams


def distribution(values):
    """Nearest-rank percentiles; no interpolation or averaging of runs."""
    values = sorted(values)

    if not values:
        return {"count": 0, "p50": None, "p95": None}

    return {
        "count": len(values),
        "p50": values[math.ceil(0.50 * len(values)) - 1],
        "p95": values[math.ceil(0.95 * len(values)) - 1],
    }


class ExecutionProbe:
    """Benchmark-only observation, after block dispatch and before release."""

    def __init__(self, executor, pool):
        self.executor = executor
        self.pool = pool
        self.capabilities = executor.capabilities
        self.nvtx = False
        self.reset()

    def reset(self):
        self.peak_used_blocks = 0
        self.steps_by_mode = Counter()

    def execute(self, batch):
        used = self.pool.num_blocks - self.pool.free_block_count

        self.peak_used_blocks = max(self.peak_used_blocks, used)
        self.steps_by_mode[batch.mode.name] += 1

        name = f"tinyvllm/execute/{batch.mode.name.lower()}"

        with nvtx_range(name, self.nvtx):
            return self.executor.execute(batch)

    def release(self, request_id):
        self.executor.release(request_id)

    def clear(self):
        self.executor.clear()


def replay(
    engine,
    records,
    *,
    seed=0,
    nvtx=False,
    clock=time.monotonic_ns,
    sleep=time.sleep,
):
    """Replay validated workload records on an idle engine owned by this caller.

    The injected clock must share the engine's monotonic time domain.
    Arrivals are scheduled independently of completion (open-loop).
    """
    if engine.has_unfinished() or not records:
        raise ValueError("replay requires an idle engine and nonempty records")

    prepared = []
    previous_arrival = -1

    for index, record in enumerate(records):
        if (
            record["ordinal"] != index
            or record["arrival_ns"] < max(0, previous_arrival)
        ):
            raise ValueError("records must have ordered ordinals and arrivals")

        prompt = tuple(record["prompt_token_ids"])
        sampling = SamplingParams(
            max_new_tokens=record["max_new_tokens"],
            temperature=record["temperature"],
            ignore_eos=record["ignore_eos"],
            seed=seed,
        )

        if len(prompt) + sampling.max_new_tokens > engine.scheduler.max_model_len:
            raise ValueError("benchmark output would be truncated by max_model_len")

        engine.scheduler.validate_request(prompt, sampling)
        engine.executor.capabilities.validate_token_ids(prompt)

        prepared.append((prompt, sampling))
        previous_arrival = record["arrival_ns"]

    rows = [
        {
            "request_id": record["request_id"],
            "arrival_ns": record["arrival_ns"],
            "submitted_ns": None,
            "finished_ns": None,
            "token_ids": [],
            "token_times_ns": [],
        }
        for record in records
    ]

    origin = clock()
    cursor = 0
    finished = set()
    last_observed = 0

    while cursor < len(records) or engine.has_unfinished():
        now = clock() - origin

        while cursor < len(records) and records[cursor]["arrival_ns"] <= now:
            prompt, sampling = prepared[cursor]
            rows[cursor]["submitted_ns"] = clock() - origin

            engine.submit(
                prompt,
                sampling,
                request_id=cursor,
                arrival_ns=origin + records[cursor]["arrival_ns"],
            )

            cursor += 1

        if not engine.has_unfinished():
            delay = (records[cursor]["arrival_ns"] - (clock() - origin)) / 1e9

            if delay > 0:
                sleep(delay)

            continue

        with nvtx_range("tinyvllm/step", nvtx):
            events = engine.step()
            observed = clock() - origin

        if observed < last_observed:
            raise RuntimeError("benchmark clock moved backwards")

        last_observed = observed

        def active_row(request_id):
            if not 0 <= request_id < cursor or request_id in finished:
                raise RuntimeError("event for unknown or completed request")

            return rows[request_id]

        sampled = set()

        for event in events.token_events:
            row = active_row(event.request_id)

            if event.request_id in sampled:
                raise RuntimeError("duplicate token event in one step")

            vocab_size = engine.executor.capabilities.vocab_size

            if type(event.token_id) is not int or event.token_id < 0:
                raise RuntimeError("invalid output token ID")

            if vocab_size is not None and event.token_id >= vocab_size:
                raise RuntimeError("output token ID exceeds vocabulary")

            sampled.add(event.request_id)
            row["token_ids"].append(event.token_id)
            row["token_times_ns"].append(observed)

        for event in events.completion_events:
            row = active_row(event.request_id)

            if tuple(row["token_ids"]) != event.output_token_ids:
                raise RuntimeError(
                    "token events disagree with complete output sequence"
                )

            if event.status is not RequestStatus.FINISHED:
                raise RuntimeError("benchmark request did not finish successfully")

            requested = records[event.request_id]["max_new_tokens"]
            count = len(row["token_ids"])

            if not 0 < count <= requested:
                raise RuntimeError("invalid output token count")

            if records[event.request_id]["ignore_eos"] and count != requested:
                raise RuntimeError("fixed-length request finished early")

            if not records[event.request_id]["ignore_eos"]:
                eos_ids = engine.scheduler.eos_token_ids
                tokens = row["token_ids"]

                if any(token in eos_ids for token in tokens[:-1]):
                    raise RuntimeError("tokens emitted after EOS")

                if count < requested and tokens[-1] not in eos_ids:
                    raise RuntimeError("early completion without EOS")

            row["finished_ns"] = observed
            finished.add(event.request_id)

    if len(finished) != len(records):
        raise RuntimeError("missing completion events")

    all_itl = []

    for row in rows:
        times = row["token_times_ns"]
        itl = [
            (right - left) / 1e6
            for left, right in zip(times, times[1:])
        ]

        row["ttft_ms"] = (times[0] - row["arrival_ns"]) / 1e6
        row["tpot_ms"] = (
            (times[-1] - times[0]) / (len(times) - 1) / 1e6
            if itl else None
        )
        row["e2e_ms"] = (row["finished_ns"] - row["arrival_ns"]) / 1e6

        all_itl.extend(itl)

    seconds = last_observed / 1e9
    output_tokens = sum(len(row["token_ids"]) for row in rows)

    return {
        "duration_ns": last_observed,
        "completed_requests": len(finished),
        "completion_rate": len(finished) / len(records),
        "output_tokens": output_tokens,
        "output_tokens_per_second": (
            output_tokens / seconds if seconds > 0 else None
        ),
        "ttft_ms": distribution(row["ttft_ms"] for row in rows),
        "itl_ms": distribution(all_itl),
        "tpot_ms": distribution(
            row["tpot_ms"] for row in rows if row["tpot_ms"] is not None
        ),
        "e2e_ms": distribution(row["e2e_ms"] for row in rows),
        "submit_lag_ms": distribution(
            (row["submitted_ns"] - row["arrival_ns"]) / 1e6
            for row in rows
        ),
        "requests": rows,
    }