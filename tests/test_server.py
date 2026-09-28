from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import threading
import time

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web

from engine_helpers import make_engine
from tinyvllm.async_engine import WorkerError
from tinyvllm.batch import StepResult
from tinyvllm.config import EngineConfig
from tinyvllm.engine import Engine
from tinyvllm.executor import ExecutorCapabilities
from tinyvllm.tokenizer import StreamingDecoder, Tokenizer
from tinyvllm import server


class Harness:
    """Real engine/bridge/HTTP lifecycle with deterministic CPU execution."""

    def __init__(self, monkeypatch, tmp_path, *, eos_token_ids=()):
        self.config = EngineConfig(
            model=tmp_path,
            max_model_len=16,
            max_num_seqs=4,
            max_num_batched_tokens=8,
            prefill_chunk_size=2,
            prefill_reserve_tokens=1,
            prefill_reserve_requests=1,
        )
        self.tokens = (7, 8, 9)
        self.token_bytes = {7: b"x", 8: b"y", 9: b"z"}
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.encode_entered = threading.Event()
        self.encode_resume = threading.Event()
        self.cancelled = threading.Event()
        self.closed = threading.Event()
        self.block = False
        self.block_encoding = False
        self.fail = False
        self.bridge = None
        self.submissions = []
        self.chat_calls = []
        harness = self

        class Backend:
            def encode(self, text, **kwargs):
                if harness.block_encoding:
                    harness.encode_entered.set()
                    if not harness.encode_resume.wait(5):
                        raise RuntimeError("HTTP tokenizer gate was not released")
                return [1] * len(text)

            def apply_chat_template(self, messages, **kwargs):
                harness.chat_calls.append((messages, kwargs))
                return [1, 2]

            def decode(self, token_ids, **kwargs):
                return b"".join(
                    harness.token_bytes.get(token_id, b"")
                    for token_id in token_ids
                ).decode("utf-8", errors="replace")

        class FakeTokenizer(Tokenizer):
            def __init__(self, model):
                self._tokenizer = Backend()

            def stream_decoder(self):
                return StreamingDecoder(harness.token_bytes)

        class Executor:
            capabilities = ExecutorCapabilities(vocab_size=32)

            def __init__(self):
                self.positions = {}

            def execute(self, batch):
                if harness.block:
                    harness.block = False
                    harness.entered.set()
                    if not harness.resume.wait(5):
                        raise RuntimeError("HTTP test gate was not released")
                if harness.fail:
                    raise RuntimeError("injected HTTP execution failure")
                tokens = []
                for request_id in batch.sample_request_ids:
                    position = self.positions.get(request_id, 0)
                    tokens.append(harness.tokens[min(position, len(harness.tokens) - 1)])
                    self.positions[request_id] = position + 1
                return StepResult(batch.plan_id, batch.sample_request_ids, tuple(tokens))

            def release(self, request_id):
                self.positions.pop(request_id, None)

            def clear(self):
                self.positions.clear()
                harness.closed.set()

        def factory(cls, config):
            engine = make_engine(Executor(), eos_token_ids=eos_token_ids, clock=time.monotonic_ns)
            harness.engine = engine
            original_submit = engine.submit
            original_cancel = engine.cancel

            def submit(prompt, sampling, **kwargs):
                harness.submissions.append((prompt, sampling))
                return original_submit(prompt, sampling, **kwargs)

            def cancel(request_id):
                harness.cancelled.set()
                return original_cancel(request_id)

            engine.submit = submit
            engine.cancel = cancel
            return engine

        original_bridge = server.AsyncEngine

        def capture_bridge(*args, **kwargs):
            harness.bridge = original_bridge(*args, **kwargs)
            return harness.bridge

        monkeypatch.setattr(Engine, "from_config", classmethod(factory))
        monkeypatch.setattr(server, "Tokenizer", FakeTokenizer)
        monkeypatch.setattr(server, "AsyncEngine", capture_bridge)

        original_worker = server._TokenizerWorker

        def capture_worker(*args, **kwargs):
            harness.cpu = original_worker(*args, **kwargs)
            return harness.cpu

        monkeypatch.setattr(server, "_TokenizerWorker", capture_worker)

    @asynccontextmanager
    async def serve(self, **options):
        app = server.create_app(self.config, **options)
        runner = web.AppRunner(app, handler_cancellation=True, shutdown_timeout=0.05)
        self.runner = runner
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with aiohttp.ClientSession(
                base_url=f"http://127.0.0.1:{port}",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as client:
                yield client
        finally:
            self.resume.set()
            self.encode_resume.set()
            if self.fail:
                with pytest.raises(WorkerError, match="injected HTTP execution failure"):
                    await runner.cleanup()
            else:
                await runner.cleanup()
        assert self.closed.is_set()
        assert not self.bridge._thread.is_alive()
        assert not self.bridge._slots
        assert self.bridge._reserved_tokens == 0


def run(coroutine):
    async def bounded():
        return await asyncio.wait_for(coroutine, 15)

    return asyncio.run(bounded())


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


def body(**changes):
    return {"model": "Qwen3-0.6B", "prompt": "hi", "max_tokens": 3, **changes}


async def read_sse(response):
    assert response.status == 200
    assert response.content_type == "text/event-stream"
    data = []
    async for line in response.content:
        if line.startswith(b"data: "):
            data.append(line[6:].decode().strip())
    assert data[-1] == "[DONE]"
    assert data.count("[DONE]") == 1
    return [json.loads(item) for item in data[:-1]]


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_http_generation_preserves_all_tokens_and_sampling(monkeypatch, tmp_path, chat, stream):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with harness.serve() as client:
            async with client.get("/health") as response:
                assert response.status == 200
            async with client.get("/v1/models") as response:
                assert (await response.json())["data"][0]["id"] == "Qwen3-0.6B"

            payload = body(stream=stream, temperature=0.7, top_p=0.9, top_k=5, seed=11, n=1)
            path = "/v1/completions"
            if chat:
                payload.pop("prompt")
                payload.update(messages=[{"role": "user", "content": "hi"}], enable_thinking=True)
                path = "/v1/chat/completions"
            async with client.post(path, json=payload) as response:
                if stream:
                    chunks = await read_sse(response)
                    choices = [choice for chunk in chunks for choice in chunk["choices"]]
                    text = "".join(
                        choice["delta"].get("content", "") if chat else choice["text"]
                        for choice in choices
                    )
                    assert text == "xyz"
                    assert [choice["finish_reason"] for choice in choices if choice["finish_reason"]] == ["length"]
                    assert len({chunk["id"] for chunk in chunks}) == 1
                    result = chunks[-1]
                else:
                    assert response.status == 200
                    result = await response.json()
                    choice = result["choices"][0]
                    assert (choice["message"]["content"] if chat else choice["text"]) == "xyz"
                    assert choice["finish_reason"] == "length"
                assert result["usage"] == {
                    "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
                }
                assert result["model"] == "Qwen3-0.6B"
            sampling = harness.submissions[0][1]
            assert (sampling.max_new_tokens, sampling.temperature, sampling.top_p, sampling.top_k, sampling.seed) == (3, 0.7, 0.9, 5, 11)
            if chat:
                assert harness.chat_calls[0][1]["enable_thinking"] is True

    run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_http_zero_output_eos_and_ignored_eos(monkeypatch, tmp_path, stream):
    harness = Harness(monkeypatch, tmp_path, eos_token_ids=(8,))
    harness.token_bytes.pop(8)

    async def scenario():
        async with harness.serve() as client:
            for max_tokens, ignore_eos, text, count, reason in (
                (0, False, "", 0, "length"),
                (3, False, "x", 2, "stop"),
                (3, True, "xz", 3, "length"),
            ):
                async with client.post("/v1/completions", json=body(
                    stream=stream, max_tokens=max_tokens, ignore_eos=ignore_eos,
                )) as response:
                    if stream:
                        chunks = await read_sse(response)
                        choices = [choice for chunk in chunks for choice in chunk["choices"]]
                        assert "".join(choice["text"] for choice in choices) == text
                        assert [choice["finish_reason"] for choice in choices if choice["finish_reason"]] == [reason]
                        result = chunks[-1]
                    else:
                        assert response.status == 200
                        result = await response.json()
                        assert result["choices"][0]["text"] == text
                        assert result["choices"][0]["finish_reason"] == reason
                    assert result["usage"]["completion_tokens"] == count

    run(scenario())


def test_http_input_boundary_rejects_invalid_and_unsupported_requests(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with harness.serve() as client:
            invalid = [
                [], {}, body(model="wrong"), body(prompt=""), body(prompt="x" * 17),
                body(stream=1), body(n=2), body(n=True), body(max_tokens=True),
                body(temperature=-1), body(top_p=0), body(stop=["x"]),
            ]
            for payload in invalid:
                async with client.post("/v1/completions", json=payload) as response:
                    assert response.status == 400, payload
                    error = (await response.json())["error"]
                    assert error["message"] and error["type"]
            async with client.post("/v1/chat/completions", json={
                "model": "Qwen3-0.6B", "messages": [{"role": "tool", "content": "x"}],
            }) as response:
                assert response.status == 400
            for data, status in (("{broken", 400), ("x" * (1024**2 + 1), 413)):
                async with client.post("/v1/completions", data=data) as response:
                    assert response.status == status
                    assert (await response.json())["error"]["message"]
            assert not harness.submissions

    run(scenario())


def test_http_overload_recovers_after_request_finishes(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with harness.serve(max_requests=1) as client:
            harness.block = True
            first = asyncio.create_task(client.post("/v1/completions", json=body()))
            await until(harness.entered.is_set)
            async with client.post("/v1/completions", json=body()) as response:
                assert response.status == 429
                assert (await response.json())["error"]["message"]
            harness.resume.set()
            async with await first as response:
                assert response.status == 200
                assert (await response.json())["choices"][0]["text"] == "xyz"
            async with client.post("/v1/completions", json=body()) as response:
                assert response.status == 200

    run(scenario())


def test_cancelled_tokenizer_waiter_keeps_native_work_bounded(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with harness.serve(max_requests=1) as client:
            harness.block_encoding = True
            request = asyncio.create_task(client.post("/v1/completions", json=body()))
            await until(harness.encode_entered.is_set)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert harness.cpu.pending == 1
            for _ in range(3):
                async with client.post("/v1/completions", json=body()) as response:
                    assert response.status == 429
                    assert (await response.json())["error"]["message"]
            assert harness.cpu.pending == 1
            assert not harness.submissions
            harness.block_encoding = False
            harness.encode_resume.set()
            await until(lambda: harness.cpu.pending == 0)
            async with client.post("/v1/completions", json=body()) as response:
                assert response.status == 200

    run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_http_disconnect_cancels_request_and_returns_credit(monkeypatch, tmp_path, stream):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with harness.serve(max_requests=1) as client:
            harness.block = True
            request = asyncio.create_task(client.post("/v1/completions", json=body(stream=stream)))
            await until(harness.entered.is_set)
            if stream:
                response = await request
                response.close()
            else:
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
            await until(lambda: all(slot.released for slot in harness.bridge._slots.values()))
            harness.resume.set()
            await until(harness.cancelled.is_set)
            await until(lambda: not harness.bridge._slots)
            assert harness.bridge._reserved_tokens == 0
            async with client.post("/v1/completions", json=body()) as response:
                assert response.status == 200

    run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_http_timeout_cancels_and_reports_protocol_error(monkeypatch, tmp_path, stream):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with harness.serve(request_timeout=0.1) as client:
            harness.block = True
            async with client.post("/v1/completions", json=body(stream=stream)) as response:
                if stream:
                    chunks = await read_sse(response)
                    assert chunks[-1]["error"]["type"]
                else:
                    assert response.status == 504
                    assert (await response.json())["error"]["type"]
            harness.resume.set()
            await until(harness.cancelled.is_set)
            await until(lambda: not harness.bridge._slots)
            assert harness.bridge._reserved_tokens == 0

    run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_http_worker_failure_reaches_client_and_health(monkeypatch, tmp_path, stream):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with harness.serve() as client:
            harness.fail = True
            async with client.post("/v1/completions", json=body(stream=stream)) as response:
                if stream:
                    chunks = await read_sse(response)
                    assert chunks[-1]["error"]["type"]
                else:
                    assert response.status == 503
                    assert (await response.json())["error"]["type"]
            async with client.get("/health") as response:
                assert response.status == 503
            async with client.post("/v1/completions", json=body()) as response:
                assert response.status == 503

    run(scenario())


def test_http_shutdown_cancels_active_handler_before_join(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        async with harness.serve() as client:
            harness.block = True
            response = await client.post("/v1/completions", json=body(stream=True))
            await until(harness.entered.is_set)
            closing = asyncio.create_task(harness.runner.cleanup())
            await until(lambda: all(slot.released for slot in harness.bridge._slots.values()))
            assert harness.bridge._thread.is_alive()
            harness.resume.set()
            await closing
            response.close()
            assert not harness.bridge._thread.is_alive()
            assert harness.closed.is_set()

    run(scenario())


def test_json_send_keeps_request_admission_until_body_is_written(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        entered = asyncio.Event()
        resume = asyncio.Event()
        original = web.Response.write_eof

        async def gated_eof(response, data=b""):
            if b'"choices"' in (response.body or b"") and not entered.is_set():
                entered.set()
                await resume.wait()
            return await original(response, data)

        monkeypatch.setattr(web.Response, "write_eof", gated_eof)
        async with harness.serve(max_requests=1) as client:
            first = asyncio.create_task(client.post("/v1/completions", json=body()))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                async with client.post("/v1/completions", json=body()) as response:
                    assert response.status == 429
            finally:
                resume.set()
            async with await first as response:
                assert (await response.json())["choices"][0]["text"] == "xyz"

    run(scenario())


def test_stream_eof_timeout_never_emits_a_second_terminal(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        original = web.StreamResponse.write_eof
        attempted = False

        async def stalled_eof(response, data=b""):
            nonlocal attempted
            if response.content_type == "text/event-stream" and not attempted:
                attempted = True
                await asyncio.Future()
            return await original(response, data)

        monkeypatch.setattr(web.StreamResponse, "write_eof", stalled_eof)
        async with harness.serve(request_timeout=0.1) as client:
            async with client.post("/v1/completions", json=body(stream=True)) as response:
                data = bytearray()
                try:
                    async for chunk in response.content.iter_any():
                        data.extend(chunk)
                except aiohttp.ClientPayloadError:
                    # The deadline may abort TCP after DONE but before chunked EOF.
                    pass
                assert attempted
                assert bytes(data).count(b"data: [DONE]") == 1
                assert bytes(data).count(b'"finish_reason": "length"') == 1
                assert b'"error"' not in data

    run(scenario())


def test_slow_stream_overflow_isolated_from_other_http_requests(monkeypatch, tmp_path):
    harness = Harness(monkeypatch, tmp_path)

    async def scenario():
        role_entered = asyncio.Event()
        send_resume = asyncio.Event()
        original = server._send

        async def gated_role(response, data):
            if (
                isinstance(data, dict)
                and data.get("choices", [{}])[0].get("delta") == {"role": "assistant"}
            ):
                role_entered.set()
                await send_resume.wait()
            return await original(response, data)

        monkeypatch.setattr(server, "_send", gated_role)
        async with harness.serve(max_requests=2, max_buffered_tokens=1) as client:
            harness.block = True
            slow = await client.post("/v1/chat/completions", json={
                "model": "Qwen3-0.6B",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "max_tokens": 3,
            })
            try:
                await asyncio.wait_for(role_entered.wait(), 5)
                await until(harness.entered.is_set)
                harness.resume.set()
                await until(lambda: all(
                    slot.retired for slot in harness.bridge._slots.values()
                ))
                async with client.post("/v1/completions", json=body(max_tokens=0)) as healthy:
                    assert healthy.status == 200
                    assert (await healthy.json())["usage"]["completion_tokens"] == 0
            finally:
                send_resume.set()
            chunks = await read_sse(slow)
            slow.close()
            assert chunks[-1]["error"]["type"] == "slow_consumer"
            await until(lambda: not harness.bridge._slots)
            assert harness.bridge._reserved_tokens == 0
            # No owner work remains, so observe reclamation before service shutdown.
            pool = harness.engine.scheduler.block_manager.pool
            assert pool.free_block_count == pool.num_blocks
            assert pool.open_reservation_count == 0

    run(scenario())
