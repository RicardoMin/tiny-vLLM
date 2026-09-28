"""Opt-in: real localhost HTTP -> tokenizer -> owner -> model -> SSE."""

import asyncio
import json

import pytest

aiohttp = pytest.importorskip("aiohttp", reason="install tinyvllm[server]")
from aiohttp import web

from tinyvllm.server import create_app


@pytest.mark.cuda
@pytest.mark.model
@pytest.mark.parametrize("backend", [
    "reference", "cuda", pytest.param("flashinfer", marks=pytest.mark.flashinfer),
])
def test_real_http_stream_matches_complete_text_and_usage(small_model_config):
    async def run():
        runner = web.AppRunner(
            create_app(small_model_config),
            handler_cancellation=True,
            shutdown_timeout=0.1,
        )
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        try:
            async with aiohttp.ClientSession() as client:
                for route, input_fields in (
                    ("completions", {"prompt": "你好，请简短介绍你自己。"}),
                    ("chat/completions", {"messages": [
                        {"role": "user", "content": "请计算 1+1。"},
                    ]}),
                ):
                    body = {"model": "Qwen3-0.6B", "max_tokens": 8, **input_fields}
                    async with client.post(f"{url}/v1/{route}", json=body) as response:
                        assert response.status == 200
                        complete = await response.json()
                    async with client.post(f"{url}/v1/{route}", json={**body, "stream": True}) as response:
                        assert response.status == 200
                        frames = []
                        async for line in response.content:
                            if line.startswith(b"data: "):
                                frames.append(line[6:].strip())
                    assert frames[-1] == b"[DONE]" and frames.count(b"[DONE]") == 1
                    chunks = [json.loads(frame) for frame in frames[:-1]]
                    assert not any("error" in chunk for chunk in chunks)
                    chat = route.startswith("chat/")
                    text = "".join(
                        chunk["choices"][0].get("delta", {}).get("content", "")
                        if chat else chunk["choices"][0]["text"]
                        for chunk in chunks
                    )
                    choice = complete["choices"][0]
                    assert text == (choice["message"]["content"] if chat else choice["text"])
                    assert chunks[-1]["usage"] == complete["usage"]
                    assert chunks[-1]["choices"][0]["finish_reason"] == choice["finish_reason"]
                    assert all(chunk["choices"][0]["finish_reason"] is None for chunk in chunks[:-1])
                    print(small_model_config.attention_backend, route, repr(text), complete["usage"])
        finally:
            await runner.cleanup()

    asyncio.run(asyncio.wait_for(run(), 120))
