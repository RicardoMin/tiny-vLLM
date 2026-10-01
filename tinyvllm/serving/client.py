from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from collections.abc import Callable

import aiohttp

from tinyvllm.serving.terminal import chat_loop


async def _sse_data(content):
    """Reassemble SSE events independently of TCP/HTTP chunk boundaries."""
    buffer = b""
    fields = []
    size = 0
    first_line = True

    while True:
        block = await content.read(4096)
        buffer += block
        while match := re.search(rb"\r\n|\r|\n", buffer):
            if block and match.group() == b"\r" and match.end() == len(buffer):
                break
            line = buffer[:match.start()].decode("utf-8")
            buffer = buffer[match.end():]
            if first_line:
                line = line.removeprefix("\ufeff")
                first_line = False

            if not line:
                if fields:
                    yield "\n".join(fields)
                    fields = []
                size = 0
            elif line == "data" or line.startswith("data:"):
                value = line[5:] if line.startswith("data:") else ""
                fields.append(value.removeprefix(" "))
                size += len(line.encode("utf-8"))

            if size > 1024**2:
                raise ValueError("SSE event exceeds 1 MiB")

        if size + len(buffer) > 1024**2:
            raise ValueError("SSE event exceeds 1 MiB")
        if not block:
            if buffer or fields:
                raise ValueError("incomplete SSE event")
            return
        
        
async def stream_chat(
    base_url: str,
    body: dict,
    *,
    timeout: float,
    on_text: Callable[[str], None],
) -> str:
    pieces = []
    finished = False
    done = False
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as session:
        async with session.post(
            base_url.rstrip("/") + "/chat/completions",
            json=body,
        ) as response:
            if response.status != 200:
                detail = await response.text()
                try:
                    message = json.loads(detail)["error"]["message"]
                except (ValueError, KeyError, TypeError):
                    message = response.reason
                raise ValueError(f"HTTP {response.status}: {message}")
            if response.content_type != "text/event-stream":
                raise ValueError("expected a text/event-stream response")

            async for data in _sse_data(response.content):
                if done:
                    raise ValueError("received data after [DONE]")
                if data == "[DONE]":
                    if not finished:
                        raise ValueError("[DONE] arrived before finish_reason")
                    done = True
                    continue

                chunk = json.loads(data)
                if not isinstance(chunk, dict):
                    raise ValueError("expected a JSON stream object")
                if "error" in chunk:
                    raise ValueError(f"server error: {chunk['error']}")

                choices = chunk.get("choices")
                if choices == [] and finished and isinstance(chunk.get("usage"), dict):
                    continue
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError("expected exactly one chat choice")
                choice = choices[0]
                if not isinstance(choice, dict) or choice.get("index") != 0:
                    raise ValueError("expected chat choice index 0")
                if finished:
                    raise ValueError("received a choice after finish_reason")

                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    raise ValueError("expected a chat delta")
                if delta.get("tool_calls") or delta.get("function_call"):
                    raise ValueError("this client supports text chat only")
                text = delta.get("content")
                if text is not None:
                    if not isinstance(text, str):
                        raise ValueError("chat content must be text")
                    if text:
                        on_text(text)
                        pieces.append(text)

                reason = choice.get("finish_reason")
                if reason is not None:
                    if reason not in ("stop", "length"):
                        raise ValueError(f"unsupported finish_reason: {reason}")
                    finished = True

    if not done:
        raise ValueError("stream ended without [DONE]")
    return "".join(pieces)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Chat with a running tiny-vLLM server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3-0.6B", help="served model name")
    parser.add_argument("--system", default="")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")

    with asyncio.Runner() as runner:
        def turn(messages, on_text):
            body = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "enable_thinking": args.thinking,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            return runner.run(stream_chat(
                args.base_url,
                body,
                timeout=args.timeout,
                on_text=on_text,
            ))

        chat_loop(
            turn,
            system=args.system,
            title=f"tiny-vLLM | HTTP | {args.model}",
            errors=(aiohttp.ClientError, TimeoutError, ValueError),
        )