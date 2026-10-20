from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from pathlib import Path
import time
import uuid

from aiohttp import web

from tinyvllm.async_engine import (
    OverloadedError,
    SlowConsumerError,
)
from tinyvllm.config import EngineConfig
from tinyvllm.serving.async_llm import AsyncLLM, TextDelta
from tinyvllm.request import SamplingParams


logger = logging.getLogger(__name__)

            
def _parse(body, *, chat: bool, model: str):
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")

    fields = {
        "model",
        "stream",
        "stream_options",
        "n",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "ignore_eos",
    }
    fields |= {"messages", "enable_thinking"} if chat else {"prompt"}
    
    if body.keys() - fields:
        raise ValueError(f"unsupported fields: {sorted(body.keys() - fields)}")

    if body.get("model") != model:
        raise ValueError(f"model must be {model!r}")
    
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
        raise ValueError("only n=1 is supported")
    
    if type(body.get("stream", False)) is not bool:
        raise ValueError("stream must be bool")

    options = body.get("stream_options")
    include_usage = False
    if options is not None:
        if not body.get("stream", False):
            raise ValueError("stream_options requires stream=true")
        if not isinstance(options, dict) or options.keys() - {"include_usage"}:
            raise ValueError("stream_options only supports include_usage")
        include_usage = options.get("include_usage", False)
        if type(include_usage) is not bool:
            raise ValueError("include_usage must be bool")

    sampling = SamplingParams(
        max_new_tokens=body.get("max_tokens", 64),
        temperature=body.get("temperature", 0.0),
        top_p=body.get("top_p", 1.0),
        top_k=body.get("top_k", 0),
        seed=body.get("seed", 0),
        ignore_eos=body.get("ignore_eos", False),
    )
    return sampling, body.get("stream", False), include_usage


def _error(message: str, kind: str) -> dict:
    return {"error": {"message": message, "type": kind}}

    
async def _send(response, data) -> None:
    payload = data if isinstance(data, str) else json.dumps(
        data,
        ensure_ascii=False,
    )
    await response.write(f"data: {payload}\n\n".encode("utf-8"))
    
    
def _abort_response(request, response) -> None:
    response.force_close()
    if request.transport is not None:
        request.transport.abort()
        
        
class _Server:
    def __init__(
        self,
        config,
        model,
        max_requests,
        max_reserved_tokens,
        max_buffered_tokens,
        request_timeout,
    ) -> None:
        self.model = model
        self.limit = max_requests
        self.timeout = request_timeout
        self.active = 0
        self.closing = False
        
        self.llm = AsyncLLM(
            config,
            max_requests=max_requests,
            max_reserved_tokens=max_reserved_tokens,
            max_buffered_tokens=max_buffered_tokens,
        )

        
    async def lifespan(self, app):
        async with self.llm:
            logger.info(
                "Text service ready: model=%s max_requests=%d request_timeout=%.1fs",
                self.model,
                self.limit,
                self.timeout,
            )
            yield
            
    async def shutdown(self, app):
        self.closing = True
        
        
    async def health(self, request):
        ready = not self.closing and self.llm.is_running
        return web.json_response(
            {"status": "ok" if ready else "unavailable"},
            status=200 if ready else 503,
        )

        
    async def models(self, request):
        return web.json_response({
            "object": "list",
            "data": [{
                "id": self.model,
                "object": "model",
                "owned_by": "tinyvllm",
            }],
        })

    
    async def generate(self, request):
        request_id = (
            "chatcmpl-"
            if request.path == "/v1/chat/completions"
            else "cmpl-"
        ) + uuid.uuid4().hex

        started = time.perf_counter()
        
        summary = {
            "outcome": "internal_error",
            "http_status": None,
            "engine_request_id": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "first_text_ms": None,
        }

        admitted = False
        
        try:
            if self.closing or not self.llm.is_running:
                summary.update(
                    outcome="unavailable",
                    http_status=503,
                )

                response = web.json_response(
                    _error("engine unavailable", "unavailable"),
                    status=503,
                    headers={"X-Request-ID": request_id},
                )

            elif self.active >= self.limit:
                summary.update(
                    outcome="overloaded",
                    http_status=429,
                )

                response = web.json_response(
                    _error("request capacity exhausted", "overloaded"),
                    status=429,
                    headers={"X-Request-ID": request_id},
                )

            else:
                self.active += 1
                admitted = True
                
                response = await self._generate(
                    request,
                    request_id,
                    summary,
                    started,
                )

            return response

        except asyncio.CancelledError:
            summary["outcome"] = "cancelled"
            raise
        
        finally:
            if admitted:
                self.active -= 1
                
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "request_id=%s %s",
                    request_id,
                    json.dumps({
                        **summary,
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000,
                            3,
                        ),
                    }),
                )
            
            
    async def _generate(self, request, request_id, summary, started):
        streaming = False
        submitted = False
        terminal_started = False
        
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        })

        try:
            async with asyncio.timeout(self.timeout):
                body = await request.json()
                chat = request.path == "/v1/chat/completions"
                sampling, streaming, include_usage = _parse(
                    body,
                    chat=chat,
                    model=self.model,
                )

                context = (
                    self.llm.chat_stream(
                        body.get("messages"),
                        sampling,
                        enable_thinking=body.get("enable_thinking", False)
                    )
                    if chat
                    else self.llm.stream(body.get("prompt"), sampling)
                )

                common = {
                    "id": request_id,
                    "object": "chat.completion" if chat else "text_completion",
                    "created": int(time.time()),
                    "model": self.model,
                }

                if streaming and include_usage:
                    common["usage"] = None

                async with context as stream:
                    submitted = True
                    if streaming:
                        await response.prepare(request)
                        summary["http_status"] = response.status
                        if chat:
                            await _send(response, self._chunk(common, chat, role=True))

                    async for event in stream:
                        if isinstance(event, TextDelta):
                            if summary["first_text_ms"] is None:
                                # HTTP application observation,
                                # not GPU TTFT or wire receipt.
                                summary["first_text_ms"] = round(
                                    (time.perf_counter() - started) * 1000,
                                    3,
                                )
                                summary["engine_request_id"] = event.request_id
                            
                            if streaming:
                                await _send(
                                    response,
                                    self._chunk(common, chat, text=event.text),
                                )
                            continue

                        summary.update(
                            engine_request_id=event.request_id,
                            finish_reason=event.finish_reason,
                            prompt_tokens=event.prompt_token_count,
                            completion_tokens=event.completion_token_count,
                        )
                        
                        usage = {
                            "prompt_tokens": event.prompt_token_count,
                            "completion_tokens": event.completion_token_count,
                            "total_tokens": event.total_token_count,
                        }
                        if streaming:
                            final = self._chunk(common, chat, reason=event.finish_reason)
                            terminal_started = True
                            await _send(response, final)
                            if include_usage:
                                await _send(response, {
                                    **final,
                                    "choices": [],
                                    "usage": usage,
                                })
                            await _send(response, "[DONE]")
                            await response.write_eof()

                            summary["outcome"] = "completed"
                            
                            return response
                        
                        choice = {"index": 0, "finish_reason": event.finish_reason}
                        if chat:
                            choice["message"] = {
                                "role": "assistant",
                                "content": event.text
                            }
                        else:
                            choice["text"] = event.text 
                        response = web.json_response(
                            {
                                **common,
                                "choices": [choice],
                                "usage": usage,
                            },
                            headers={"X-Request-ID": request_id},
                        )
                        await response.prepare(request)
                        summary["http_status"] = response.status
                        
                        await response.write_eof()
                        summary["outcome"] = "completed"
                        
                        return response
            
        except web.HTTPRequestEntityTooLarge:
            status = 413
            failure = _error(
                "request body exceeds 1 MiB",
                "invalid_request",
            )
        except OverloadedError as error:
            status, failure = 429, _error(str(error), "overloaded")
        except TimeoutError:
            status, failure = 504, _error(
                "request deadline exceeded",
                "timeout",
            )
        except SlowConsumerError:
            status, failure = 503, _error(
                "output consumer is too slow",
                "slow_consumer",
            )
        except ConnectionError:
            summary["outcome"] = "disconnected"
            _abort_response(request, response)
            return response
        except RuntimeError:
            logger.exception(
                "request_id=%s generation failed",
                request_id,
            )
            status, failure = 503, _error(
                "generation unavailable",
                "unavailable",
            )

        except Exception as error:
            if not submitted and isinstance(error, (TypeError, ValueError)):
                status, failure = 400, _error(str(error), "invalid_request")
            else:
                logger.exception(
                    "request_id=%s unexpected generation error",
                    request_id,
                )
                status, failure = 500, _error("internal generation error", "server_error")

        summary["outcome"] = failure["error"]["type"]
                
        if terminal_started:
            _abort_response(request, response)
            return response
        
        # Never send a second terminal; bound best-effort error delivery too.
        try:
            async with asyncio.timeout(1.0):
                if not response.prepared:
                    response = web.json_response(
                        failure, 
                        status=status,
                        headers={"X-Request-ID": request_id},
                    )

                    await response.prepare(request)
                    summary["http_status"] = response.status
                    
                    await response.write_eof()
                elif streaming:
                    await _send(response, failure)
                    await _send(response, "[DONE]")
                    await response.write_eof()
                else:
                    _abort_response(request, response)
        except (TimeoutError, ConnectionError):
            _abort_response(request,response)
            
        return response
    
    
    @staticmethod
    def _chunk(common, chat, *, text="", reason=None, role=False):
        choice = {
            "index": 0,
            "finish_reason": reason,
        }

        if chat:
            choice["delta"] = (
                {"role": "assistant"}
                if role
                else ({"content": text} if text else {})
            )
        else:
            choice["text"] = text
            
        return {
            **common,
            "object": "chat.completion.chunk" if chat else "text_completion",
            "choices": [choice],
        }
        
        
def create_app(
    config: EngineConfig,
    *,
    served_model: str = "Qwen3-0.6B",
    max_requests: int = 64,
    max_reserved_tokens: int = 65536,
    max_buffered_tokens: int = 128,
    request_timeout: float = 120.0,
) -> web.Application:
    if not isinstance(served_model, str) or not served_model:
        raise ValueError("served_model must be a nonempty string")

    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("request_timeout must be finite and positive")

    server = _Server(
        config,
        served_model,
        max_requests,
        max_reserved_tokens,
        max_buffered_tokens,
        request_timeout,
    )
    
    app = web.Application(
        client_max_size=1024**2,
        handler_args={"handler_cancellation": True},
    )
    app.cleanup_ctx.append(server.lifespan)
    app.on_shutdown.append(server.shutdown)
    
    app.add_routes([
        web.get("/health", server.health),
        web.get("/v1/models", server.models),
        web.post("/v1/completions", server.generate),
        web.post("/v1/chat/completions", server.generate),
    ])
    return app


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve the tiny-vLLM text API",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", default="Qwen3-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--attention-backend",
        choices=("reference", "cuda", "flashinfer"),
        default="cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA Graph capture to save memory and startup time",
    )
    parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable reuse of computed prefix KV blocks",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=1024,
        help="Total prefill and decode query-token limit per engine step",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="Per-request prefill query-token limit per engine step",
    )
    parser.add_argument("--kv-cache-mib", type=int, default=512)
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument("--max-reserved-tokens", type=int, default=65536)
    parser.add_argument("--max-buffered-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=120.0)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = EngineConfig(
        model=args.model,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=not args.disable_prefix_cache,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        prefill_reserve_tokens=0,
        prefill_reserve_requests=0,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024**2,
    )
    app = create_app(
        config,
        served_model=args.served_model_name,
        max_requests=args.max_requests,
        max_reserved_tokens=args.max_reserved_tokens,
        max_buffered_tokens=args.max_buffered_tokens,
        request_timeout=args.request_timeout,
    )

    web.run_app(
        app,
        host=args.host,
        port=args.port,
        shutdown_timeout=5.0,
    )