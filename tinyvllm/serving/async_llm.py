from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from dataclasses import dataclass
from functools import partial

from tinyvllm.async_engine import AsyncEngine, OverloadedError
from tinyvllm.config import EngineConfig
from tinyvllm.engine import TokenEvent
from tinyvllm.serving.llm import TextOutput
from tinyvllm.request import RequestStatus, SamplingParams
from tinyvllm.serving.tokenizer import Tokenizer


@dataclass(frozen=True, slots=True)
class TextDelta:
    request_id: int
    text: str
    
    
class _TokenizerWorker:
    """Count native CPU work until completion, even if its waiter is cancelled."""

    def __init__(self, limit: int) -> None:
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tinyvllm-text")
        self.limit = limit
        self.pending = 0
        
        
    async def call(self, function, *args, **kwargs):
        if self.pending >= self.limit:
            raise OverloadedError("tokenizer capacity exhausted")

        future = asyncio.get_running_loop().run_in_executor(
            self.pool,
            partial(function, *args, **kwargs),
        )
        self.pending += 1
        future.add_done_callback(self._completed)
        return await asyncio.shield(future)
    
    
    def _completed(self, future) -> None:
        self.pending -= 1
        if not future.cancelled():
            future.exception()
            
            
class AsyncLLM:
    """One async context and event loop; concurrent request streams are supported."""

    def __init__(
        self,
        config: EngineConfig,
        *,
        max_requests: int = 64,
        max_reserved_tokens: int = 65536,
        max_buffered_tokens: int = 128,
    ) -> None:
        self._config = config
        self._limit = max_requests
        self._bridge = AsyncEngine(
            config,
            max_requests=max_requests,
            max_reserved_tokens=max_reserved_tokens,
            max_buffered_tokens=max_buffered_tokens,
        )
        self._lifetime: AsyncExitStack | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        
        
    @property
    def is_running(self) -> bool:
        return self._bridge.is_running
    
    
    def _load_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(self._config.model)
        tokenizer.stream_decoder()
        return tokenizer
    
    
    async def __aenter__(self) -> AsyncLLM:
        if self._lifetime is not None:
            raise RuntimeError("AsyncLLM supports one lifetime context")

        self._loop = asyncio.get_running_loop()
        self._lifetime = AsyncExitStack()
        self._cpu = _TokenizerWorker(self._limit)
        self._lifetime.push_async_callback(
            asyncio.to_thread,
            self._cpu.pool.shutdown,
            wait=True,
            cancel_futures=True,
        )

        try:
            self._tokenizer = await self._cpu.call(self._load_tokenizer)
            await self._lifetime.enter_async_context(self._bridge)
        except BaseException:
            await self._lifetime.aclose()
            raise
        return self
    
    
    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self._lifetime.__aexit__(exc_type, exc, traceback)
        
        
    def stream(self, prompt: str, sampling: SamplingParams | None = None):
        return self._stream(prompt, sampling, chat=False, enable_thinking=False)

    
    def chat_stream(
        self,
        messages: Sequence[Mapping[str, str]],
        sampling: SamplingParams | None = None,
        *,
        enable_thinking: bool = False,
    ):
        return self._stream(messages, sampling, chat=True, enable_thinking=enable_thinking)

        
    async def generate(
        self,
        prompt: str,
        sampling: SamplingParams | None = None,
    ) -> TextOutput:
        return await self._collect(self.stream(prompt, sampling))

        
    async def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        sampling: SamplingParams | None = None,
        *,
        enable_thinking: bool = False,
    ) -> TextOutput:
        return await self._collect(
            self.chat_stream(messages, sampling, enable_thinking=enable_thinking)
        )

        
    @staticmethod
    async def _collect(context) -> TextOutput:
        async with context as stream:
            async for event in stream:
                if isinstance(event, TextOutput):
                    return event
                
                
    @asynccontextmanager
    async def _stream(self, source, sampling, *, chat, enable_thinking):
        if asyncio.get_running_loop() is not self._loop or not self.is_running:
            raise RuntimeError(
                "use AsyncLLM inside its active context and event loop"
            )

        sampling = SamplingParams() if sampling is None else sampling
        if not isinstance(sampling, SamplingParams):
            raise TypeError("sampling must be SamplingParams")

        if chat:
            prompt = await self._cpu.call(
                self._tokenizer.encode_chat,
                source,
                enable_thinking=enable_thinking,
            )
        else:
            prompt = await self._cpu.call(self._tokenizer.encode, source)

        decoder = self._tokenizer.stream_decoder()
        async with self._bridge.submit(prompt, sampling) as stream:
            async with aclosing(self._decode(stream, decoder, sampling)) as output:
                yield output
                
                
    async def _decode(
        self,
        stream,
        decoder,
        sampling
    ) -> AsyncIterator[TextDelta | TextOutput]:
        pieces = []
        async for event in stream:
            if isinstance(event, TokenEvent):
                text = decoder.push(event.token_id)
                if text:
                    pieces.append(text)
                    yield TextDelta(event.request_id, text)
                continue
            
            if event.status is RequestStatus.ABORTED:
                raise RuntimeError("generation was cancelled")

            tail = decoder.finish()
            if tail:
                pieces.append(tail)
                yield TextDelta(event.request_id, tail)

            yield TextOutput.from_result(
                event,
                "".join(pieces),
                sampling,
                self._bridge.eos_token_ids,
            )
            return
        
        raise RuntimeError("engine stream ended without completion")