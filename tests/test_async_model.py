from __future__ import annotations

import asyncio
import gc
import threading
import weakref

import pytest

from tinyvllm import Engine, GenerateRequest, SamplingParams, TokenEvent
from tinyvllm.async_engine import AsyncEngine
from tinyvllm.request import CompletionEvent
from tinyvllm.tokenizer import Tokenizer


@pytest.mark.cuda
@pytest.mark.model
@pytest.mark.parametrize('backend', [
    'reference', 'cuda',
    pytest.param('flashinfer', marks=pytest.mark.flashinfer),
])
def test_real_owner_stream_matches_token_api_and_disposes_resources(backend, monkeypatch, small_model_config):
    config = small_model_config
    tokenizer = Tokenizer(config.model)
    prompts = [tokenizer.encode_chat([{'role': 'user', 'content': text}]) for text in (
        '请用一句话介绍你自己。', '请介绍一下分页KV缓存。',
    )]
    sampling = SamplingParams(max_new_tokens=4)
    original = Engine.from_config
    expected = []
    thread_ids = []
    references = []

    def factory(cfg):
        thread_ids.append(threading.get_ident())
        engine = original(cfg)
        runner = engine.executor.runner
        references.extend((weakref.ref(runner.model), weakref.ref(runner.kv_cache)))
        if runner.prefill is not None:
            references.append(weakref.ref(runner.prefill.workspace))
        expected.extend(engine.generate(tuple(
            GenerateRequest(prompt, sampling, request_id=index)
            for index, prompt in enumerate(prompts)
        )))
        for name in ('submit', 'step', 'close'):
            method = getattr(engine, name)

            def checked(*args, _method=method, **kwargs):
                thread_ids.append(threading.get_ident())
                return _method(*args, **kwargs)

            setattr(engine, name, checked)
        return engine

    monkeypatch.setattr(Engine, 'from_config', factory)

    async def run():
        async with AsyncEngine(config) as bridge:
            streams = [bridge.submit(prompt, sampling) for prompt in prompts]

            async def collect(stream):
                async with stream:
                    return [event async for event in stream]

            outputs = await asyncio.gather(*(collect(stream) for stream in streams))
        assert not bridge._thread.is_alive()
        assert bridge._slots == {} and bridge._reserved_tokens == 0
        return outputs

    outputs = asyncio.run(asyncio.wait_for(run(), timeout=120))
    assert len(set(thread_ids)) == 1
    assert thread_ids[0] != threading.get_ident()
    for events, direct in zip(outputs, expected):
        assert isinstance(events[-1], CompletionEvent)
        assert all(isinstance(event, TokenEvent) for event in events[:-1])
        tokens = tuple(event.token_id for event in events[:-1])
        assert tokens == events[-1].output_token_ids == direct.output_token_ids
        assert events[-1].prompt_token_count == direct.prompt_token_count
        print(backend, repr(tokenizer.decode(tokens)), tokens)
    gc.collect()
    assert all(reference() is None for reference in references)
