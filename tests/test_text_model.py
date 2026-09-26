from __future__ import annotations

import gc
import weakref

import pytest

from tinyvllm import GenerateRequest, LLM, SamplingParams
from tinyvllm.tokenizer import Tokenizer


@pytest.mark.model
def test_real_tokenizer_preserves_unicode_and_uses_exact_qwen_template(model_path):
    tokenizer = Tokenizer(model_path)
    texts = ('Hello, tiny-vLLM!\n', '你好，世界🙂\n第二行。', '  space\t space\n')
    for text in texts:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    messages = [{'role': 'user', 'content': '请用一句话介绍你自己。'}]
    for thinking in (False, True):
        rendered = tokenizer._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
        )
        assert tokenizer.encode_chat(messages, enable_thinking=thinking) == tokenizer.encode(rendered)
    assert tokenizer.decode((tokenizer._tokenizer.eos_token_id,)) == ''


@pytest.mark.model
@pytest.mark.cuda
@pytest.mark.parametrize('backend', [
    'reference', 'cuda',
    pytest.param('flashinfer', marks=pytest.mark.flashinfer),
])
def test_real_text_matches_token_api_and_close_releases_model_and_kv(backend, small_model_config):
    config = small_model_config
    messages = [{'role': 'user', 'content': '请用一句话介绍你自己。'}]
    sampling = SamplingParams(max_new_tokens=8)
    with LLM(config) as llm:
        engine = llm._engine
        runner = engine.executor.runner
        model_ref = weakref.ref(runner.model)
        parameter_ref = weakref.ref(next(runner.model.parameters()))
        cache_ref = weakref.ref(runner.kv_cache)
        workspace_ref = weakref.ref(runner.prefill.workspace) if runner.prefill else None

        tokens = llm.tokenizer.encode_chat(messages)
        direct = engine.generate((GenerateRequest(tokens, sampling),))[0]
        text = llm.chat(messages, sampling)
        assert text.output_token_ids == direct.output_token_ids
        assert text.text == llm.tokenizer.decode(direct.output_token_ids)
        assert text.prompt_token_count == len(tokens)
        assert text.completion_token_count == len(direct.output_token_ids)
        assert text.total_token_count == len(tokens) + len(direct.output_token_ids)
        assert not engine.has_unfinished()
        assert engine.scheduler.block_manager.free_block_count == runner.kv_cache.num_blocks
        assert text.text
        print(backend, repr(text.text), text.finish_reason, text.output_token_ids)

    gc.collect()
    assert runner.model is runner.kv_cache is runner.prefill is None
    assert model_ref() is parameter_ref() is cache_ref() is None
    assert workspace_ref is None or workspace_ref() is None
    with pytest.raises(RuntimeError, match='closed'):
        engine.generate(())
