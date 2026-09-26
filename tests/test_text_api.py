from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tinyvllm import EngineConfig, LLM, SamplingParams, TextOutput
from tinyvllm.engine import Engine
from engine_helpers import CpuFakeExecutor
from tinyvllm.tokenizer import Tokenizer
from engine_helpers import make_engine


class FakeTokenizer:
    def encode(self, prompt):
        if prompt == '':
            raise ValueError('empty prompt')
        return tuple(range(1, len(prompt) + 1))

    def encode_chat(self, messages, *, enable_thinking=False):
        return (1, 2, 3, 4 if enable_thinking else 5)

    def decode(self, tokens):
        return ''.join('x' for token in tokens if token != 7)


def make_llm(monkeypatch, tmp_path, *, eos=(), token=6):
    engine = make_engine(eos_token_ids=eos, executor=CpuFakeExecutor(token_id=token))
    tokenizer = FakeTokenizer()
    monkeypatch.setattr('tinyvllm.llm.Tokenizer', lambda model: tokenizer)
    monkeypatch.setattr(Engine, 'from_config', Mock(return_value=engine))
    return LLM(EngineConfig(model=tmp_path)), engine


def test_tokenizer_uses_only_local_assets_and_does_not_add_or_cleanup_tokens(monkeypatch):
    backend = Mock()
    backend.encode.side_effect = [[1, 2], [3], []]
    backend.decode.return_value = '中 文\n🙂'
    loader = Mock(return_value=backend)
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', loader)
    tokenizer = Tokenizer('/local/model')
    loader.assert_called_once_with('/local/model', local_files_only=True, trust_remote_code=False, use_fast=True)
    assert tokenizer.encode('hello') == (1, 2)
    assert tokenizer.encode(' ') == (3,)
    with pytest.raises(ValueError, match='at least one'):
        tokenizer.encode('')
    with pytest.raises(TypeError, match='string'):
        tokenizer.encode([1])
    assert all(call.kwargs == {'add_special_tokens': False} for call in backend.encode.call_args_list)
    assert tokenizer.decode((10, 11)) == '中 文\n🙂'
    backend.decode.assert_called_once_with([10, 11], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def test_chat_calls_template_once_without_reencoding_or_mutating_messages():
    tokenizer = object.__new__(Tokenizer)
    tokenizer._tokenizer = Mock()
    tokenizer._tokenizer.apply_chat_template.return_value = [10, 20, 30]
    messages = [{'role': 'system', 'content': '助手'}, {'role': 'user', 'content': '你好'}]
    assert tokenizer.encode_chat(messages) == (10, 20, 30)
    tokenizer._tokenizer.apply_chat_template.assert_called_once_with(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    tokenizer._tokenizer.encode.assert_not_called()
    assert messages[-1] == {'role': 'user', 'content': '你好'}


@pytest.mark.parametrize('messages', [
    [], 'hello', [{'role': 'tool', 'content': 'x'}],
    [{'role': 'user', 'content': ['image']}],
    [{'role': 'user', 'content': 'x', 'tools': []}],
    [{'role': 'assistant', 'content': 'prefix'}],
])
def test_invalid_chat_is_rejected_before_template_execution(messages):
    tokenizer = object.__new__(Tokenizer)
    tokenizer._tokenizer = Mock()
    with pytest.raises((TypeError, ValueError)):
        tokenizer.encode_chat(messages)
    tokenizer._tokenizer.apply_chat_template.assert_not_called()


def test_batch_text_uses_one_generate_call_and_keeps_submission_order(monkeypatch, tmp_path):
    llm, engine = make_llm(monkeypatch, tmp_path)
    original = engine.generate
    engine.generate = Mock(wraps=original)
    with llm:
        outputs = llm.generate(['long', 'a'], SamplingParams(max_new_tokens=3))
        assert [item.request_id for item in outputs] == [0, 1]
        assert [item.prompt_token_count for item in outputs] == [4, 1]
        assert [item.text for item in outputs] == ['xxx', 'xxx']
        assert all(item.output_token_ids == (6, 6, 6) for item in outputs)
        assert all(item.completion_token_count == 3 for item in outputs)
        assert [item.total_token_count for item in outputs] == [7, 4]
        assert all(item.finish_reason == 'length' for item in outputs)
        assert not engine.has_unfinished()
    engine.generate.assert_called_once()
    assert engine.scheduler.block_manager.free_block_count == 16


@pytest.mark.parametrize('bad_prompts', [['ok', ''], ['ok', 'x' * 17], ['ok', 3], {'a': 'b'}])
def test_bad_batch_is_atomic_and_does_not_consume_request_ids(bad_prompts, monkeypatch, tmp_path):
    llm, engine = make_llm(monkeypatch, tmp_path)
    with llm:
        with pytest.raises((ValueError, TypeError)):
            llm.generate(bad_prompts)
        assert not engine.has_unfinished()
        assert llm.generate('ok', SamplingParams(max_new_tokens=1))[0].request_id == 0


@pytest.mark.parametrize('ignore,limit,expected_reason,expected_count', [
    (False, 4, 'stop', 1), (False, 1, 'stop', 1),
    (True, 3, 'length', 3), (False, 0, 'length', 0),
])
def test_eos_length_and_usage_keep_raw_tokens(ignore, limit, expected_reason, expected_count, monkeypatch, tmp_path):
    llm, _ = make_llm(monkeypatch, tmp_path, eos=(7,), token=7)
    with llm:
        output = llm.generate('a', SamplingParams(max_new_tokens=limit, ignore_eos=ignore))[0]
        assert output.finish_reason == expected_reason
        assert output.output_token_ids == (7,) * expected_count
        assert output.completion_token_count == expected_count
        assert output.text == ''


def test_chat_passes_thinking_explicitly_and_closed_llm_rejects_work(monkeypatch, tmp_path):
    llm, engine = make_llm(monkeypatch, tmp_path)
    llm.tokenizer.encode_chat = Mock(return_value=(1, 2, 3))
    messages = [{'role': 'user', 'content': 'hi'}]
    output = llm.chat(messages, SamplingParams(max_new_tokens=1), enable_thinking=True)
    assert output.prompt_token_count == 3
    llm.tokenizer.encode_chat.assert_called_once_with(messages, enable_thinking=True)
    llm.close()
    llm.close()
    assert llm._engine is None
    with pytest.raises(RuntimeError, match='closed'):
        llm.generate([])
    with pytest.raises(RuntimeError, match='closed'):
        engine.submit((1,), SamplingParams())


def test_context_cleanup_does_not_replace_the_original_error(monkeypatch, tmp_path):
    llm, engine = make_llm(monkeypatch, tmp_path)
    original = engine.close

    def failed_close():
        original()
        raise RuntimeError('cleanup')

    monkeypatch.setattr(engine, 'close', failed_close)
    with pytest.raises(ValueError, match='original') as captured:
        with llm:
            raise ValueError('original')
    assert 'LLM cleanup failed' in captured.value.__notes__[0]


def test_context_can_close_an_already_failed_engine(monkeypatch, tmp_path):
    llm, engine = make_llm(monkeypatch, tmp_path)
    engine.executor = SimpleNamespace(
        capabilities=CpuFakeExecutor.capabilities,
        execute=Mock(side_effect=RuntimeError('injected layer failure')),
        release=Mock(), clear=Mock(),
    )
    with pytest.raises(RuntimeError, match='executor failed'):
        with llm:
            llm.generate('hello')
    assert llm._engine is None
    assert not engine.has_unfinished()
    engine.close()


def test_missing_tokenizer_fails_before_model_loading(monkeypatch, tmp_path):
    monkeypatch.setattr('tinyvllm.llm.Tokenizer', Mock(side_effect=OSError('missing assets')))
    load = Mock()
    monkeypatch.setattr(Engine, 'from_config', load)
    with pytest.raises(OSError, match='missing assets'):
        LLM(EngineConfig(model=tmp_path))
    load.assert_not_called()


def test_text_cli_passes_arguments_and_prints_structured_unicode(monkeypatch, tmp_path, capsys):
    from tinyvllm.__main__ import main

    output = TextOutput(0, '你好\n', (10, 11), 3, 'length')
    llm = Mock()
    context = Mock()
    context.__enter__ = Mock(return_value=llm)
    context.__exit__ = Mock(return_value=False)
    llm.generate.return_value = [output]
    factory = Mock(return_value=context)
    monkeypatch.setattr('tinyvllm.__main__.LLM', factory)
    main(['--model', str(tmp_path), '--prompt', '你好', '--prompt', 'world', '--max-new-tokens', '2'])
    record = json.loads(capsys.readouterr().out)['outputs'][0]
    assert record['text'] == '你好\n'
    assert record['output_token_ids'] == [10, 11]
    assert record['completion_token_count'] == 2 and record['total_token_count'] == 5
    assert llm.generate.call_args.args[0] == ['你好', 'world']
    assert llm.generate.call_args.args[1].max_new_tokens == 2
    assert factory.call_args.args[0].prefill_reserve_tokens == 0
    context.__exit__.assert_called_once()


def test_text_cli_rejects_thinking_with_raw_prompt(tmp_path):
    from tinyvllm.__main__ import main

    with pytest.raises(SystemExit) as error:
        main(['--model', str(tmp_path), '--prompt', 'hello', '--thinking'])
    assert error.value.code == 2
