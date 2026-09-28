from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers

from tinyvllm.tokenizer import Tokenizer


@pytest.fixture
def byte_tokenizer():
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    backend = BackendTokenizer(models.BPE({char: i for i, char in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<stop>"])
    backend.add_tokens(["<think>"])

    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer._tokenizer = SimpleNamespace(backend_tokenizer=backend)
    tokenizer._stream_bytes = None
    return tokenizer, backend


def test_split_utf8_special_tokens_and_literal_replacement_character(byte_tokenizer):
    tokenizer, backend = byte_tokenizer
    ids = backend.encode("中�🙂").ids
    # The minimal vocabulary emits individual bytes. Special tokens disappear
    # even in the middle of an unfinished UTF-8 code point.
    ids.insert(1, backend.token_to_id("<stop>"))
    decoder = tokenizer.stream_decoder()
    deltas = [decoder.push(token_id) for token_id in ids]

    assert deltas[:4] == ["", "", "", "中"]
    assert "�" in deltas  # A literal U+FFFD must not be held until finish().
    assert "".join(deltas) + decoder.finish() == backend.decode(ids) == "中�🙂"


def test_partial_bytes_flush_and_non_special_added_token_boundary(byte_tokenizer):
    tokenizer, backend = byte_tokenizer
    partial = backend.encode("中").ids[:2]
    decoder = tokenizer.stream_decoder()
    assert "".join(decoder.push(token_id) for token_id in partial) == ""
    assert decoder.finish() == backend.decode(partial) == "�"

    ids = partial + [backend.token_to_id("<think>")]
    decoder = tokenizer.stream_decoder()
    assert "".join(decoder.push(token_id) for token_id in ids) + decoder.finish() == (
        backend.decode(ids)
    ) == "�<think>"


def test_streams_share_vocabulary_but_keep_independent_utf8_state(byte_tokenizer):
    tokenizer, backend = byte_tokenizer
    left = tokenizer.stream_decoder()
    right = tokenizer.stream_decoder()
    first, *rest = backend.encode("中").ids

    assert left.push(first) == ""
    assert "".join(right.push(i) for i in backend.encode("A").ids) == "A"
    assert right.finish() == ""
    assert "".join(left.push(i) for i in rest) + left.finish() == "中"


def test_unknown_padded_vocabulary_ids_match_backend_decode(byte_tokenizer):
    tokenizer, backend = byte_tokenizer
    ids = backend.encode("text").ids + [10_000]
    decoder = tokenizer.stream_decoder()

    assert "".join(decoder.push(i) for i in ids) + decoder.finish() == backend.decode(ids)


@pytest.mark.parametrize("unsupported", ("decoder", "vocabulary"))
def test_unsupported_tokenizer_is_rejected_before_streaming(byte_tokenizer, unsupported):
    tokenizer, backend = byte_tokenizer
    if unsupported == "decoder":
        backend.decoder = decoders.WordPiece()
    else:
        backend.add_tokens(["中文"])

    with pytest.raises(ValueError, match="ByteLevel"):
        tokenizer.stream_decoder()


@pytest.mark.model
def test_qwen3_streaming_matches_full_decode_for_text_and_generated_ids(model_path):
    import random

    tokenizer = Tokenizer(model_path)
    samples = [
        tokenizer.encode(text)
        for text in (
            "中文、English、🙂、é、�，以及\n换行。",
            "<think>reasoning</think>\n你好<|im_end|>",
            "tabs\t  spaces\r\n and punctuation: 1 + 2 = 3",
        )
    ]
    # Model-generated IDs need not encode valid Unicode or ordinary input text.
    # Include the checkpoint's special IDs, incomplete byte tokens and padded IDs.
    random_source = random.Random(20260926)
    vocabulary = list(tokenizer._tokenizer.get_vocab().values())
    for _ in range(64):
        samples.append(random_source.choices(vocabulary, k=64) + [151935])

    for ids in samples:
        decoder = tokenizer.stream_decoder()
        actual = "".join(decoder.push(token_id) for token_id in ids) + decoder.finish()
        assert actual == tokenizer.decode(ids)
