from __future__ import annotations

import codecs
from collections.abc import Mapping, Sequence
from pathlib import Path


class StreamingDecoder:
    """One request's UTF-8 state; token byte strings are shared and immutable."""

    def __init__(self, token_bytes: Mapping[int, bytes]) -> None:
        self._token_bytes = token_bytes
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")


    def push(self, token_id: int) -> str:
        return self._utf8.decode(self._token_bytes.get(token_id, b""))


    def finish(self) -> str:
        return self._utf8.decode(b"", final=True)


class Tokenizer:
    """Local text-only tokenizer; scheduling and stopping stay in the engine."""

    def __init__(self, model: Path | str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )

        self._stream_bytes: dict[int, bytes] | None = None


    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("prompt must be a string")

        tokens = tuple(self._tokenizer.encode(text, add_special_tokens=False))
        if not tokens:
            raise ValueError("prompt must encode to at least one token")

        return tokens


    def encode_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        enable_thinking: bool = False,
    ) -> tuple[int, ...]:
        if not isinstance(messages, Sequence) or not messages:
            raise ValueError("messages must be a nonempty sequence") 
        if type(enable_thinking) is not bool:
            raise TypeError("enable_thinking must be bool")

        normalized = []
        for message in messages:
            if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                raise ValueError("each message needs exactly role and content")
            if message["role"] not in ("system", "user", "assistant"):
                raise ValueError("only system, user and assistant roles are supported")
            if not isinstance(message["content"], str):
                raise TypeError("message content must be text")
            normalized.append(dict(message))

        if normalized[-1]["role"] != "user":
            raise ValueError("the final message must be a user message")

        tokens = self._tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if not tokens:
            raise ValueError("chat template produced an empty prompt")

        return tuple(tokens)


    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


    def stream_decoder(self) -> StreamingDecoder:
        """Create request-local state for the currently supported ByteLevel format.
        
        Prepare the shared vocabulary once during service startup. ByteLevel's
        Unicode alphabet encodes bytes, not necessarily complete UTF-8 characters.
        """

        if self._stream_bytes is None:
            from tokenizers.decoders import ByteLevel

            backend = self._tokenizer.backend_tokenizer
            if not isinstance(backend.decoder, ByteLevel):
                raise ValueError("streaming requires a ByteLevel tokenizer decoder")

            byte_values = [*range(33, 127), *range(161, 173), *range(174, 256)]
            code_points = byte_values.copy()
            extra = 0

            for value in range(256):
                if value not in byte_values:
                    byte_values.append(value)
                    code_points.append(256 + extra)
                    extra += 1

            byte_decoder = {
                chr(code_point): value
                for value, code_point in zip(byte_values, code_points)
            }

            special_ids = {
                token_id
                for token_id, token in backend.get_added_tokens_decoder().items()
                if token.special
            }

            try:
                self._stream_bytes = {
                    token_id: bytes(byte_decoder[char] for char in token)
                    for token, token_id in backend.get_vocab().items()
                    if token_id not in special_ids
                }
            except KeyError as error:
                raise ValueError(
                    "streaming requires a ByteLevel-compatible vocabulary"
                ) from error

                
        return StreamingDecoder(self._stream_bytes)