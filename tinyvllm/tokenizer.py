from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


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