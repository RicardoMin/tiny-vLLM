from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from tinyvllm.config import EngineConfig
from tinyvllm.engine import Engine, GenerateRequest, RequestOutput
from tinyvllm.request import CompletionEvent, RequestStatus, SamplingParams
from tinyvllm.serving.tokenizer import Tokenizer

@dataclass(frozen=True, slots=True)
class TextOutput:
    request_id: int
    text: str
    output_token_ids: tuple[int, ...]
    prompt_token_count: int
    finish_reason: Literal["stop", "length", "cancelled"]


    @property
    def completion_token_count(self) -> int:
        return len(self.output_token_ids)


    @property
    def total_token_count(self) -> int:
        return self.prompt_token_count + self.completion_token_count

        
    @classmethod
    def from_result(
        cls,
        result: RequestOutput | CompletionEvent,
        text: str,
        sampling: SamplingParams,
        eos_token_ids: Collection[int],
    ) -> TextOutput:
        tokens = result.output_token_ids
        if result.status is RequestStatus.ABORTED:
            reason = "cancelled"
        elif tokens and not sampling.ignore_eos and tokens[-1] in eos_token_ids:
            reason = "stop"
        else:
            reason = "length"
            
        return cls(
            request_id=result.request_id,
            text=text,
            output_token_ids=tokens,
            prompt_token_count=result.prompt_token_count,
            finish_reason=reason,
        )


class LLM:
    """Synchronous text facade. One caller owns this instance at a time."""

    def __init__(self, config: EngineConfig) -> None:
        self.tokenizer = Tokenizer(config.model)
        self._engine: Engine | None = Engine.from_config(config)


    def _get_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("LLM is closed")
        return self._engine


    def generate(
        self,
        prompts: str | Sequence[str],
        sampling: SamplingParams | None = None,
    ) -> list[TextOutput]:
        self._get_engine()
        if isinstance(prompts, str):
            prompts = (prompts,)
        if not isinstance(prompts, Sequence) or any(
            not isinstance(prompt, str) for prompt in prompts
        ):
            raise TypeError("prompts must be a string or a sequence of strings")

        # Complete tokenization before publishing any engine requests.
        tokens = tuple(self.tokenizer.encode(prompt) for prompt in prompts)
        return self._generate(tokens, sampling)


    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        sampling: SamplingParams | None = None,
        *,
        enable_thinking: bool = False,
    ) -> TextOutput:
        self._get_engine()
        tokens = self.tokenizer.encode_chat(
            messages,
            enable_thinking=enable_thinking,
        )
        return self._generate((tokens,), sampling)[0]


    def _generate(
        self,
        token_batches: Sequence[tuple[int, ...]],
        sampling: SamplingParams | None,
    ) -> list[TextOutput]:
        engine = self._get_engine()
        sampling = SamplingParams() if sampling is None else sampling
        if not isinstance(sampling, SamplingParams):
            raise TypeError("sampling must be SamplingParams")

        requests = tuple(
            GenerateRequest(tokens, sampling)
            for tokens in token_batches
        )
        outputs = engine.generate(requests)

        return [
            TextOutput.from_result(
                output,
                self.tokenizer.decode(output.output_token_ids),
                sampling,
                engine.scheduler.eos_token_ids,
            )
            for output in outputs
        ]


    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None


    def __enter__(self) -> LLM:
        self._get_engine()
        return self


    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"LLM cleanup failed: {cleanup_error!r}")