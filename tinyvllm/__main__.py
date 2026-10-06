from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path

from tinyvllm.config import EngineConfig
from tinyvllm.serving.async_llm import AsyncLLM, TextDelta
from tinyvllm.serving.llm import LLM, TextOutput
from tinyvllm.serving.terminal import chat_loop
from tinyvllm.request import SamplingParams


async def _print_stream(config, args, messages, sampling) -> None:
    async with AsyncLLM(config) as llm:
        context = (
            llm.chat_stream(messages, sampling, enable_thinking=args.thinking)
            if args.messages is not None
            else llm.stream(args.prompt[0], sampling)
        )
        async with context as stream:
            async for event in stream:
                if isinstance(event, TextDelta):
                    print(event.text, end="", flush=True)
                    
        print()
        
        
def _interactive_chat(config, args, sampling) -> None:
    llm = AsyncLLM(config)
    
    async def turn(messages, on_text):
        async with llm.chat_stream(
            messages,
            sampling,
            enable_thinking=args.thinking,
        ) as stream:
            async for event in stream:
                if isinstance(event, TextDelta):
                    on_text(event.text)
                elif isinstance(event, TextOutput):
                    return event.text
                
    with asyncio.Runner() as runner:
        runner.run(llm.__aenter__())
        try:
            chat_loop(
                lambda messages, on_text: runner.run(turn(messages, on_text)),
                system=args.system,
                title=f"tiny-vLLM | local | {args.model.name}",
            )
        finally:
            runner.run(llm.__aexit__(None, None, None))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run local tiny-vLLM text generation")
    parser.add_argument("--model", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt", action="append", help="raw prompt; repeat for a batch")
    inputs.add_argument("--messages", type=Path, help="UTF-8 JSON chat message array")
    inputs.add_argument("--chat", action="store_true", help="interactive local chat")
    parser.add_argument("--system", default="", help="system prompt for --chat")
    parser.add_argument("--stream", action="store_true", help="print incremental text")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=("reference", "cuda", "flashinfer"),
        default="cuda",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--kv-cache-mib", type=int, default=512)
    args = parser.parse_args(argv)

    if args.thinking and args.messages is None and not args.chat:
        parser.error("--thinking requires --messages or chat")
    if args.system and not args.chat:
        parser.error("--system requires --chat")

    if args.stream and args.prompt is not None and len(args.prompt) != 1:
        parser.error("--stream requires a single --prompt or --messages")
        
    messages = None
    if args.messages is not None:
        messages = json.loads(args.messages.read_text(encoding="utf-8"))

    sampling = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        ignore_eos=args.ignore_eos,
    )
    config = EngineConfig(
        model=args.model,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        prefill_reserve_tokens=0,
        prefill_reserve_requests=0,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024**2,
    )

    if args.chat:
        _interactive_chat(config, args, sampling)
        return
    if args.stream:
        asyncio.run(_print_stream(config, args, messages, sampling))
        return

    with LLM(config) as llm:
        if args.messages is not None:
            outputs = [llm.chat(messages, sampling, enable_thinking=args.thinking)]
        else:
            outputs = llm.generate(args.prompt, sampling)

    records = [
        {
            **asdict(output),
            "completion_token_count": output.completion_token_count,
            "total_token_count": output.total_token_count,
        }
        for output in outputs
    ]
    print(json.dumps({"outputs": records}, ensure_ascii=False))


if __name__ == "__main__":
    main()