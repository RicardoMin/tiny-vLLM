from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from tinyvllm.config import EngineConfig
from tinyvllm.llm import LLM
from tinyvllm.request import SamplingParams


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run local tiny-vLLM text generation")
    parser.add_argument("--model", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt", action="append", help="raw prompt; repeat for a batch")
    inputs.add_argument("--messages", type=Path, help="UTF-8 JSON chat message array")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=("reference", "cuda", "flashinfer"),
        default="reference",
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

    if args.thinking and args.messages is None:
        parser.error("--thinking requires --messages")

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