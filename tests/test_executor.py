from __future__ import annotations

import torch

from tinyvllm.batch import (
    BatchBuilder,
    StepPlan,
    WorkItem,
)
from tinyvllm.config import EngineConfig
from tinyvllm.executor import EagerExecutor
from tinyvllm.models.qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
)
from tinyvllm.request import (
    RequestState,
    SamplingParams,
    WorkClass,
)
from tinyvllm.scheduler import AddCommand, Scheduler


def tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config.from_dict(
        {
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "rope_theta": 1_000_000,
            "tie_word_embeddings": True,
        }
    )

    torch.manual_seed(7)

    return Qwen3ForCausalLM(
        config,
        dtype=torch.float32,
    )


def test_executor_closes_one_request_scheduler_loop() -> None:
    model = tiny_model()
    executor = EagerExecutor(
        model,
        max_model_len=32,
    )

    scheduler = Scheduler(
        max_active_requests=1,
        max_step_requests=1,
        max_scheduled_tokens=8,
        prefill_chunk_size=8,
        max_model_len=32,
        kv_page_size=4,
        kv_num_pages=8,
    )

    builder = BatchBuilder(max_model_len=32)

    scheduler.apply_commands(
        (
            AddCommand(
                seq=0,
                request_id=11,
                prompt_token_ids=(1, 2, 3),
                sampling=SamplingParams(
                    max_new_tokens=3,
                ),
                arrival_ns=0,
            ),
        ),
        now_ns=0,
    )

    now_ns = 0

    while not scheduler.completed_request_ids:
        txn = scheduler.begin_step(now_ns)
        assert txn is not None

        requests = tuple(
            scheduler.get_request(item.request_id)
            for item in txn.plan.items
        )

        batch = builder.prepare(
            txn.plan,
            requests,
            kv=txn.kv,
        )

        scheduler.mark_dispatched(
            txn,
            now_ns,
        )

        result = executor.execute(batch)

        scheduler.commit_step(
            txn,
            result,
            now_ns,
        )

        now_ns += 1

        if not scheduler.completed_request_ids:
            scheduler.apply_commands(
                (),
                now_ns=now_ns,
            )

    event = scheduler.consume_finished()[0]

    assert event.request_id == 11
    assert len(event.output_token_ids) == 3

    # The last sampled token has not entered the KV cache yet.
    assert executor._caches[11].length == 5

    executor.clear()
    assert executor._caches == {}


def test_executor_skips_logits_for_partial_prefill() -> None:
    model = tiny_model()
    executor = EagerExecutor(
        model,
        max_model_len=32,
    )

    request = RequestState(
        request_id=1,
        submit_seq=0,
        prompt_token_ids=[1, 2, 3, 4],
        sampling=SamplingParams(
            max_new_tokens=1,
        ),
        arrival_ns=0,
    )

    plan = StepPlan(
        plan_id=0,
        items=(
            WorkItem(
                request_id=1,
                request_slot=0,
                work_class=WorkClass.PREFILL,
                token_start=0,
                query_len=2,
                context_len_before=0,
                should_sample=False,
            ),
        ),
    )

    batch = BatchBuilder(
        max_model_len=32,
    ).prepare(
        plan,
        (request,),
    )

    result = executor.execute(batch)

    assert result.sample_request_ids == ()
    assert result.sampled_token_ids == ()
    assert executor._caches[1].length == 2


def test_executor_accepts_single_request_batch() -> None:
    executor = EagerExecutor(
        tiny_model(),
        max_model_len=32,
    )

    request = RequestState(
        request_id=1,
        submit_seq=0,
        prompt_token_ids=[1],
        sampling=SamplingParams(max_new_tokens=1),
        arrival_ns=0,
    )

    plan = StepPlan(
        plan_id=0,
        items=(
            WorkItem(
                1,
                0,
                WorkClass.PREFILL,
                0,
                1,
                0,
                True,
            ),
        ),
    )

    result = executor.execute(
        BatchBuilder(max_model_len=32).prepare(plan, (request,))
    )

    assert result.sample_request_ids == (1,)


def test_executor_accepts_top_k_sampling() -> None:
    model = tiny_model()
    executor = EagerExecutor(
        model,
        max_model_len=32,
    )

    request = RequestState(
        request_id=1,
        submit_seq=0,
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(
            max_new_tokens=1,
            temperature=1.0,
            top_k=1,
            top_p=0.5,
            seed=7,
        ),
        arrival_ns=0,
    )

    plan = StepPlan(
        plan_id=0,
        items=(
            WorkItem(
                request_id=1,
                request_slot=0,
                work_class=WorkClass.PREFILL,
                token_start=0,
                query_len=3,
                context_len_before=0,
                should_sample=True,
            ),
        ),
    )

    batch = BatchBuilder(
        max_model_len=32,
    ).prepare(
        plan,
        (request,),
    )

    result = executor.execute(batch)

    assert result.sample_request_ids == (1,)
    assert len(result.sampled_token_ids) == 1
    assert 0 <= result.sampled_token_ids[0] < model.config.vocab_size


def test_executor_handles_single_and_multi_request_batches() -> None:
    executor = EagerExecutor(tiny_model(), max_model_len=32)
    requests = tuple(
        RequestState(
            request_id=request_id,
            submit_seq=request_id,
            prompt_token_ids=[request_id],
            sampling=SamplingParams(max_new_tokens=1),
            arrival_ns=0,
        )
        for request_id in (1, 2, 3)
    )

    single_plan = StepPlan(
        plan_id=1,
        items=(WorkItem(1, 0, WorkClass.PREFILL, 0, 1, 0, True),),
    )
    single = executor.execute(
        BatchBuilder(max_model_len=32).prepare(single_plan, (requests[0],))
    )
    assert single.sample_request_ids == (1,)

    multi_plan = StepPlan(
        plan_id=2,
        items=(
            WorkItem(2, 1, WorkClass.PREFILL, 0, 1, 0, True),
            WorkItem(3, 2, WorkClass.PREFILL, 0, 1, 0, True),
        ),
    )
    multi = executor.execute(
        BatchBuilder(max_model_len=32).prepare(
            multi_plan,
            (requests[1], requests[2]),
        )
    )
    assert multi.sample_request_ids == (2, 3)
