from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from copy import deepcopy

from tinyvllm.batch import (
    BatchBuilder,
    PreparedBatch,
    StepMode,
    StepPlan,
    WorkItem,
)
from tinyvllm.executor import EagerExecutor
from tinyvllm.models.qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
)
from tinyvllm.request import (
    RequestState,
    RequestStatus,
    SamplingParams,
    WorkClass,
)


MAX_MODEL_LEN = 32


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
            "max_position_embeddings": MAX_MODEL_LEN,
            "rope_theta": 1_000_000,
            "tie_word_embeddings": True,
        }
    )

    torch.manual_seed(23)

    return Qwen3ForCausalLM(
        config,
        dtype=torch.float32,
    )


def make_request(
    request_id: int,
    prompt: tuple[int, ...],
    *,
    sampling: SamplingParams | None = None,
    output: tuple[int, ...] = (),
) -> RequestState:
    return RequestState(
        request_id=request_id,
        submit_seq=request_id,
        prompt_token_ids=list(prompt),
        sampling=(
            SamplingParams(max_new_tokens=4)
            if sampling is None
            else sampling
        ),
        arrival_ns=0,
        output_token_ids=list(output),
        status=RequestStatus.WAITING,
    )


def prefill_item(
    request: RequestState,
    *,
    slot: int,
    token_start: int = 0,
    query_len: int | None = None,
    context_len_before: int = 0,
    should_sample: bool = True,
) -> WorkItem:
    if query_len is None:
        query_len = len(request.prompt_token_ids) - token_start

    return WorkItem(
        request_id=request.request_id,
        request_slot=slot,
        work_class=WorkClass.PREFILL,
        token_start=token_start,
        query_len=query_len,
        context_len_before=context_len_before,
        should_sample=should_sample,
    )


def decode_item(
    request: RequestState,
    *,
    slot: int,
    token_start: int,
    context_len_before: int,
) -> WorkItem:
    return WorkItem(
        request_id=request.request_id,
        request_slot=slot,
        work_class=WorkClass.DECODE,
        token_start=token_start,
        query_len=1,
        context_len_before=context_len_before,
        should_sample=True,
    )


def prepare(
    plan: StepPlan,
    requests: tuple[RequestState, ...],
) -> PreparedBatch:
    return BatchBuilder(
        max_model_len=MAX_MODEL_LEN,
    ).prepare(
        plan,
        requests,
    )


def test_batch_builder_records_request_ids_in_plan_order() -> None:
    first = make_request(11, (1, 2, 3))
    second = make_request(22, (4, 5))

    plan = StepPlan(
        plan_id=1,
        items=(
            prefill_item(first, slot=0),
            prefill_item(second, slot=1),
        ),
    )

    batch = prepare(plan, (first, second))

    assert tuple(batch.request_ids) == (11, 22)
    assert batch.q_offsets == (0, 3, 5)
    assert batch.sample_request_ids == (11, 22)


def test_batched_prefill_uses_independent_caches() -> None:
    first = make_request(11, (1, 2, 3))
    second = make_request(22, (4, 5))

    executor = EagerExecutor(
        tiny_model(),
        max_model_len=MAX_MODEL_LEN,
    )

    plan = StepPlan(
        plan_id=2,
        items=(
            prefill_item(first, slot=0),
            prefill_item(second, slot=1),
        ),
    )

    result = executor.execute(
        prepare(plan, (first, second))
    )

    assert result.sample_request_ids == (11, 22)
    assert len(result.sampled_token_ids) == 2

    assert executor._caches[11] is not executor._caches[22]
    assert executor._caches[11].length == 3
    assert executor._caches[22].length == 2


def test_batched_prefill_matches_independent_torch_executors() -> None:
    requests = (
        make_request(11, (1, 2, 3)),
        make_request(22, (4, 5)),
    )

    batched = EagerExecutor(
        tiny_model(),
        max_model_len=MAX_MODEL_LEN,
    )

    plan = StepPlan(
        plan_id=3,
        items=tuple(
            prefill_item(request, slot=index)
            for index, request in enumerate(requests)
        ),
    )

    batched_result = batched.execute(
        prepare(plan, requests)
    )

    batched_tokens = dict(
        zip(
            batched_result.sample_request_ids,
            batched_result.sampled_token_ids,
        )
    )

    expected_tokens: dict[int, int] = {}

    for request in requests:
        single = EagerExecutor(
            deepcopy(batched.model),
            max_model_len=MAX_MODEL_LEN,
        )

        single_plan = StepPlan(
            plan_id=3,
            items=(
                prefill_item(
                    request,
                    slot=0,
                ),
            ),
        )

        result = single.execute(
            prepare(single_plan, (request,))
        )

        expected_tokens[request.request_id] = (
            result.sampled_token_ids[0]
        )

    assert batched_tokens == expected_tokens


def test_partial_prefill_does_not_sample() -> None:
    request = make_request(
        11,
        (1, 2, 3, 4),
    )

    executor = EagerExecutor(
        tiny_model(),
        max_model_len=MAX_MODEL_LEN,
    )

    plan = StepPlan(
        plan_id=4,
        items=(
            prefill_item(
                request,
                slot=0,
                query_len=2,
                should_sample=False,
            ),
        ),
    )

    result = executor.execute(
        prepare(plan, (request,))
    )

    assert result.sample_request_ids == ()
    assert result.sampled_token_ids == ()
    assert executor._caches[11].length == 2


def test_decode_grows_each_request_cache_without_cross_contamination() -> None:
    requests = [
        make_request(11, (1, 2, 3)),
        make_request(22, (4, 5)),
    ]

    executor = EagerExecutor(
        tiny_model(),
        max_model_len=MAX_MODEL_LEN,
    )

    prefill_plan = StepPlan(
        plan_id=5,
        items=tuple(
            prefill_item(request, slot=index)
            for index, request in enumerate(requests)
        ),
    )

    prefill_result = executor.execute(
        prepare(prefill_plan, tuple(requests))
    )

    prefill_tokens = dict(
        zip(
            prefill_result.sample_request_ids,
            prefill_result.sampled_token_ids,
        )
    )

    for request in requests:
        request.output_token_ids.append(
            prefill_tokens[request.request_id]
        )

    decode_plan = StepPlan(
        plan_id=6,
        items=(
            decode_item(
                requests[0],
                slot=0,
                token_start=3,
                context_len_before=3,
            ),
            decode_item(
                requests[1],
                slot=1,
                token_start=2,
                context_len_before=2,
            ),
        ),
    )

    decode_result = executor.execute(
        prepare(decode_plan, tuple(requests))
    )

    assert decode_result.sample_request_ids == (11, 22)
    assert executor._caches[11].length == 4
    assert executor._caches[22].length == 3

    decoded_tokens = dict(
        zip(
            decode_result.sample_request_ids,
            decode_result.sampled_token_ids,
        )
    )

    for request in requests:
        oracle_request = make_request(
            request.request_id,
            tuple(request.prompt_token_ids),
        )

        oracle = EagerExecutor(
            tiny_model(),
            max_model_len=MAX_MODEL_LEN,
        )

        oracle_prefill = StepPlan(
            plan_id=5,
            items=(
                prefill_item(
                    oracle_request,
                    slot=0,
                ),
            ),
        )

        oracle_prefill_result = oracle.execute(
            prepare(
                oracle_prefill,
                (oracle_request,),
            )
        )

        oracle_request.output_token_ids.append(
            oracle_prefill_result.sampled_token_ids[0]
        )

        oracle_decode = StepPlan(
            plan_id=6,
            items=(
                decode_item(
                    oracle_request,
                    slot=0,
                    token_start=len(
                        oracle_request.prompt_token_ids
                    ),
                    context_len_before=len(
                        oracle_request.prompt_token_ids
                    ),
                ),
            ),
        )

        oracle_result = oracle.execute(
            prepare(
                oracle_decode,
                (oracle_request,),
            )
        )

        assert decoded_tokens[request.request_id] == (
            oracle_result.sampled_token_ids[0]
        )


def test_mixed_plan_skips_logits_for_partial_prefill() -> None:
    decode_request = make_request(
        11,
        (1, 2, 3),
        output=(7,),
    )

    partial_request = make_request(
        22,
        (8, 9, 10, 11),
    )

    executor = EagerExecutor(
        tiny_model(),
        max_model_len=MAX_MODEL_LEN,
    )

    warmup = StepPlan(
        plan_id=7,
        items=(
            prefill_item(
                decode_request,
                slot=0,
                should_sample=False,
            ),
        ),
    )

    warmup_result = executor.execute(
        prepare(warmup, (decode_request,))
    )

    assert warmup_result.sample_request_ids == ()

    plan = StepPlan(
        plan_id=8,
        items=(
            decode_item(
                decode_request,
                slot=0,
                token_start=3,
                context_len_before=3,
            ),
            prefill_item(
                partial_request,
                slot=1,
                query_len=2,
                should_sample=False,
            ),
        ),
    )

    batch = prepare(
        plan,
        (
            decode_request,
            partial_request,
        ),
    )

    assert batch.mode is StepMode.MIXED
    assert batch.sample_request_ids == (11,)

    result = executor.execute(batch)

    assert result.sample_request_ids == (11,)
    assert len(result.sampled_token_ids) == 1
    assert executor._caches[11].length == 4
    assert executor._caches[22].length == 2


def test_reordering_batch_preserves_request_sample_mapping() -> None:
    params = SamplingParams(
        max_new_tokens=2,
        temperature=1.0,
        top_k=4,
        seed=10,
    )

    forward_requests = (
        make_request(11, (1, 2, 3), sampling=params),
        make_request(22, (4, 5), sampling=params),
    )

    reverse_requests = (
        make_request(22, (4, 5), sampling=params),
        make_request(11, (1, 2, 3), sampling=params),
    )

    forward = EagerExecutor(
        tiny_model(),
        max_model_len=MAX_MODEL_LEN,
    )

    reverse = EagerExecutor(
        tiny_model(),
        max_model_len=MAX_MODEL_LEN,
    )

    forward_plan = StepPlan(
        plan_id=9,
        items=(
            prefill_item(
                forward_requests[0],
                slot=0,
            ),
            prefill_item(
                forward_requests[1],
                slot=1,
            ),
        ),
    )

    reverse_plan = StepPlan(
        plan_id=9,
        items=(
            prefill_item(
                reverse_requests[0],
                slot=0,
            ),
            prefill_item(
                reverse_requests[1],
                slot=1,
            ),
        ),
    )

    forward_result = forward.execute(
        prepare(
            forward_plan,
            forward_requests,
        )
    )

    reverse_result = reverse.execute(
        prepare(
            reverse_plan,
            reverse_requests,
        )
    )

    forward_tokens = dict(
        zip(
            forward_result.sample_request_ids,
            forward_result.sampled_token_ids,
        )
    )

    reverse_tokens = dict(
        zip(
            reverse_result.sample_request_ids,
            reverse_result.sampled_token_ids,
        )
    )

    assert forward_tokens == reverse_tokens


@pytest.mark.parametrize(
    "request_ids",
    [
        (),
        (11, 11),
    ],
)
def test_executor_requires_unique_request_id_per_item(
    request_ids: tuple[int, ...],
) -> None:
    first = make_request(11, (1, 2))
    second = make_request(22, (3, 4))

    plan = StepPlan(
        plan_id=10,
        items=(
            prefill_item(first, slot=0),
            prefill_item(second, slot=1),
        ),
    )

    batch = replace(
        prepare(
            plan,
            (first, second),
        ),
        request_ids=request_ids,
    )

    with pytest.raises(ValueError):
        EagerExecutor(
            tiny_model(),
            max_model_len=MAX_MODEL_LEN,
        ).execute(batch)


def test_release_discards_request_cache() -> None:
    request = make_request(
        11,
        (1, 2, 3),
    )

    executor = EagerExecutor(
        tiny_model(),
        max_model_len=MAX_MODEL_LEN,
    )

    plan = StepPlan(
        plan_id=11,
        items=(
            prefill_item(
                request,
                slot=0,
            ),
        ),
    )

    executor.execute(
        prepare(plan, (request,))
    )

    assert executor._caches[11].length == 3

    executor.release(11)

    assert 11 not in executor._caches

    restarted_plan = StepPlan(
        plan_id=12,
        items=(
            prefill_item(
                request,
                slot=0,
            ),
        ),
    )

    executor.execute(
        prepare(restarted_plan, (request,))
    )

    assert executor._caches[11].length == 3


def test_clear_discards_caches_and_sampler_rng() -> None:
    request = make_request(
        11,
        (1, 2, 3),
        sampling=SamplingParams(
            max_new_tokens=1,
            temperature=1.0,
            top_k=2,
            seed=23,
        ),
    )

    executor = EagerExecutor(
        tiny_model(),
        max_model_len=MAX_MODEL_LEN,
    )

    plan = StepPlan(
        plan_id=13,
        items=(
            prefill_item(
                request,
                slot=0,
            ),
        ),
    )

    executor.execute(
        prepare(plan, (request,))
    )

    assert executor._caches
    assert executor.sampler._generators

    executor.clear()

    assert executor._caches == {}
    assert executor.sampler._generators == {}
