from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from tinyvllm.checkpoint import (
    CheckpointError,
    load_weights,
)


class Sink:
    required_weight_names = frozenset(
        {"first", "second"}
    )

    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}

    def load_weight(
        self,
        source_name: str,
        tensor: torch.Tensor,
    ) -> str:
        self.values[source_name] = tensor
        return source_name


def test_streams_indexed_safetensors(
    tmp_path,
) -> None:
    first_shard = "model-00001-of-00002.safetensors"
    second_shard = "model-00002-of-00002.safetensors"

    save_file(
        {"first": torch.ones(2, 2)},
        str(tmp_path / first_shard),
    )
    save_file(
        {"second": torch.full((2, 2), 2.0)},
        str(tmp_path / second_shard),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": first_shard,
                    "second": second_shard,
                }
            }
        ),
        encoding="utf-8",
    )

    model = Sink()
    load_weights(model, tmp_path)

    assert torch.equal(
        model.values["first"],
        torch.ones(2, 2),
    )
    assert torch.equal(
        model.values["second"],
        torch.full((2, 2), 2.0),
    )


def test_reports_missing_weight(
    tmp_path,
) -> None:
    save_file(
        {"first": torch.ones(2, 2)},
        str(tmp_path / "model.safetensors"),
    )

    with pytest.raises(
        CheckpointError,
        match="missing model weights",
    ):
        load_weights(Sink(), tmp_path)


def test_rejects_unsafe_index_path(
    tmp_path,
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": "../outside.safetensors"
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        CheckpointError,
        match="invalid checkpoint index",
    ):
        load_weights(Sink(), tmp_path)


def test_rejects_malformed_index_schema(
    tmp_path,
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": None}),
        encoding="utf-8",
    )

    with pytest.raises(
        CheckpointError,
        match="invalid checkpoint index",
    ):
        load_weights(Sink(), tmp_path)
