from __future__ import annotations

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


def test_streams_safetensors(
    tmp_path,
) -> None:
    save_file(
        {
            "first": torch.ones(2, 2),
            "second": torch.full((2, 2), 2.0),
        },
        str(tmp_path / "model.safetensors"),
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