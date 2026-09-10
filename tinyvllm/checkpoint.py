from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import torch


class WeightLoadTarget(Protocol):
    @property
    def required_weight_names(self) -> frozenset[str]: ...

    def load_weight(
        self,
        source_name: str,
        tensor: torch.Tensor,
    ) -> str: ...


class CheckpointError(RuntimeError):
    pass


def _shard_files(
    model_dir: str | Path,
) -> tuple[Path, ...]:
    root = Path(model_dir).expanduser().resolve()

    if not root.is_dir():
        raise CheckpointError(
            f"model directory does not exist: {root}"
        )

    index = root / "model.safetensors.index.json"

    if index.is_file():
        try:
            names = set(
                json.loads(
                    index.read_text(encoding="utf-8")
                )["weight_map"].values()
            )
            files = tuple(
                sorted((root / name).resolve() for name in names)
            )

            for path in files:
                path.relative_to(root)
        except (
            OSError,
            UnicodeError,
            KeyError,
            TypeError,
            AttributeError,
            ValueError,
        ) as exc:
            raise CheckpointError(
                f"invalid checkpoint index: {index}"
            ) from exc
    else:
        files = tuple(
            sorted(root.glob("*.safetensors"))
        )

    if not files:
        raise CheckpointError(
            f"no safetensors shard found in {root}"
        )

    missing = next(
        (path for path in files if not path.is_file()),
        None,
    )

    if missing is not None:
        raise CheckpointError(
            f"missing checkpoint shard: {missing}"
        )

    return files


def load_weights(
    model: WeightLoadTarget,
    model_dir: str | Path,
) -> None:
    from safetensors import safe_open

    required = set(model.required_weight_names)
    loaded: set[str] = set()
    seen: set[str] = set()

    for shard_path in _shard_files(model_dir):
        try:
            with safe_open(
                str(shard_path),
                framework="pt",
                device="cpu",
            ) as shard:
                for source_name in shard.keys():
                    if source_name in seen:
                        raise CheckpointError(
                            f"duplicate tensor: {source_name}"
                        )

                    seen.add(source_name)

                    try:
                        slot = model.load_weight(
                            source_name,
                            shard.get_tensor(source_name),
                        )
                    except Exception as exc:
                        raise CheckpointError(
                            f"failed to load {source_name}"
                        ) from exc

                    if not isinstance(slot, str):
                        raise CheckpointError(
                            "load_weight() must return a string"
                        )

                    if slot not in required:
                        raise CheckpointError(
                            f"unexpected model weight: {slot}"
                        )

                    loaded.add(slot)
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(
                f"failed to read checkpoint shard: {shard_path}"
            ) from exc

    missing = required - loaded

    if missing:
        names = ", ".join(sorted(missing)[:8])
        raise CheckpointError(
            f"missing model weights: {names}"
        )
