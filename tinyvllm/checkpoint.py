from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CheckpointError(RuntimeError):
    pass


def _shard_files(
    model_dir: str | Path,
) -> tuple[Path, ...]:
    root = Path(model_dir).expanduser().resolve()

    index = root / "model.safetensors.index.json"

    if index.is_file():
        data = json.loads(
            index.read_text(encoding="utf-8")
        )
        names = set(data["weight_map"].values())
        files = tuple(
            sorted(root / name for name in names)
        )
    else:
        files = tuple(
            sorted(root.glob("*.safetensors"))
        )

    if not files or any(
        not path.is_file() for path in files
    ):
        raise CheckpointError(
            f"no usable safetensors shard in {root}"
        )

    return files


def load_weights(
    model: Any,
    model_dir: str | Path,
) -> None:
    from safetensors import safe_open

    required = set(model.required_weight_names)
    loaded: set[str] = set()
    seen: set[str] = set()

    for shard_path in _shard_files(model_dir):
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

                loaded.add(slot)

    missing = required - loaded

    if missing:
        names = ", ".join(
            sorted(missing)[:8]
        )
        raise CheckpointError(
            f"missing model weights: {names}"
        )