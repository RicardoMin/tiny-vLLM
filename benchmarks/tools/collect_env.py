from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re, stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()

def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, "LC_ALL": "C"},
        text=True,
        errors="surrogateescape",
        capture_output=True,
        timeout=15,
        check=False,
    )

    if result.returncode:
        raise RuntimeError(result.stderr.strip())

    return result.stdout.strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env={**os.environ, "LC_ALL": "C"},
        capture_output=True,
        timeout=15,
        check=True,
    )

    return result.stdout


def git_identity(repo: Path) -> dict:
    top = _git_bytes(
        repo,
        "rev-parse",
        "--show-toplevel",
    ).removesuffix(b"\n")

    repo = Path(os.fsdecode(top))

    status = run_git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    index = _git_bytes(repo, "ls-files", "--stage", "-z")
    paths = _git_bytes(
        repo,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )

    files = []

    for name in sorted(set(paths.split(b"\0")) - {b""}):
        path = repo / os.fsdecode(name)

        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            entry = {"kind": "missing"}
        else:
            if stat.S_ISLNK(mode):
                entry = {
                    "kind": "symlink",
                    "target": os.fsencode(os.readlink(path)).hex(),
                }
            elif stat.S_ISREG(mode):
                entry = {
                    "kind": "file",
                    "executable": bool(mode & stat.S_IXUSR),
                    "sha256": sha256_file(path),
                }
            else:
                raise ValueError(
                    f"cannot fingerprint submodule or special file: {path}"
                )

        files.append({"path": name.hex(), **entry})

    return {
        "commit": run_git(repo, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": fingerprint(status) if status else None,
        "worktree_sha256": fingerprint({
            "index_sha256": hashlib.sha256(index).hexdigest(),
            "files": files,
        }),
    }

def packages() -> list[dict]:
    result = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            normalized = re.sub(r"[-_.]+", "-", name).lower()
            result.append({"name": normalized, "version": dist.version})
    return sorted(result, key=lambda item: (item["name"], item["version"]))


def model_manifest(root: Path) -> dict:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not files:
        raise RuntimeError(f"model directory is empty: {root}")
    return {
        "file_count": len(files),
        "total_bytes": sum(file["bytes"] for file in files),
        "manifest_sha256": fingerprint(files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("tiny", "nano"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--nano-repo", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identity = {
        "collector_sha256": sha256_file(Path(__file__)),
        "environment": args.environment,
        "repositories": {
            "tinyvllm": git_identity(args.repo_root),
            "nano_vllm": git_identity(args.nano_repo),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": packages(),
        "lock": {
            "name": args.lock_file.name,
            "sha256": sha256_file(args.lock_file),
        },
        "model": {
            "model_id": args.model_id,
            **model_manifest(args.model_path),
        },
    }

    payload = {
        "schema_version": "tinyvllm.software-identity.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity_sha256": fingerprint(identity),
        "identity": identity,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(payload["identity_sha256"])


if __name__ == "__main__":
    main()