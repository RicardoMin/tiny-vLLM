from __future__ import annotations

import os
import subprocess

import pytest

from benchmarks.tools.collect_env import git_identity


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Identity Test")
    git(tmp_path, "config", "user.email", "identity@example.invalid")
    (tmp_path / "tracked.py").write_bytes(b"original\n")
    (tmp_path / ".gitignore").write_text("ignored/\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


def test_stable_identity_ignores_metadata_and_ignored_files(repo):
    before = git_identity(repo)
    os.utime(repo / "tracked.py", (1, 1))
    (repo / ".git" / "irrelevant").write_bytes(b"metadata")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "output.bin").write_bytes(b"result")

    assert git_identity(repo) == before
    assert before["dirty"] is False
    assert before["status_sha256"] is None
    assert len(before["commit"]) == 40
    assert len(before["worktree_sha256"]) == 64


@pytest.mark.parametrize("staged", [False, True])
def test_content_changes_with_identical_status_are_distinguished(repo, staged):
    tracked = repo / "tracked.py"
    tracked.write_bytes(b"first\x00\xff")
    if staged:
        git(repo, "add", "tracked.py")
    before = git_identity(repo)

    tracked.write_bytes(b"second\x00\xfe")
    if staged:
        git(repo, "add", "tracked.py")
    after = git_identity(repo)

    assert before["dirty"] and after["dirty"]
    assert before["status_sha256"] == after["status_sha256"]
    assert before["worktree_sha256"] != after["worktree_sha256"]


def test_index_changes_are_detected_even_with_identical_worktree(repo):
    tracked = repo / "tracked.py"
    tracked.write_bytes(b"staged A")
    git(repo, "add", "tracked.py")
    tracked.write_bytes(b"actual source")
    before = git_identity(repo)

    tracked.write_bytes(b"staged B")
    git(repo, "add", "tracked.py")
    tracked.write_bytes(b"actual source")
    after = git_identity(repo)

    assert before["status_sha256"] == after["status_sha256"]
    assert before["worktree_sha256"] != after["worktree_sha256"]


def test_untracked_binary_and_unusual_filenames(repo):
    git(repo, "config", "core.quotePath", "false")
    name = os.fsdecode(b"new \t\n\xff.bin")
    path = repo / name
    path.write_bytes(b"\x00\xff")
    before = git_identity(repo)
    assert git_identity(repo) == before

    path.write_bytes(b"\x00\xfe")
    after = git_identity(repo)
    assert before["status_sha256"] == after["status_sha256"]
    assert before["worktree_sha256"] != after["worktree_sha256"]

    git(repo, "add", "--", name)
    assert git_identity(repo)["worktree_sha256"] != after["worktree_sha256"]


def test_deletion_rename_and_executable_mode_have_distinct_identities(repo):
    original = git_identity(repo)
    tracked = repo / "tracked.py"
    tracked.chmod(0o755)
    executable = git_identity(repo)
    tracked.chmod(0o644)
    assert git_identity(repo) == original

    tracked.rename(repo / "renamed.py")
    renamed = git_identity(repo)
    git(repo, "add", "-A")
    staged_rename = git_identity(repo)
    (repo / "renamed.py").unlink()
    deleted = git_identity(repo)
    git(repo, "add", "-A")
    staged_delete = git_identity(repo)

    identities = [original, executable, renamed, staged_rename, deleted, staged_delete]
    assert len({item["worktree_sha256"] for item in identities}) == len(identities)


def test_symlink_fingerprints_target_without_following_it(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "weights.bin"
    outside.write_bytes(b"model A")
    link = repo / "weights-link"
    link.symlink_to(outside)
    git(repo, "add", "weights-link")
    before = git_identity(repo)

    outside.write_bytes(b"model B")
    assert git_identity(repo) == before

    link.unlink()
    link.symlink_to("missing-target")
    assert git_identity(repo)["worktree_sha256"] != before["worktree_sha256"]


def test_special_files_are_rejected_instead_of_reading_forever(repo):
    (repo / "tracked.py").unlink()
    os.mkfifo(repo / "tracked.py")
    with pytest.raises(ValueError, match="special file"):
        git_identity(repo)


def test_subdirectory_has_whole_repository_identity(repo):
    child = repo / "subdirectory"
    child.mkdir()
    assert git_identity(child) == git_identity(repo)
