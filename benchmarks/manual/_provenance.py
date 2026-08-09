"""Stable content fingerprinting for manual benchmark records."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def worktree_fingerprint(root: Path) -> dict[str, str | bool]:
    """Return the commit and a hash of tracked and untracked working content."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    hasher = hashlib.sha256(diff.encode())
    for relative in sorted(untracked):
        path = root / relative
        hasher.update(relative.encode())
        if path.is_file():
            hasher.update(path.read_bytes())
    digest = hasher.hexdigest()[:16]
    return {
        "commit": commit,
        "dirty": bool(status),
        "worktree_fingerprint": f"{commit[:12]}-{'dirty' if status else 'clean'}-{digest}",
    }
