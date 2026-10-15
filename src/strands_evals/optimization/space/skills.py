"""Skill libraries as values: checking one, and storing each version as an immutable, content-addressed directory.

The skill format and its checks are the skill-library formula's (ported from harness-optimizer's skill-library formula):
a library is a directory of ``<name>/SKILL.md`` (plus any files a skill needs), and a skill whose frontmatter has no
description can never be loaded.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def check_library(path: str) -> str | None:
    """Why ``path`` is not a usable skill library, if it is not."""
    from .._reflection import skill_dirs_in, validate_skill_dir

    if not os.path.isdir(path):
        return f"no skill library at {path}"
    problems = [
        f"{os.path.basename(os.path.normpath(d))}: {'; '.join(p)}"
        for d in skill_dirs_in(path)
        if (p := validate_skill_dir(d))
    ]
    return "; ".join(problems) or None


def digest(path: str | Path) -> str:
    """A content hash of a directory: every file's relative path and bytes."""
    root = Path(path)
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(root)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def store(members: dict[str, str], into: str | Path) -> str:
    """Write a library (``{name: skill directory}``) as one version under ``into/<hash>/``, and return its path.
    The same library always lands in the same directory, which is never changed afterwards."""
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=into) as tmp:
        for name, src in sorted(members.items()):
            shutil.copytree(src, Path(tmp) / name)
        target = into / digest(tmp)
        if not target.exists():
            os.replace(tmp, target)
            os.makedirs(tmp)  # TemporaryDirectory removes what it made
    return str(target.resolve())
