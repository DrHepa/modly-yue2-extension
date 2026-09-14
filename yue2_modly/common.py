"""Dependency-free identities, safe paths and atomic small-file writes."""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ID = "modly-yue2-extension"
STATE = ROOT / ".setup-state.json"
BUNDLE_SCHEMA = "yue2-modly-bundle/v1"


def read_json(path: Path, limit: int = 32 * 1024 * 1024) -> Any:
    if path.stat().st_size > limit:
        raise ValueError(f"JSON file exceeds {limit} bytes: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: invalid(f"Non-finite JSON number: {x}"))


def invalid(message: str):
    raise ValueError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Invalid relative artifact path")
    parts = name.split("/")
    if any(x in ("", ".", "..") for x in parts):
        raise ValueError("Unsafe relative artifact path")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink artifacts are not accepted")
        if current.exists():
            attrs = getattr(current.lstat(), "st_file_attributes", 0)
            if attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError("Reparse-point artifacts are not accepted")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes its directory")
    return current


def absolute_directory(value: str, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an absolute directory")
    path = Path(value).expanduser()
    if not path.is_absolute() or (path.exists() and not path.is_dir()):
        raise ValueError(f"{label} must be an absolute directory")
    return path.resolve()


def file_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", value):
        raise ValueError("Song id must be filename-safe, 1–180 ASCII characters")
    return value


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE", "PIP_USER",
                 "PIP_TARGET", "PIP_PREFIX", "PIP_REQUIRE_VIRTUALENV", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(name, None)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
    return env


@contextmanager
def exclusive_lock(path: Path, timeout: float = 60.0):
    """Cross-platform advisory lock; process exit releases it without stale PID files."""
    import time
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    deadline = time.monotonic() + timeout
    acquired = False
    try:
        while True:
            try:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Another YuE2 setup is using this storage; retry Repair after it finishes')
                time.sleep(.2)
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
