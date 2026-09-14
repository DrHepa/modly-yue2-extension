"""Versioned, hash-checked stage artifacts. No pickle, code loading or shell commands."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
from .common import BUNDLE_SCHEMA, read_json, write_json, sha256, safe_child


def seal(directory: Path, kind: str, metadata: dict) -> dict:
    artifacts = {}
    for path in sorted(directory.iterdir()):
        if path.name in {"bundle.json", "CANCEL", ".running"} or path.name.endswith(".tmp"):
            continue
        if path.is_symlink():
            raise ValueError("Cannot seal symlink artifacts")
        if path.is_file():
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    data = dict(schema=BUNDLE_SCHEMA, kind=kind, metadata=metadata, artifacts=artifacts)
    write_json(directory / "bundle.json", data)
    return descriptor(directory)


def descriptor(directory: Path) -> dict:
    path = directory / "bundle.json"
    return dict(schema=BUNDLE_SCHEMA, directory=str(directory.resolve()), sha256=sha256(path))


def verify(directory: Path, expected_digest: str | None = None) -> dict:
    path = directory / "bundle.json"
    if path.is_symlink():
        raise ValueError("Invalid bundle manifest")
    if expected_digest and sha256(path) != expected_digest:
        raise ValueError("Bundle descriptor no longer matches its manifest")
    data = read_json(path)
    if data.get("schema") != BUNDLE_SCHEMA or not isinstance(data.get("artifacts"), dict):
        raise ValueError("Unsupported bundle schema")
    if data.get("kind") not in {"plan", "semantic", "latents", "encoded", "song", "decoded", "batch"}:
        raise ValueError("Unknown bundle stage")
    for name, info in data["artifacts"].items():
        file = safe_child(directory, name)
        if not file.is_file() or file.stat().st_size != info["bytes"] or sha256(file) != info["sha256"]:
            raise ValueError(f"Missing or changed artifact: {name}; edit ABC as a NEW request, not an exact saved plan")
    return data


def locate(text: str = "", path: str = "") -> tuple[Path, dict]:
    expected = None
    if text.strip().startswith("{"):
        data = json.loads(text)
        if data.get("schema") != BUNDLE_SCHEMA:
            raise ValueError("Expected a YuE2 bundle descriptor")
        path = data.get("directory", "")
        expected = data.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Missing descriptor checksum")
    elif text.strip() and not path:
        path = text.strip()
    directory = Path(path).expanduser()
    if not path or not directory.is_absolute():
        raise ValueError("Connect a YuE2 bundle descriptor or supply an absolute bundle_path")
    if directory.is_file():
        directory = directory.parent
    directory = directory.resolve()
    return directory, verify(directory, expected)


def array(path: Path, kind: str) -> np.ndarray:
    if path.stat().st_size > 1024 * 1024 * 1024:
        raise ValueError("Array exceeds 1 GiB safety limit")
    data = np.load(path, allow_pickle=False)
    if not isinstance(data, np.ndarray):
        raise ValueError("Expected a single NPY array")
    if kind == "semantic":
        if data.ndim != 1 or data.dtype.kind not in "iu" or not len(data) or np.any(data < 0) or np.any(data >= 32768):
            raise ValueError("Semantic tokens must be a nonempty integer vector in [0,32768)")
    elif kind == "latent":
        good_shape = (data.ndim == 2 and data.shape[1] == 64 and data.shape[0] > 0) or (data.ndim == 3 and data.shape[:2] == (1,64) and data.shape[2] > 0)
        if not good_shape or data.dtype.kind != "f" or not np.isfinite(data).all():
            raise ValueError("Latents must be finite float [T,64] or [1,64,T]")
    return data


def copy_stage(source: Path, destination: Path, names: tuple[str, ...]) -> None:
    for name in names:
        file = safe_child(source, name)
        if file.is_file():
            shutil.copyfile(file, destination / name)

PLAN_FILES = ("plan.json", "plan_manifest.json", "abc_tokens.npy", "prefix.npy", "score.abc")
SEMANTIC_FILES = PLAN_FILES + ("semantic.npy", "semantic_info.json")
