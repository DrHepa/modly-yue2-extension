"""Bind storage to the real host; never guess where to put 8+ GB of weights."""
from __future__ import annotations
import os
from pathlib import Path
from .common import ROOT, STATE, EXTENSION_ID, absolute_directory, read_json, safe_child


def _one(values: dict, keys: tuple[str, ...]) -> Path | None:
    paths = {absolute_directory(values[k], k) for k in keys if values.get(k)}
    if len(paths) > 1:
        raise ValueError("Conflicting models_dir settings")
    return next(iter(paths), None)


def _separate(path: Path, root: Path) -> Path:
    if path == root or path.is_relative_to(root) or root.is_relative_to(path):
        raise ValueError("models_dir must be separate from extension code and runtimes")
    return path


def resolve_models_root(context=None, *, root=ROOT, state_file=STATE, env=None) -> Path:
    context = context or {}
    env = os.environ if env is None else env
    root = root.resolve()
    explicit = _one(context, ("models_dir", "modelsDir"))
    if explicit is not None:
        return _separate(explicit, root)
    explicit = _one(env, ("MODELS_DIR", "MODLY_MODELS_DIR"))
    if explicit is not None:
        return _separate(explicit, root)
    candidates = [root.parent.parent / "settings.json"]
    if env.get("MODLY_USER_DATA"):
        candidates.insert(0, Path(env["MODLY_USER_DATA"]) / "settings.json")
    if context.get("python_exe"):
        candidates.extend(p / "settings.json" for p in Path(context["python_exe"]).parents)
    config_home = Path(env.get("APPDATA", str(Path.home() / "AppData/Roaming"))) if os.name == "nt" else Path(env.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    candidates.extend(config_home / name / "settings.json" for name in ("modly", "Modly"))
    matches = set()
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            settings = read_json(candidate)
            ext = absolute_directory(settings.get("extensionsDir", str(candidate.parent / "extensions")), "extensionsDir")
            if (ext / EXTENSION_ID).resolve() == root:
                matches.add(absolute_directory(settings.get("modelsDir", str(candidate.parent / "models")), "modelsDir"))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            if candidate == root.parent.parent / "settings.json":
                raise ValueError("Cannot read this Modly installation's settings.json") from exc
    if len(matches) > 1:
        raise ValueError("Ambiguous Modly installations: provide models_dir explicitly")
    if matches:
        return _separate(matches.pop(), root)
    if state_file.is_file():
        state = read_json(state_file)
        # Compare canonical paths, not raw strings: Windows can spell the same
        # directory with an 8.3 alias, different casing, or forward slashes.
        try:
            saved_root = absolute_directory(state.get("extension_root"), "saved extension_root")
        except (OSError, ValueError, RuntimeError):
            saved_root = None  # Invalid/unresolvable state must not bind to this host.
        if saved_root == root:
            return _separate(absolute_directory(state["models_root"], "saved models_root"), root)
    raise ValueError("[MODELS_DIR_MISSING] Set MODELS_DIR to Modly Settings > Storage > Models, then run Repair/setup. No download was attempted into a guessed folder.")


def asset_root(models_root: Path) -> Path:
    return safe_child(models_root, EXTENSION_ID + "/checkpoints")


def asset_path(models_root: Path, key: str) -> Path:
    lock = read_json(ROOT / "upstream.lock.json")
    info = lock["weights"][key]
    return safe_child(asset_root(models_root), info["directory"] + "/" + info["revision"])
