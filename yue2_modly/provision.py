"""Setup-only weight provisioning. Verified files survive Repair and ABI switches."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from .common import ROOT, read_json, write_json, sha256, safe_child, exclusive_lock
from .paths import asset_path

ALLOWED = ["config.json", "generation_config.json", "yue2_generation_config.json", "weights_manifest.json",
           "model.safetensors", "model.safetensors.index.json", "model-?????-of-?????.safetensors",
           "qwen.tiktoken", "modeling_yue2.py", "modeling_vae.py", "LICENSE", "THIRD_PARTY_NOTICES.md",
           "licenses/SnakeBeta-NVIDIA-MIT.txt", "licenses/stable-audio-tools-MIT.txt"]


def verified_existing(folder: Path, spec: dict) -> bool:
    marker = folder / ".complete.json"
    if not marker.is_file():
        return False
    try:
        saved = read_json(marker)
        if saved["repo_id"] != spec["repo_id"] or saved["revision"] != spec["revision"] or not saved["files"]:
            return False
        for name, info in saved["files"].items():
            path = safe_child(folder, name)
            if not path.is_file() or path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
                return False
        return True
    except (ValueError, OSError, KeyError, TypeError):
        return False


def _provision(models_root: Path):
    from huggingface_hub import snapshot_download, hf_hub_download
    from yue2.storage import model_identity
    lock = read_json(ROOT / "upstream.lock.json")
    # Download metadata and locks stay under models_dir, not a global HF cache.
    cache = models_root / "modly-yue2-extension" / ".hub"
    os.environ.update(HF_HOME=str(cache), HF_HUB_DISABLE_TELEMETRY="1")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or False
    for key, spec in lock["weights"].items():
        folder = asset_path(models_root, key)
        print(f"[YuE2 setup] Checking {spec['repo_id']} at {spec['revision']}", flush=True)
        if verified_existing(folder, spec):
            print("[YuE2 setup] Verified snapshot reused; no weight download", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        # Remove only the readiness marker, never the entire snapshot.
        marker = folder / ".complete.json"
        try:
            previous = read_json(marker) if marker.is_file() else {}
        except (ValueError, OSError):
            previous = {}
        if not isinstance(previous, dict) or not isinstance(previous.get("files", {}), dict):
            previous = {}
        marker.unlink(missing_ok=True)
        # Hub local_dir metadata may consider a corrupt local file up to date.
        # Force only files whose recorded content no longer matches.
        bad = []
        for name, info in previous.get("files", {}).items():
            file = safe_child(folder, name)
            if file.is_file() and (file.stat().st_size != info["bytes"] or sha256(file) != info["sha256"]):
                bad.append(name)
        snapshot_download(spec["repo_id"], revision=spec["revision"], local_dir=str(folder),
                          cache_dir=str(cache), allow_patterns=ALLOWED, token=token, max_workers=2)
        for name in bad:
            hf_hub_download(spec["repo_id"], filename=name, revision=spec["revision"], local_dir=str(folder),
                            cache_dir=str(cache), token=token, force_download=True)
        expected = read_json(folder / "weights_manifest.json")["files"]
        # Resume an interrupted first setup whose local weights became corrupt.
        for name, info in expected.items():
            path = safe_child(folder, name)
            if not path.is_file() or sha256(path) != info["sha256"]:
                hf_hub_download(spec["repo_id"], filename=name, revision=spec["revision"], local_dir=str(folder),
                                cache_dir=str(cache), token=token, force_download=True)
        model_identity(folder, verify=True)
        required = ["config.json", "weights_manifest.json", "LICENSE"] + (["qwen.tiktoken"] if key == "mot" else [])
        if any(not (folder / name).is_file() for name in required):
            raise ValueError(f"Incomplete {key} checkpoint metadata")
        entries = {}
        for path in folder.rglob("*"):
            relative = path.relative_to(folder).as_posix()
            if path.is_file() and not relative.startswith(".cache/") and path.name != ".complete.json":
                safe_child(folder, relative)
                entries[relative] = {"sha256":sha256(path), "bytes":path.stat().st_size}
        write_json(marker, {"repo_id":spec["repo_id"], "revision":spec["revision"], "files":entries})
        print(f"[YuE2 setup] Ready: {key}", flush=True)


def provision(models_root: Path):
    with exclusive_lock(models_root / "modly-yue2-extension" / ".weights.lock"):
        _provision(models_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    args = parser.parse_args()
    provision(args.models_root)
