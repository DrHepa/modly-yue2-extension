"""Host-derived Python lanes and pinned source preparation. No build-tool fallback."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from .common import ROOT, clean_env, read_json, safe_child

PROBE = """import sys,platform,struct,json
print(json.dumps({'version':list(sys.version_info[:3]),'implementation':sys.implementation.name,
'bits':struct.calcsize('P')*8,'machine':platform.machine(),'system':platform.system(),
'executable':sys.executable,'base_executable':getattr(sys,'_base_executable',sys.executable)}))"""


def log(message):
    print("[YuE2 setup] " + message, flush=True)


def probe(executable: Path) -> dict:
    result = subprocess.run([str(executable), "-I", "-c", PROBE], check=True, capture_output=True,
                            encoding="utf-8", timeout=30, env=clean_env())
    return json.loads(result.stdout)


def choose_lane(identity: dict, context: dict) -> dict:
    if identity["implementation"] != "cpython" or identity["bits"] != 64 or identity["version"][:2] not in ([3,11], [3,12]):
        raise ValueError("[PYTHON_ABI_UNSUPPORTED] Use the exact 64-bit Modly CPython 3.11 interpreter or the private Modly CPython 3.12 interpreter")
    machine = identity["machine"].lower()
    arch = {"amd64":"x86_64", "x64":"x86_64", "x86_64":"x86_64", "aarch64":"aarch64", "arm64":"aarch64"}.get(machine)
    system = identity["system"]
    if (system, arch) not in {("Windows","x86_64"),("Linux","x86_64"),("Linux","aarch64")}:
        raise ValueError("This installer provides Windows x86_64, Linux x86_64 and Linux ARM64 lanes only")
    accelerator = context.get("accelerator", "cuda")
    if accelerator not in {"cuda", "cpu"}:
        raise ValueError("Unsupported accelerator; no implicit CUDA-to-CPU fallback")
    reported = context.get("cuda_version")
    if accelerator == "cuda" and reported:
        # Modly reports e.g. 128 for CUDA 12.8. Unknown formats are rejected.
        text = str(reported).strip()
        if "." in text:
            major, minor = text.split(".")[:2]
            reported = int(major)*10 + int(minor)
        else:
            reported = int(text)
        if reported < 128:
            raise ValueError("[CUDA_LANE_UNSUPPORTED] The pinned PyTorch CUDA lane requires a driver compatible with CUDA 12.8; update the driver or deliberately use CPU")
    # NVIDIA GB10 is SM 12.1.  The published ARM64 cu128 wheel advertises
    # support only through SM 12.0 and fails even on a tiny CUDA kernel.
    # Modly's current GPU probe caps its reported CUDA value at 12.8, so the
    # SM 12.1 evidence is the authoritative selector for the validated cu130
    # ARM64 lane.
    gb10 = system == "Linux" and arch == "aarch64" and int(context.get("gpu_sm", 0)) >= 121
    if gb10 and accelerator == "cuda":
        flavor, minimum = "cu130", 130
        if reported and reported not in (128, 129) and reported < minimum:
            raise ValueError("[CUDA_LANE_UNSUPPORTED] GB10 requires a driver compatible with CUDA 13.0")
    else:
        flavor, minimum = "cu128", 128
        if accelerator == "cuda" and reported and reported < minimum:
            raise ValueError("[CUDA_LANE_UNSUPPORTED] The pinned PyTorch cu128 lane requires a driver compatible with CUDA 12.8; update the driver or deliberately use CPU")
    torch_spec = "torch==2.10.0+" + flavor if accelerator == "cuda" else "torch==2.10.0"
    return {"name": "modly-cp311" if identity["version"][1] == 11 else "private-cp312",
            "python":identity["version"], "system":system, "arch":arch, "accelerator":accelerator,
            "torch":"2.10.0", "torch_spec":torch_spec, "index": "https://download.pytorch.org/whl/" + (flavor if accelerator == "cuda" else "cpu"),
            "cuda_flavor": flavor}


def run(command, label, env=None):
    log(label)
    subprocess.run([str(x) for x in command], cwd=ROOT, check=True, env=env or clean_env())


def ensure_venv(bootstrap: Path, identity: dict, root: Path = ROOT) -> Path:
    target = root / "venv" / ("Scripts/python.exe" if identity["system"] == "Windows" else "bin/python")
    if (root / "venv").exists():
        try:
            current = probe(target)
            if current["version"] != identity["version"] or current["machine"] != identity["machine"] or current["bits"] != identity["bits"] or Path(current["base_executable"]).resolve() != Path(identity["base_executable"]).resolve():
                raise ValueError("Runtime ABI differs from the host")
            return target
        except (OSError, ValueError, subprocess.SubprocessError):
            backup = root / ("venv.previous-" + uuid.uuid4().hex[:8])
            (root / "venv").rename(backup)
            log(f"Preserved previous runtime as {backup.name}; checkpoints are unaffected")
    run([bootstrap, "-m", "venv", root / "venv"], "Creating isolated venv with the exact host interpreter")
    created = probe(target)
    if created["version"] != identity["version"]:
        raise RuntimeError("Created venv does not match the host's exact Python version")
    return target


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def ensure_source(root: Path = ROOT) -> Path:
    lock = read_json(root / "upstream.lock.json")["source"]
    source = root / "vendor" / ("YuE-" + lock["revision"])
    source.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    # Embedded CPython may need Modly's already-installed CA bundle.
    try:
        import certifi
        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    for name, expected in lock["git_blobs"].items():
        path = safe_child(source, name)
        if path.is_file() and git_blob(path.read_bytes()) == expected:
            continue
        url = f"https://raw.githubusercontent.com/{lock['repository']}/{lock['revision']}/{name}"
        data = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60, context=context) as response:
                    data = response.read(4 * 1024 * 1024 + 1)
                if len(data) > 4 * 1024 * 1024 or git_blob(data) != expected:
                    raise ValueError(f"Pinned source checksum mismatch: {name}")
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        log(f"Verified upstream source: {name}")
    return source
