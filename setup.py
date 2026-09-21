"""Modly Install/Repair entry: isolated exact-Python runtime, then shared weights."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from yue2_modly.common import ROOT, STATE, clean_env, read_json, write_json, exclusive_lock
from yue2_modly.config import boolean
from yue2_modly.installer import probe, choose_lane, ensure_venv, ensure_source, run, log
from yue2_modly.paths import resolve_models_root


def parse_args(argv):
    if len(argv) == 2:
        value = json.loads(argv[1])
        if not isinstance(value, dict):
            raise ValueError("Setup argument must be a JSON object")
        return value
    if len(argv) in (4,5):
        value = {"python_exe":argv[1], "ext_dir":argv[2], "gpu_sm":int(argv[3])}
        if len(argv) == 5:
            value["cuda_version"] = int(argv[4])
        return value
    raise ValueError("Expected one Modly JSON argument or <python_exe> <ext_dir> <gpu_sm> [cuda_version]")


def setup(context):
    if Path(context.get("ext_dir", str(ROOT))).resolve() != ROOT:
        raise ValueError("ext_dir must identify this extension")
    bootstrap = Path(context.get("python_exe", sys.executable))
    if not bootstrap.is_absolute() or not bootstrap.is_file():
        raise ValueError("python_exe must be the absolute existing Modly interpreter; PATH Python is never selected")
    identity = probe(bootstrap)
    lane = choose_lane(identity, context)
    if lane["system"] == "Linux":
        libc, version = platform.libc_ver()
        if libc == "glibc" and tuple(map(int,version.split(".")[:2])) < (2,28):
            raise RuntimeError("PyTorch 2.10 wheels require glibc >=2.28")
    config = read_json(ROOT / "setup-config.json") if (ROOT / "setup-config.json").is_file() else {}
    settings_context = {**config, **context}
    models = resolve_models_root(settings_context)
    models.mkdir(parents=True, exist_ok=True)
    log(f"Lane {lane['name']}: exact Python {'.'.join(map(str,identity['version']))}, {lane['system']}/{lane['arch']}, {lane['accelerator']}")
    log(f"Persistent models_dir: {models}")
    log("Model checkpoints use CC BY-NC 4.0; the source code uses Apache-2.0. See README and upstream licenses.")
    source = ensure_source()
    python = ensure_venv(bootstrap, identity)
    env = clean_env()
    env.update(PIP_CACHE_DIR=str(models / "modly-yue2-extension" / ".pip-cache"))
    # Repair must not inherit the offline flags used by inference.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(name, None)
    run([python,"-m","pip","install","--only-binary=:all:","pip==25.3","setuptools==80.9.0","wheel==0.45.1"],"Preparing isolated packaging tools",env)
    run([python,"-m","pip","install","--only-binary=:all:",lane["torch_spec"],"--index-url",lane["index"]],"Installing the pinned native PyTorch wheel (no compilation fallback)",env)
    constraints = ROOT / ".runtime-constraints.txt"
    constraints.write_text(lane["torch_spec"] + "\n",encoding="utf-8")
    run([python,"-m","pip","install","--only-binary=:all:","-r",ROOT/"requirements.txt","-c",constraints],"Installing pinned YuE2 dependencies",env)
    run([python,"-m","pip","install","--no-deps","--no-build-isolation",source],"Installing checksum-verified YuE2 source",env)
    fast = boolean(settings_context.get("install_fast", False))
    if fast:
        if lane["system"] != "Linux" or lane["accelerator"] != "cuda":
            raise ValueError("install_fast requires Linux CUDA; native Windows vLLM is not supported")
        run([python,"-m","pip","install","--only-binary=:all:","vllm==0.19.0","triton==3.6.0","-c",constraints],"Installing optional upstream fast backend; missing/conflicting wheels fail rather than compile",env)
    run([python,ROOT/"tools/check_native_api.py"],"Checking the installed upstream API signatures",env)
    check = subprocess.run([str(python), "-m", "pip", "check"], cwd=ROOT, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check.returncode:
        report = (check.stdout + check.stderr).strip()
        # PyTorch's ARM64 cu128 wheel depends on NVIDIA's SBSA cuSPARSELt
        # package.  pip 25.3 does not consider the vendor's
        # ``manylinux2014_sbsa`` tag compatible with the host's generic
        # ``manylinux_aarch64`` tags, although the wheel is the published
        # native dependency selected by PyTorch itself.  Keep pip check
        # fatal for every other problem and validate this narrow exception
        # through the native API check immediately below.
        known_sbsa = {
            "nvidia-cusparselt-cu12 0.7.1 is not supported on this platform",
            "nvidia-cusparselt-cu13 0.8.0 is not supported on this platform",
        }
        reported_lines = {line.strip() for line in report.splitlines() if line.strip()}
        allowed = (identity["system"] == "Linux" and identity["machine"].lower() in {"aarch64", "arm64"}
                   and reported_lines and reported_lines <= known_sbsa)
        if not allowed:
            if report:
                print(report, file=sys.stderr, flush=True)
            raise subprocess.CalledProcessError(check.returncode, check.args)
        log("pip check reported the known PyTorch ARM64 SBSA cuSPARSELt tag mismatch; native API validation remains mandatory")
    run([python,"-m","yue2_modly.health","--accelerator",lane["accelerator"]],"Checking imports and small native kernels before weight downloads",env)
    run([python,"-m","yue2_modly.provision","--models-root",models],"Provisioning/reusing all three pinned checkpoints",env)
    freeze = subprocess.run([str(python),"-m","pip","freeze"],check=True,capture_output=True,text=True,env=env)
    (ROOT/".installed-requirements.txt").write_text(freeze.stdout,encoding="utf-8")
    write_json(STATE,{"extension_root":str(ROOT),"models_root":str(models),"lane":lane,"host_interpreter":identity,
                      "upstream_revision":read_json(ROOT/"upstream.lock.json")["source"]["revision"],"install_fast":fast})
    log("Setup complete. Weights are shared by all nodes and survive extension updates. GPU song inference must still be smoke-tested on this machine.")


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):
            stream.reconfigure(encoding="utf-8",errors="replace")
    try:
        with exclusive_lock(ROOT / ".setup.lock"):
            setup(parse_args(sys.argv))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        traceback.print_exc()
        log(f"ERROR: {type(exc).__name__}: {exc}. Setup did not complete. Existing checkpoints are retained; correct the error and run Repair.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
