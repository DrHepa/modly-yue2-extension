"""Pre-download dependency/kernel probe. This is not a full-model inference test."""
import argparse
import importlib.metadata
import json
import tempfile
from pathlib import Path


def check(accelerator: str) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    import safetensors
    import tiktoken
    import accelerate
    from yue2 import YuE2Pipeline
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.modeling_vae import YuE2VAE
    from yue2.protocol import GenerationConfig
    report = {name:importlib.metadata.version(name) for name in ("torch","transformers","numpy","soundfile","yue2-infer")}
    if accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable after installing the cu128 wheel; weights were not downloaded")
        capability = torch.cuda.get_device_capability()
        report.update(device=torch.cuda.get_device_name(), capability=capability,
                      native_bf16=torch.cuda.is_bf16_supported(including_emulation=False))
        # Check wheel architecture/kernel support (especially Linux ARM64 GPUs).
        x = torch.ones((32,32),device="cuda",dtype=torch.float32)
        (x @ x).sum().item()
        if report["native_bf16"]:
            x = x.to(torch.bfloat16)
            (x @ x).sum().item()
        else:
            print("[YuE2 setup] WARNING: this GPU can run FP32 VAE tools but is not supported for native BF16 song generation", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.wav"
        sf.write(str(path), np.zeros((1920,2),np.float32), 48000, subtype="FLOAT")
        audio, rate = sf.read(str(path), always_2d=True)
        assert audio.shape == (1920,2) and rate == 48000
    report["probe"] = "imports, small native kernels and WAV I/O only; not YuE2 inference"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator",choices=("cuda","cpu"),required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.accelerator),indent=2),flush=True)
