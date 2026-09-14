# Third-party notices

This archive contains a new Modly integration wrapper, tests, documentation and source/checkpoint locks. It does not contain model weights or a copy of the upstream YuE2 runtime.

At setup time, source is downloaded from the pinned `multimodal-art-projection/YuE` revision, verified against recorded Git blob hashes and installed into the extension's isolated venv. Its `LICENSE` (Apache-2.0), `MODEL_LICENSE`, `THIRD_PARTY_NOTICES.md`, and the stable-audio-tools / SnakeBeta license files are preserved in the pinned source directory. Checkpoint LICENSE and THIRD_PARTY_NOTICES files are downloaded alongside the weights.

YuE2 source: Multimodal Art Projection / YuE contributors, Apache License 2.0.

YuE2-3B, YuE2-Vae and YuE2-Vae-legacy checkpoints: CC BY-NC 4.0 as declared by their upstream model cards and checkpoint licenses. The MIT wrapper license does not override those conditions.

Modly: lightningpixel / Modly contributors. This extension targets the public process contract; it is not an official Modly or Multimodal Art Projection release.

Python, PyTorch, Transformers, Hugging Face Hub, safetensors, tiktoken, NumPy, SoundFile, Accelerate and optional vLLM/Triton keep their own licenses. Dependencies are installed, not relicensed by this wrapper. No affiliation, performance endorsement or commercial-use clearance is implied.
