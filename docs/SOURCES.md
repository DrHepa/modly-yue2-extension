# Audited primary sources

Audit date: 2026-09-13. Code and model snapshots are pinned in `upstream.lock.json`; mutable documentation links are evidence of the audit, not instructions to load arbitrary HEAD code during inference.

## Host

- Latest release metadata: https://api.github.com/repos/lightningpixel/modly/releases/latest — returned v0.4.2, published 2026-08-28.
- Release: https://github.com/lightningpixel/modly/releases/tag/v0.4.2
- Process runner: https://github.com/lightningpixel/modly/blob/v0.4.2/electron/main/process-runner.ts
- Extension/artifact/UI types: https://github.com/lightningpixel/modly/blob/v0.4.2/src/shared/types/electron.d.ts
- Current host Python setup: https://github.com/lightningpixel/modly/blob/main/electron/main/python-setup.ts
- Current Python bridge: https://github.com/lightningpixel/modly/blob/main/electron/main/python-bridge.ts
- Main snapshot returned by tree inspection: `1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65`.
- Audited process runner blob: `758f62d22054c40aec6ddfcf29c5f6fb8e9877b5`.
- Audited release/current extension type blob: `1a4d6fde6dde6fe78f77ee90c9d152733b697d04`.

The public build-modly-extension references were read as background, not treated as current authority where their July-era audio/picker behavior differed from the release source. Their generic UI-only weight policy is not applied to this explicitly setup-managed **process** extension.

- https://github.com/DrHepa/build-modly-extension/tree/main/build-modly-extension
- Representative setup-managed audio process: https://github.com/DrHepa/modly-acestep-15-extension

## Model/runtime

- Upstream repository: https://github.com/multimodal-art-projection/YuE
- Pinned inference source: https://github.com/multimodal-art-projection/YuE/tree/88da114a67df892af0329472073b96a5ef700b93
- Native API: https://github.com/multimodal-art-projection/YuE/blob/88da114a67df892af0329472073b96a5ef700b93/src/yue2/pipeline.py
- Request/sampling protocol: https://github.com/multimodal-art-projection/YuE/blob/88da114a67df892af0329472073b96a5ef700b93/src/yue2/protocol.py
- VAE API: https://github.com/multimodal-art-projection/YuE/blob/88da114a67df892af0329472073b96a5ef700b93/src/yue2/modeling_vae.py
- Experimental FP8: https://github.com/multimodal-art-projection/YuE/blob/88da114a67df892af0329472073b96a5ef700b93/src/yue2/quantization.py
- Native artifact integrity: https://github.com/multimodal-art-projection/YuE/blob/88da114a67df892af0329472073b96a5ef700b93/src/yue2/storage.py
- Direct dependency pins: https://github.com/multimodal-art-projection/YuE/blob/88da114a67df892af0329472073b96a5ef700b93/pyproject.toml
- Covers: https://github.com/multimodal-art-projection/YuE/blob/88da114a67df892af0329472073b96a5ef700b93/docs/covers.md
- Editing: https://github.com/multimodal-art-projection/YuE/blob/88da114a67df892af0329472073b96a5ef700b93/docs/editing.md

## Checkpoints

- https://huggingface.co/m-a-p/YuE2-3B/tree/29b3558dd46954a0cd9021dc76d5c91864a0f1c7
- https://huggingface.co/m-a-p/YuE2-Vae/tree/9a94e1d0ea9f8087e98f77fa88df4a4068104d2a
- https://huggingface.co/m-a-p/YuE2-Vae-legacy/tree/b54118f0fc462f08999d1ec07e88817f4ee3f770

The native weight manifests verify safetensors contents. Setup additionally records small-file hashes. Metadata-only model-card changes do not require node-specific copies of the same checkpoint.

## Native wheels / external transcription boundary

- Official CUDA 12.8 wheel index: https://download.pytorch.org/whl/cu128/torch/ — entries inspected for torch 2.10.0, CPython 3.11/3.12, Windows x86_64, manylinux_2_28 x86_64 and aarch64.
- Optional external transcription model: https://huggingface.co/m-a-p/SheetSage2 — separate Python/PyTorch/torchaudio/MERT/FFmpeg setup, not included in this package.

## Evidence boundaries

Published wheels establish a distribution possibility, not that this extension passed on those platforms. Inspected signatures establish the intended native API, not generated-audio correctness. Upstream benchmarks are upstream measurements; this deliverable contains no newly measured YuE2 generation speed, GPU-memory requirement, musical score or full-model compatibility result.
