# Validation status — 0.1.0

## Executed in this delivery

**59 unittest cases passed** in the provisioned extension venv, with 2 platform-specific skips. The adapter and contract suite passed after setup. Python source compilation passed and the generated manifest was compared byte-for-byte by a test.

Local execution environment:

| Component | Actual value |
|---|---|
| OS / machine | Linux ARM64, NVIDIA GB10 / SM121 |
| Python | CPython **3.11.9** |
| PyTorch | **2.10.0+cu130** |
| CUDA available | **Yes; CUDA 13.0** |
| NumPy | **2.2.6** |
| SoundFile | **0.13.1** |
| YuE2 model weights loaded | **Yes; all three pinned snapshots** |
| Pinned YuE2 package installed | **yue2-infer 0.1.6** |

The exact Modly CPython 3.11.9 lane was installed by `setup.py`, including the pinned source, native dependencies and all three checkpoints. The GB10 lane uses cu130 because the host GPU reports SM121; the generic ARM64 cu128 lane is not selected for this machine.

Covered checks include manifest/node/default validity, matching upstream sampling defaults, bad parameter rejection, path binding to host settings, relocation/state precedence, fail-closed missing storage, traversal and symlink rejection, setup-lane selection metadata, checkpoint-marker reuse/corruption, descriptor and data tampering, non-pickle array validation and single-terminal NDJSON subprocess behavior.

The 17 adapter-integration cases use **explicit test doubles** for YuE2. They check full and staged control flow, exact-plan reuse without replanning, edited ABC/new-request handling, full-decode metadata, score/agent extraction, VAE array plumbing, input sample-rate validation, native-style import, batch reuse/identity checks and cancellation-before-load. Small WAV/FLAC files are real files written with SoundFile, but their audio comes from the fixtures: they are **not YuE2-generated music**. The fake implementation exists only under `tests/`; production code contains no synthetic inference fallback.

## Not executed in this delivery

Windows/x86_64 lanes, private CPython 3.12, vLLM, FP8-specific execution, full-length musical-quality evaluation, throughput/VRAM benchmarking and UI Cancel behavior were not tested. Modly UI reload/registration was not driven interactively in this pass; the installed runtime path, manifest, setup state and actual process IPC were exercised directly.

The included GitHub Actions workflow declares CPU unit-test jobs for Python 3.11/3.12 on Windows x86_64, Linux x86_64 and Linux ARM64. Its execution status is tracked in GitHub Actions; source publication alone is not evidence of a successful run. A green CPU CI run must still not be described as GPU certification.

## Installer checks that will run on the user's machine

The installer verifies the selected host interpreter, chooses only recognized lanes, creates/reuses an exact-version venv, checks pinned source Git blob identities, installs pinned direct dependency versions using binary wheels, checks native API signatures and dependency consistency, then runs small import/kernel/WAV probes before weight downloads. It verifies checkpoints and records installed versions afterward.

These checks were exercised online on the GB10 host. The cu130 native wheels, API signatures, imports, BF16 probe, checkpoint hashes and real model loading all passed. Optional vLLM dependency compatibility remains untested and may fail closed when its wheels are unavailable. There is no compiler fallback.

## Acceptance matrix before a tested release

| Target | CPython | Installation | Real CUDA song | Status |
|---|---|---|---|---|
| Windows x86_64 | exact host 3.11 | Pending | Pending | Candidate lane |
| Windows x86_64 | private host 3.12 | Pending | Pending | Candidate lane |
| Linux x86_64 | exact host 3.11 | Pending | Pending | Candidate lane |
| Linux x86_64 | private host 3.12 | Pending | Pending | Candidate lane |
| Linux ARM64 GB10/SM121 | exact host 3.11.9 | Passed | Passed | PyTorch 2.10.0+cu130; all 13 nodes exercised |
| Linux ARM64 | private host 3.12 | Pending | Pending | Candidate lane |

Use a CUDA/BF16-capable GPU for native song-generation acceptance. Record the actual driver, GPU, compute capability, torch wheel, Python patch version and dependency freeze. Hardware compatibility is independent of successful extension registration.

## Required acceptance scenarios

1. Clean GitHub/local installation: manifest discovery, setup output, correct active `venv`, all checkpoint paths strictly under real `models_dir`, successful native API check and kernel probes.
2. Generate a short real song with `tools/smoke.py`; verify nonempty finite 48 kHz stereo audio, native result integrity and wrapper bundle integrity. Run one full-length ordinary generation separately for quality and resource assessment.
3. Exercise `full`, `melody`, `off`, external ABC, exact-plan reuse, every split stage, both VAEs, full/tiled decode and mean/seeded posterior encode. Test a genuinely long plan and token truncation.
4. Compare a split-stage run against the intended native behavior with identical parameters/seed and checkpoint identities. Do not equate unmeasured audio similarity with numerical reproducibility across CUDA backends.
5. Repair twice without redownloading unchanged weights; remove/corrupt one weight file; interrupt download; move models storage; update wrapper; change Python lane. Check that correct files are reused and partial installations are never marked ready.
6. Batch interruption and resume: same identity reuses only verified complete songs; changed request/config/checkpoint refuses the old batch; incomplete item is preserved and restarted. Check no duplicate output overwrite.
7. Exercise paths with spaces/non-ASCII, malformed JSON, missing files, corrupt bundles, invalid token arrays, missing native wheels, old CUDA driver and unsupported BF16/FP8 hardware.
8. Test Modly UI cancellation only after the host termination implementation described in `MODLY_GAPS.md`. Confirm process-tree exit, prompt settlement and GPU-memory release; the current host no-op is not solved by wrapper-only unit tests.

Initial source publication: [DrHepa/modly-yue2-extension](https://github.com/DrHepa/modly-yue2-extension). No GitHub issue, PR or registry submission is included in this publication. The Linux ARM64/GB10 lane is locally accepted; this is not a claim of fully tested cross-platform support.
