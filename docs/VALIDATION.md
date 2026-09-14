# Validation status — implementation candidate 0.1.0

## Executed in this delivery

**52 unittest cases passed** in the local container, including additional parameterized subcases. The complete successful output is in `docs/local-unittest.txt`. Python source compilation also passed. The generated manifest was regenerated and compared byte-for-byte by a test.

Local execution environment:

| Component | Actual value |
|---|---|
| OS / machine | Linux x86_64, glibc 2.41 |
| Python | CPython **3.13.5** |
| PyTorch | **2.10.0+cpu** |
| CUDA available | **No** |
| NumPy | **2.3.5** |
| SoundFile | **0.13.1** |
| YuE2 model weights loaded | **No** |
| Pinned YuE2 package installed | **No** |

**These are not the requested 3.11 and 3.12 installation lanes.** The container interpreter and some package versions differ from the delivered dependency pins. Passing these tests demonstrates wrapper logic under the stated environment, not binary compatibility or native-model correctness on either target lane.

Covered checks include manifest/node/default validity, matching upstream sampling defaults, bad parameter rejection, path binding to host settings, relocation/state precedence, fail-closed missing storage, traversal and symlink rejection, setup-lane selection metadata, checkpoint-marker reuse/corruption, descriptor and data tampering, non-pickle array validation and single-terminal NDJSON subprocess behavior.

The 17 adapter-integration cases use **explicit test doubles** for YuE2. They check full and staged control flow, exact-plan reuse without replanning, edited ABC/new-request handling, full-decode metadata, score/agent extraction, VAE array plumbing, input sample-rate validation, native-style import, batch reuse/identity checks and cancellation-before-load. Small WAV/FLAC files are real files written with SoundFile, but their audio comes from the fixtures: they are **not YuE2-generated music**. The fake implementation exists only under `tests/`; production code contains no synthetic inference fallback.

## Not executed here

No clean Modly installation, setup download, pinned source installation/import, private-host run, CPython 3.11/3.12 execution, Windows execution, ARM64 execution, CUDA inference, FP8 kernel run, vLLM inference, complete-song generation, throughput/VRAM benchmark or UI Cancel test was performed. Network access from the container was unavailable. Research sources were read using web/connector tools, not by running the installer.

The included GitHub Actions workflow declares CPU unit-test jobs for Python 3.11/3.12 on Windows x86_64, Linux x86_64 and Linux ARM64. Its execution status is tracked in GitHub Actions; source publication alone is not evidence of a successful run. A green CPU CI run must still not be described as GPU certification.

## Installer checks that will run on the user's machine

The installer verifies the selected host interpreter, chooses only recognized lanes, creates/reuses an exact-version venv, checks pinned source Git blob identities, installs pinned direct dependency versions using binary wheels, checks native API signatures and dependency consistency, then runs small import/kernel/WAV probes before weight downloads. It verifies checkpoints and records installed versions afterward.

These checks are implemented but **were not exercised online in this delivery**. The native wheels and API signatures were inspected at source level. Optional vLLM dependency compatibility must be tested explicitly; it may fail closed on a platform with missing/incompatible wheels. There is no compiler fallback.

## Acceptance matrix before a tested release

| Target | CPython | Installation | Real CUDA song | Status |
|---|---|---|---|---|
| Windows x86_64 | exact host 3.11 | Pending | Pending | Candidate lane |
| Windows x86_64 | private host 3.12 | Pending | Pending | Candidate lane |
| Linux x86_64 | exact host 3.11 | Pending | Pending | Candidate lane |
| Linux x86_64 | private host 3.12 | Pending | Pending | Candidate lane |
| Linux ARM64 | exact host 3.11 | Pending | Pending | Candidate lane |
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

Initial source publication: [DrHepa/modly-yue2-extension](https://github.com/DrHepa/modly-yue2-extension). No GitHub issue, PR or registry submission is included in this publication. This package is ready for source review and target-machine acceptance work, not for a claim of fully tested cross-platform support.
