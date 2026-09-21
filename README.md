# YuE2 for Modly — process extension

**Version 0.1.0 — validated on the local Modly GB10 host.** This repository implements the published YuE2 inference API as thirteen Modly process nodes. The pinned runtime, all three checkpoints, real CUDA generation and every node's smoke path have been exercised on Linux ARM64/GB10; other platforms remain separate candidate lanes. See [validation status](docs/VALIDATION.md).

## Target and upstream

The host contract was inspected in **Modly v0.4.2**, the latest public release returned by GitHub during the 2026-09-13 audit, and compared with relevant `main` files. This is **YuE2**, not the older YuE v1 two-stage inference implementation. The source is pinned to `multimodal-art-projection/YuE@88da114a67df892af0329472073b96a5ef700b93` (`yue2-infer` 0.1.6). Model snapshots are separately pinned in `upstream.lock.json`.

Extension integration: **DrHepa**. Upstream research, model implementation and checkpoints: **Multimodal Art Projection and the YuE contributors**. Modly: **lightningpixel and contributors**. The wrapper does not change ownership or licensing of those projects.

## Installation

Install this extension using Modly's **Install from GitHub** action with:

```text
https://github.com/DrHepa/modly-yue2-extension
```

This is the extension repository. Do not paste the upstream YuE repository as the extension repository: it is the model source, not this Modly wrapper.

For local development, extract the package and install its root folder as a local extension. Local-folder installation in the inspected host links the directory but does not provision dependencies; run **Repair** where available or invoke root `setup.py` with Modly's exact Python executable and its normal setup JSON. Restart/reload extensions after changing the active Python lane.

`setup.py` accepts Modly's one-JSON-argument protocol and the legacy positional form:

```text
<absolute-Modly-python> setup.py <absolute-Modly-python> <absolute-extension-root> <gpu_sm> [cuda_version]
```

Do not replace the first executable with an unrelated `python` from PATH. The executable supplied in `python_exe` is probed and creates `<extension>/venv`. Its complete `major.minor.micro` version and base interpreter must match when reusing an environment.

The optional `setup-config.json` can be copied from `setup-config.example.json`. `install_fast: true` requests the upstream vLLM/Triton extras on Linux CUDA. A `models_dir` absolute path may be added for explicit manual setup when the host's storage settings cannot be resolved. No credentials should be stored in this file. Environment `HF_TOKEN` is accepted by setup for public-Hub authentication/rate limits, but is not saved in setup state.

### Python and platform lanes

| Lane | Interpreter source | Active runtime |
|---|---|---|
| `modly-cp311` | Exact 64-bit CPython 3.11 executable supplied by public Modly | `<extension>/venv` |
| `private-cp312` | Exact 64-bit CPython 3.12 executable supplied by the private Modly build | `<extension>/venv` |

These are **alternative host-derived lanes**, not an automatic replacement of public Modly's Python. The extension does not download an unrelated Python 3.12, mutate Modly's own environment or assume a particular 3.11 patch version. An incompatible previous extension venv is preserved as `venv.previous-*`; weights stay outside it. No simultaneous dual-runtime daemon is installed.

The installer has branches for Windows x86_64, Linux x86_64 and Linux ARM64. Linux ARM64 GB10/SM121 selects the validated **PyTorch 2.10.0+cu130** lane; the generic CUDA lanes use **PyTorch 2.10.0+cu128**. Linux requires the wheel's glibc floor (2.28). The local GB10 lane has passed native BF16 and real YuE2 inference; Windows and Linux x86_64 remain untested here.

Dependencies use binary wheels, with no automatic compilation fallback. Optional `vllm==0.19.0` and `triton==3.6.0` are Linux-only and fail clearly if a compatible wheel/dependency solution is unavailable. Optional fast execution has not been validated here. Direct dependencies are pinned; installed transitive versions are recorded by `pip freeze`, not a universal per-platform hash lock.

## Setup-managed shared weights

Setup first prepares the isolated environment, verifies pinned source files against Git blob hashes, checks dependency consistency, checks native API signatures and performs small import/kernel/audio-I/O probes. It then downloads and validates all three snapshots:

```text
<Modly models_dir>/modly-yue2-extension/
  checkpoints/
    YuE2-3B/<pinned-revision>/
    YuE2-Vae/<pinned-revision>/
    YuE2-Vae-legacy/<pinned-revision>/
  .hub/          # setup download cache, not a global user cache
  .pip-cache/    # packaging cache
```

The checkpoint payload is approximately **8.3 GB decimal** (7.26 GB MoT plus two approximately 530.5 MB VAEs), before environments, packaging/download caches and generated artifacts. Allow substantially more disk capacity for those additional files. The normal and legacy VAEs are different weights, not duplicate node downloads.

All nodes and both Python lanes reuse these same directories. A Repair of an unchanged verified snapshot avoids downloading its weights again. Corrupt recorded files are redownloaded selectively; an interrupted first installation is checked against upstream weight manifests. Readiness markers are written only after verification. Source preparation and weight provisioning use advisory locks. Updating wrapper code does not delete checkpoints.

Storage resolution prefers explicit setup/process context, then `MODELS_DIR`/`MODLY_MODELS_DIR`, then a `settings.json` whose `extensionsDir` actually identifies this extension, then a setup record bound to this same root. Ambiguous/missing storage fails closed rather than downloading into an invented folder. A moved models directory is picked up from the bound host settings. See [host gaps](docs/MODLY_GAPS.md) for the preferable small host change: explicitly pass `modelsDir` to setup and process execution.

Runtime loading uses absolute checkpoint paths, `local_files_only=True` and offline Hub/Transformers flags. It does not secretly download missing weights. Hugging Face model-side Python files may be stored for provenance, but execution uses the reviewed, pinned installed package, not `trust_remote_code=True`.

## Nodes

| Node | Input → output | Purpose |
|---|---|---|
| Generate Song | text → audio | Lyrics/style, or native request JSON; planning `full`, `melody` or `off` |
| Plan Score | text → text | Save ABC, exact ABC tokens, prefix and request as a reusable plan |
| Render ABC / Cover / Edit | text → audio | Render an external or edited ABC score with supplied style and lyrics |
| Render Exact Saved Plan | text → audio | Resume a verified plan without decoding and retokenizing its ABC |
| Plan to Semantic Tokens | text → text | Execute native semantic generation from a saved plan |
| Semantic Tokens to Latents | text → text | Native acoustic synthesis, preserving plan and semantic tokens |
| Decode Audio Latents | text → audio | FP32 VAE, default/legacy checkpoint, tiled/full decoding |
| Encode Audio | audio → text | Native acoustic VAE encoding; mean or seeded posterior sample, optional posterior arrays |
| Audio to Score / Artifacts | audio → text | Retrieve ABC, request, descriptor, verification or an agent-edit package from a YuE2 run |
| Inspect Bundle / Agent Package | text → text | Same inspection operations for intermediate bundle descriptors |
| Import Native Saved Artifacts | text → text | Import a native saved song, exact plan, or directory containing `latent.npy` |
| Batch Requests / Resume | text → text | Sequential JSON/JSONL requests with verified completed-item reuse |
| Installation Diagnostics | text → text | Python, package and storage diagnostics without loading model weights |

Most users should start with **Generate Song**, or **Plan Score → Render Exact Saved Plan**. The fine-grained stages are useful for controlled experiments and reuse but reload the native package/model in separate host subprocesses, which can increase overhead.

Text outputs carrying intermediates are JSON **bundle descriptors**, not the ABC notation itself. Use **Inspect Bundle → abc** to get raw editable ABC. Never edit `plan.json`, `prefix.npy` or `abc_tokens.npy` inside an exact saved plan. Rendering edited ABC is a new request; it is not an exact-plan resume.

For **Render ABC**, its connected text is the ABC score. Lyrics come from the node's `lyrics` parameter or `lyrics_file`, not from that same text input. For **Generate Song**, connected text or a UTF-8 text file takes priority over the fallback lyrics parameter. Select `input_mode=request-json` to pass the full request object, as in `examples/request.json`.

### Controls and artifact behavior

The adapter exposes the native ABC and semantic sampling parameters separately, seed, CFG, `full`/`melody`/`off` planning, midpoint ODE steps, memory budget, AR offload, backend, experimental FP8, VAE selection and decode mode. Defaults match `GenerationConfig`: ABC temperature 0.7/top-p 0.9/top-k 30; semantic temperature 1.0/top-p 0.95/top-k 100; 32 ODE steps. Native CFG defaults are retained through `cfg_scale=null` (1.0 for symbolic planning and 1.01 for direct mode). `ode_method=midpoint` and context 24576 are native invariants, not fictitious selectable features.

The token budgets are **not a duration-in-seconds control**. Upstream truncation flags are preserved. Experimental FP8 is restricted to the PyTorch AR path on compatible CUDA hardware, with native BF16 requirements retained; it is not a promised 8 GB mode. CPU execution is deliberately marked experimental and requires `torch-eager` without FP8.

Full songs save native FLAC plus, by default, a WAV primary artifact, request/config metadata, ABC when present, exact prefixes, semantic tokens, acoustic latents, hashes and wrapper provenance. A FLAC primary output avoids writing the additional WAV. Outputs live in unique directories under:

```text
<workspaceDir>/Workflows/YuE2/<node>-<unique-id>/
```

The native result hash manifest and the wrapper bundle manifest are separate, avoiding a circular hash dependency. Inputs are never overwritten. Arrays use non-pickle NPY; text/JSON and artifact paths are checked. The default VAE decoder and its legacy counterpart are both selectable. A standalone re-decode is an audio artifact bundle, not a falsely labeled new full native song-generation result.

For batch resume, set `resume_directory` to a previous batch directory and resubmit the same requests/settings. Completed songs are reused only after checking manifests and batch identity. An interrupted item restarts from scratch in a new directory; partial artifacts are retained. This is **completed-item resume**, not an AR KV-cache continuation or automatic benchmark best-of-N selection.

Native `save_pretrained` export is also available through `tools/export_pipeline.py`. It deliberately copies model files only when explicitly invoked, outside active `models_dir`; it is not part of normal setup or node execution.

## Actual scope and limitations

**ABC-based covers are implemented. Direct waveform → cover is not a native YuE2 checkpoint operation.** The upstream cover workflow uses an external transcription step, for example SheetSage2, and separately obtains lyrics. This package does not silently install SheetSage2/MERT/ASR into YuE2's environment, nor claim that acoustic VAE encoding performs transcription. See [coverage](docs/CAPABILITIES.md).

**Agent-edit packages are implemented; an autonomous editor is not bundled.** An external agent or a person edits ABC/style/lyrics and submits a new request. YuE2 regenerates the song; this is not local audio inpainting that preserves untouched samples. No source-separation/stem output, automatic ASR, semantic audio tokenizer, MIDI-to-ABC converter, real-time playback or benchmark evaluator is invented by the wrapper.

The inspected Modly Python runner does not implement termination of active processes. The wrapper checks signals and a `CANCEL` file in the active run directory, but **does not claim that Modly's Cancel button currently stops inference**. Proper UI cancellation needs the host change described in `docs/MODLY_GAPS.md`. Tiled standalone VAE decoding checks cancellation between tiles; full VAE decode and encode can only be interrupted at operation boundaries in this wrapper.

There is no in-process multi-output port type in the inspected host. The wrapper uses one audio/text output and durable bundle descriptors. A richer artifact viewer or ABC editor would be a host improvement, not a prerequisite for basic inference.

## Validation commands

Run lightweight tests from the repository root:

```text
python -m unittest discover -s tests -v
```

Run these **with the provisioned extension venv Python**, not the host's unrelated system interpreter:

```text
<extension-venv-python> tools/check_native_api.py
<extension-venv-python> tools/smoke.py --workspace <absolute-test-workspace>
```

The real smoke test uses deliberately short token caps and may report truncation. It validates real audio and artifacts, not musical quality. The local GB10 run produced verified 48 kHz stereo audio and exercised all thirteen nodes. Review `docs/VALIDATION.md` before installation or registry submission.

## Licensing

The newly written integration wrapper is supplied under MIT (see `LICENSE`). **YuE2 source is Apache-2.0; all three model checkpoints are CC BY-NC 4.0.** MIT on this wrapper does not remove the checkpoints' attribution/noncommercial conditions. Consult upstream terms before any commercial use or distribution. Setup preserves upstream source/model license files and notices. See `THIRD_PARTY_NOTICES.md` and `docs/SOURCES.md`.
