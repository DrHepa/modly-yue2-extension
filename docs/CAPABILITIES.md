# Capability map — 2026-09-13

This map distinguishes published YuE2 **inference capabilities** from external components mentioned in its demonstrations. “Implemented” describes adapter code; it does not mean GPU acceptance testing has been completed.

| Upstream capability | Implementation / access | Remaining condition |
|---|---|---|
| Lyrics + style → complete vocal/accompaniment song | Generate Song, native `SongRequest` and pipeline stages | Real-model acceptance test |
| Full melody/chord ABC planning | `cot=full`, Plan Score / Generate Song | Native model semantics unchanged |
| Melody-only planning | `cot=melody` | Native model semantics unchanged |
| Direct generation | `cot=off`, empty symbolic prefix retained by upstream | No ABC artifact should be expected |
| External ABC / score-based cover | Render ABC / Cover / Edit | Requires a real ABC score and separately supplied lyrics |
| Human/agent score/style/lyric edits | Agent package extraction → edited native request or Render ABC | External editor/LLM is not bundled |
| Exact saved-plan reuse | Render Exact Saved Plan; native `SymbolicPlan.load()` | Modifying saved exact artifacts intentionally fails verification |
| Semantic token generation | Plan to Semantic Tokens | Same MoT weight identity enforced across wrapper stages |
| Acoustic synthesis | Semantic Tokens to Latents | Native midpoint solver; positive ODE step count |
| FP32 audio decoding | Decode Audio Latents | Full / tiled and both VAE checkpoints |
| Acoustic audio encoding | Encode Audio | Explicit 48 kHz stereo input, no hidden resampling |
| Deterministic mean / stochastic posterior | Encode Audio parameters | Optional saved mean/scale/stdev arrays; seed for sampling |
| Sampling controls, CFG, seed | Separate ABC/semantic fields | Native context and ODE method remain fixed |
| CUDA graph / eager PyTorch routes | `backend=torch` / `torch-eager` | Target GPU validation pending |
| Optional vLLM route | `install_fast=true` + `backend=vllm` | Linux CUDA; wheel compatibility and real inference pending |
| Experimental FP8 AR | `quantization=fp8`, PyTorch backend | CUDA compute capability >=8.9; native BF16 path still relevant |
| AR CPU offload / memory budget | Runtime parameters | Not a guaranteed low-VRAM preset |
| Default and benchmark legacy decoders | Setup downloads both; node decoder selector | Distinct checkpoint contents, not duplicated node storage |
| Native WAV/FLAC output and run artifacts | Full-song nodes | Primary audio + separate reproducibility artifacts |
| Batch requests and completed-item resume | Batch Requests / Resume | Exact configuration/request/weights identity; no partial AR-state resume |
| Saved native results / plans / latents | Import Native Saved Artifacts | Path, hash and array validation |
| `save_pretrained` model export | `tools/export_pipeline.py` | Explicit copies to a separate empty directory |
| Fine-grained callbacks | Native token callbacks; wrapper stage messages | Upstream NAR progress goes to stderr, which inspected Modly buffers |
| Cancellation | Signals, token/stage boundaries and local `CANCEL` sentinel | Host Python-process termination implementation is required for UI Cancel |

## Deliberately not mislabeled as YuE2 checkpoint features

**SheetSage2 audio transcription.** Upstream describes this as a separate score-extraction route for covers. It uses its own MERT dependency stack. The inspected SheetSage2 model card calls for Python 3.10/3.11, PyTorch/torchaudio 2.8.0 and shared FFmpeg libraries, while the pinned YuE2 runtime uses PyTorch 2.10.0. Mixing the stacks into the same venv, especially the private 3.12 lane, is not justified. This candidate therefore accepts external ABC rather than pretending the transcription model is already integrated. An optional transcription companion requires separate implementation/testing, not a new Modly core capability.

**ASR for lyrics.** Obtaining lyrics from source audio is not implemented by YuE2's `SongRequest`, nor by its acoustic VAE. Supply lyrics yourself or through a separately authorized transcription tool.

**Autonomous agent.** Upstream's editing workflow uses an external agent. This wrapper produces a structured edit package and accepts its edited request; it does not send your music to any cloud service or require an API key. It cannot promise an autonomous editor without a separately selected and tested agent integration.

**Benchmark selection.** A batch produces candidates; it does not implement SongBench, Q3O, PER/ASR ranking or automatically reproduce a reported best-of-8 score. Benchmark evaluation assets are not needed for ordinary inference and are not downloaded as padding.

**Other non-features.** No waveform inpainting, unchanged-sample preservation, isolated vocal/instrument stems, direct MIDI import, semantic audio tokenizer, real-time streaming or MP3 conversion is invented. The VAE's audio encoder yields acoustic latents, not an ABC score or a semantic tokenization route for covers.

Thus this delivery covers the main published **native inference API**, while the complete “audio → transcription → lyrics → cover” and autonomous agent demonstrations still need their external components. It must not be advertised as an end-to-end implementation of those entire demonstrations.
