"""Native API adapter. No substitute inference algorithms or synthetic production outputs."""
from __future__ import annotations
from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid
import numpy as np
from . import bundles
from .common import ROOT, STATE, BUNDLE_SCHEMA, EXTENSION_ID, absolute_directory, read_json, write_json, canonical_hash
from .config import normalize, native_generation_dict, request_dict, boolean
from .paths import resolve_models_root, asset_path


@dataclass
class Context:
    payload: dict
    emit: object
    externally_cancelled: object
    directory: Path | None = None
    last_progress: float = 0
    last_token_report: float = 0
    token_count: int = 0
    progress_offset: float = 0.0
    progress_span: float = 1.0

    def progress(self, percent: float, label: str):
        scaled = self.progress_offset + min(float(percent), 100) * self.progress_span
        self.last_progress = max(self.last_progress, min(scaled, 100))
        self.emit(dict(type="progress", percent=int(self.last_progress), label=label))

    def log(self, message: str):
        self.emit(dict(type="log", message=message))

    def cancelled(self) -> bool:
        return self.externally_cancelled() or (self.directory is not None and (self.directory / "CANCEL").exists())

    def check(self):
        if self.cancelled():
            raise InterruptedError("YuE2 cancelled; completed stage artifacts have been retained")

    def token(self, phase, token):
        self.check()
        self.token_count += 1
        now = time.monotonic()
        if now - self.last_token_report > 2:
            self.log(f"{phase}: {self.token_count} output tokens (not an ETA)")
            self.last_token_report = now

    def create(self, label="run") -> Path:
        workspace = absolute_directory(self.payload.get("workspaceDir", ""), "workspaceDir")
        base = workspace / "Workflows" / "YuE2"
        base.mkdir(parents=True, exist_ok=True)
        self.directory = base / (label + "-" + uuid.uuid4().hex)
        self.directory.mkdir()
        self.log(f"Artifacts: {self.directory}")
        return self.directory


def input_text(payload: dict, params: dict, *, key="lyrics") -> str:
    data = payload.get("input") or {}
    value = data.get("text")
    if value is not None and not isinstance(value, str):
        raise ValueError("input.text must be a string")
    if value:
        return value
    path = params.get(key + "_file", "")
    candidate = data.get("filePath")
    if candidate and Path(candidate).suffix.lower() in {".txt", ".abc", ".json", ".jsonl"}:
        path = candidate
    if path:
        file = Path(path).expanduser()
        if not file.is_absolute() or not file.is_file() or file.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Expected an existing absolute UTF-8 text file, at most 8 MiB")
        return file.read_text(encoding="utf-8-sig")
    return params.get(key, "")


def get_request(ctx: Context, params: dict, *, external_abc=False) -> dict:
    if external_abc:
        abc = input_text(ctx.payload, params, key="abc")
        # Do not accidentally interpret the ABC input as lyrics.
        lyrics = input_text({"input": {}}, params)
        if not abc.strip():
            raise ValueError("Provide an ABC score; an audio file is not a score")
        return request_dict(params, lyrics=lyrics, abc=abc)
    text = input_text(ctx.payload, params)
    if params.get("input_mode") == "request-json":
        return request_dict(params, lyrics="", overrides=json.loads(text))
    return request_dict(params, lyrics=text)


def local_paths(ctx: Context, params: dict) -> tuple[Path, Path]:
    root = resolve_models_root(ctx.payload)
    mot = asset_path(root, "mot")
    vae = asset_path(root, "vae_legacy" if params.get("decoder") == "legacy" else "vae")
    for folder in (mot, vae):
        if not (folder / ".complete.json").is_file():
            raise FileNotFoundError("[WEIGHTS_NOT_READY] Run YuE2 Repair/setup to provision and verify weights in Modly models_dir. Inference does not download weights.")
    return mot, vae


def make_pipeline(ctx: Context, params: dict):
    ctx.check()
    ctx.progress(3, "Resolving local YuE2 weights")
    mot, vae = local_paths(ctx, params)
    import torch
    from yue2 import YuE2Pipeline
    from yue2.protocol import GenerationConfig
    device = params.get("device", "cuda")
    backend = params.get("backend", "torch")
    quant = params.get("quantization", "none")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("[CUDA_UNAVAILABLE] This runtime cannot use CUDA. Check the installed lane and driver.")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("[BF16_UNSUPPORTED] Native YuE2 AR requires a BF16-capable CUDA GPU. No FP16 fallback is advertised.")
        if quant == "fp8" and torch.cuda.get_device_capability() < (8, 9):
            raise RuntimeError("[FP8_UNSUPPORTED] Experimental FP8 AR requires compute capability >=8.9")
    else:
        if backend != "torch-eager" or quant != "none":
            raise ValueError("CPU experimental execution requires torch-eager and quantization=none")
        ctx.log("CPU generation is experimental and may be impractically slow; it has not been validated here.")
    if backend == "vllm":
        if platform.system() != "Linux":
            raise RuntimeError("The optional vLLM lane is Linux-only")
        try:
            importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("Run setup with install_fast=true before selecting vllm") from exc
    ctx.progress(5, "Verifying checkpoint hashes (may read several GiB)")
    pipe = YuE2Pipeline.from_pretrained(str(mot), vae=str(vae), local_files_only=True,
        device=device, memory_budget_gib=params.get("memory_budget_gib", 24), backend=backend,
        quantization=quant, offload_ar=boolean(params.get("offload_ar", "false")),
        vae_core_frames=params.get("vae_core_frames", 0) or None,
        generation_config=GenerationConfig.from_dict(native_generation_dict(params)),
        verify_hashes=True, progress=True)
    return pipe


def bundle_input(ctx: Context, params: dict):
    data = ctx.payload.get("input") or {}
    text = data.get("text") or ""
    return bundles.locate(text, params.get("bundle_path") or data.get("filePath") or "")


def metadata(pipe, params: dict, **extra) -> dict:
    return dict(native_weights=pipe.weights, runtime_params=params, **extra)


def compatible_model(data: dict, pipe):
    previous = data.get("metadata", {}).get("native_weights", {}).get("mot")
    if previous is not None and previous != pipe.weights["mot"]:
        raise ValueError("This stage was produced with different MoT weights; use the matching checkpoint")


def result_text(directory: Path) -> dict:
    return {"text": json.dumps(bundles.descriptor(directory), ensure_ascii=False)}


def save_song(ctx: Context, pipe, plan, semantic, latents, params: dict, directory: Path) -> dict:
    from yue2.pipeline import SongResult
    from yue2.storage import identity
    ctx.check()
    ctx.progress(85, "Decoding 48 kHz stereo audio")
    started = time.perf_counter()
    audio = pipe.decode(latents, full=boolean(params.get("full_decode", "false")))
    config = pipe.effective_config(plan.request)
    # Record the actual full/tiled override, not a misleading native default.
    if boolean(params.get("full_decode", "false")):
        config["vae_decode"] = "full"
    timing = {"abc": plan.timing, "semantic": semantic.timing,
              "vae_seconds": time.perf_counter() - started, "load": dict(pipe.load_timing)}
    request_id = identity({"request": plan.request.to_dict(), "config": config, "weights": pipe.weights})
    song = SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, request_id)
    ctx.check()
    ctx.progress(95, "Saving audio and reproducible artifacts")
    # Native collect_hashes must not record a wrapper manifest that is about to change.
    (directory / "bundle.json").unlink(missing_ok=True)
    song.save_artifacts(directory)
    # Extra semantic metadata supports exact split-stage reconstruction.
    write_json(directory / "semantic_info.json", {"timing": semantic.timing, "truncated": semantic.truncated})
    output = directory / ("audio." + params.get("audio_format", "wav"))
    if output.suffix == ".wav":
        song.save(output)
    bundles.seal(directory, "song", metadata(pipe, params, request_identity=request_id))
    if plan.truncated or semantic.truncated:
        ctx.log("WARNING: upstream marked ABC and/or semantic generation as truncated; inspect result.json")
    return {"filePath": str(output.resolve())}


def render_from_plan(ctx: Context, pipe, plan, params: dict, directory: Path) -> dict:
    ctx.check()
    plan.save(directory)
    ctx.progress(30, "Generating semantic music tokens")
    semantic = pipe.generate_semantic(plan, cancelled=ctx.cancelled, on_token=ctx.token)
    np.save(directory / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32), allow_pickle=False)
    write_json(directory / "semantic_info.json", {"timing": semantic.timing, "truncated": semantic.truncated})
    bundles.seal(directory, "semantic", metadata(pipe, params))
    ctx.progress(65, "Synthesizing acoustic latents")
    latents = pipe.synthesize(semantic, cancelled=ctx.cancelled)
    np.save(directory / "latent.npy", np.asarray(latents, dtype=np.float32), allow_pickle=False)
    bundles.seal(directory, "latents", metadata(pipe, params))
    return save_song(ctx, pipe, plan, semantic, latents, params, directory)


def execute_generation(node: str, ctx: Context, params: dict) -> dict:
    from yue2.pipeline import SymbolicPlan, SemanticResult
    from yue2.protocol import SongRequest
    directory = ctx.create(node)
    # Validate user request BEFORE loading 7 GB of model tensors.
    request = SongRequest(**get_request(ctx, params, external_abc=node == "render-abc")) if node in {"generate", "plan", "render-abc"} else None
    source = data = None
    if request is None:
        source, data = bundle_input(ctx, params)
        if node in {"render-plan", "semantic"} and data["kind"] not in {"plan", "semantic", "latents", "song"}:
            raise ValueError("This node needs an exact saved symbolic plan")
        if node == "synthesize" and data["kind"] not in {"semantic", "latents", "song"}:
            raise ValueError("This node needs saved semantic tokens")
    with make_pipeline(ctx, params) as pipe:
        if source is not None:
            compatible_model(data, pipe)
            plan = SymbolicPlan.load(source)
        else:
            ctx.progress(12, "Planning symbolic score")
            plan = pipe.plan(request=request, cancelled=ctx.cancelled, on_token=ctx.token)
        plan.save(directory)
        bundles.seal(directory, "plan", metadata(pipe, params))
        if node == "plan":
            return result_text(directory)
        if node == "synthesize":
            tokens = bundles.array(source / "semantic.npy", "semantic").tolist()
            info = read_json(source / "semantic_info.json")
            semantic = SemanticResult(plan, tokens, info["timing"], bool(info["truncated"]))
            ctx.progress(30, "Synthesizing saved semantic tokens")
            latents = pipe.synthesize(semantic, cancelled=ctx.cancelled)
            bundles.copy_stage(source, directory, bundles.SEMANTIC_FILES)
            np.save(directory / "latent.npy", np.asarray(latents, dtype=np.float32), allow_pickle=False)
            bundles.seal(directory, "latents", metadata(pipe, params))
            return result_text(directory)
        if node == "semantic":
            ctx.progress(30, "Generating semantic music tokens")
            semantic = pipe.generate_semantic(plan, cancelled=ctx.cancelled, on_token=ctx.token)
            np.save(directory / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32), allow_pickle=False)
            write_json(directory / "semantic_info.json", {"timing": semantic.timing, "truncated": semantic.truncated})
            bundles.seal(directory, "semantic", metadata(pipe, params))
            return result_text(directory)
        return render_from_plan(ctx, pipe, plan, params, directory)


def vae_model(ctx, params, *, encoder=False):
    import torch
    from yue2.modeling_vae import YuE2VAE
    from yue2.storage import model_identity
    _, path = local_paths(ctx, params)
    device = params.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for this VAE runtime")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    ctx.progress(10, "Verifying and loading local FP32 VAE")
    weight_identity = model_identity(path, verify=True)
    model = YuE2VAE.from_pretrained(path, decoder_only=not encoder, device=device, local_files_only=True)
    return model, weight_identity


def encode(ctx, params):
    import torch
    import soundfile as sf
    path = (ctx.payload.get("input") or {}).get("filePath") or params["audio_path"]
    file = Path(path).expanduser()
    if not path or not file.is_absolute() or not file.is_file():
        raise ValueError("Connect an audio artifact or provide an absolute WAV/FLAC path")
    info = sf.info(str(file))
    if info.samplerate != 48000 or info.channels != 2:
        raise ValueError("Native VAE encode requires 48 kHz stereo. Resample/upmix explicitly before this node; no implicit audio changes are made.")
    if info.frames < 1920:
        raise ValueError("Audio must contain at least 1920 samples")
    if info.frames * info.channels * 4 > 1024**3:
        raise ValueError("Input exceeds the 1 GiB decoded-audio safety limit")
    directory = ctx.create("encode")
    audio, _ = sf.read(str(file), dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    model, weights = vae_model(ctx, params, encoder=True)
    try:
        ctx.progress(30, "Encoding waveform to acoustic latents (not a score)")
        ctx.check()
        generator = torch.Generator(device=params["device"]).manual_seed(params["seed"])
        want_info = boolean(params["save_posterior_info"])
        encoded = model.encode(torch.from_numpy(audio.T.copy()).unsqueeze(0),
                               sample=boolean(params["posterior_sample"]), generator=generator, return_info=want_info)
        latent, extra = encoded if want_info else (encoded, {})
        np.save(directory / "latent.npy", latent.detach().float().cpu().numpy(), allow_pickle=False)
        for name, value in extra.items():
            np.save(directory / ("posterior_" + name + ".npy"), value.detach().float().cpu().numpy(), allow_pickle=False)
        bundles.seal(directory, "encoded", {"vae_weights": weights, "runtime_params": params,
                     "source_frames": info.frames, "sample_rate": 48000, "channels": 2,
                     "warning": "Acoustic latents are not semantic tokens or ABC transcription"})
        ctx.check()
        return result_text(directory)
    finally:
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def decode(ctx, params):
    import torch
    import soundfile as sf
    source, data = bundle_input(ctx, params)
    if data["kind"] not in {"latents", "encoded", "song", "decoded"}:
        raise ValueError("This node needs an acoustic latent bundle")
    latents = bundles.array(source / "latent.npy", "latent")
    directory = ctx.create("decode")
    model, weights = vae_model(ctx, params)
    try:
        z = torch.as_tensor(latents, dtype=torch.float32)
        if z.ndim == 2:
            z = z.T.unsqueeze(0)
        ctx.check()
        ctx.progress(30, "Decoding stereo waveform")
        if boolean(params["full_decode"]):
            audio = model.decode(z).cpu()
        else:
            def report(done, total):
                ctx.check()
                ctx.progress(30 + 55 * done / total, "Decoding audio tiles")
            frames = params["vae_core_frames"] or (512 if params.get("memory_budget_gib", 24) <= 12 else 1024)
            audio = model.decode_tiled(z, core_frames=frames, halo_frames=16, output_device="cpu", on_progress=report)
        if not torch.isfinite(audio).all():
            raise ValueError("VAE produced non-finite audio")
        output = directory / ("audio." + params["audio_format"])
        sf.write(str(output), audio[0].float().clamp(-1,1).T.numpy(), 48000,
                 subtype="PCM_24" if output.suffix == ".flac" else "FLOAT")
        shutil.copyfile(source / "latent.npy", directory / "latent.npy")
        bundles.seal(directory, "decoded", {"vae_weights": weights, "runtime_params": params,
                     "source": bundles.descriptor(source), "sample_rate": 48000})
        ctx.check()
        return {"filePath": str(output.resolve())}
    finally:
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def inspect_bundle(ctx, params):
    source, data = bundle_input(ctx, params)
    choice = params["extract"]
    if choice == "descriptor":
        return result_text(source)
    if choice == "metadata":
        value = data
    elif choice == "verify":
        value = {"verified": True, "kind": data["kind"], "file_count": len(data["artifacts"]), "directory": str(source)}
    elif choice == "abc":
        if "score.abc" not in data["artifacts"]:
            raise ValueError("No ABC score in this bundle (direct mode deliberately has none)")
        return {"text": (source / "score.abc").read_text(encoding="utf-8")}
    elif choice == "request":
        file = source / "request.json"
        value = read_json(file) if file.is_file() else read_json(source / "plan.json")["request"]
    else:
        file = source / "request.json"
        request = read_json(file) if file.is_file() else read_json(source / "plan.json")["request"]
        value = {"instruction": "Edit style, lyrics and/or the ABC score. Submit a NEW request to Generate Song (request-json) or Render ABC. Do not mutate exact saved-plan artifacts. The full song is regenerated; this is not waveform inpainting.",
                 "request": {**request, "abc": (source / "score.abc").read_text(encoding="utf-8") if (source / "score.abc").is_file() else None},
                 "source_bundle": bundles.descriptor(source)}
    return {"text": json.dumps(value, ensure_ascii=False, indent=2)}


def import_artifacts(ctx, params):
    from yue2.pipeline import SymbolicPlan
    from yue2.storage import verify_result
    raw = input_text(ctx.payload, {}, key="bundle_path").strip() or params["bundle_path"]
    source = absolute_directory(raw, "native artifact directory")
    directory = ctx.create("import")
    if (source / "bundle.json").is_file():
        data = bundles.verify(source)
        for name in data["artifacts"]:
            src = source / name
            dst = directory / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        bundles.seal(directory, data["kind"], data["metadata"])
        return result_text(directory)
    if (source / "result.json").is_file():
        native = verify_result(source)
        SymbolicPlan.load(source)
        # verify_result has checked every relative source path before copying.
        for name in native["artifacts"]:
            dst = directory / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, dst)
        shutil.copyfile(source / "result.json", directory / "result.json")
        write_json(directory / "semantic_info.json", {"timing": native.get("timing", {}).get("semantic", {}), "truncated": native["truncated"]["semantic"]})
        bundles.seal(directory, "song", {"native_weights": native["weights"], "imported": True})
    elif (source / "plan_manifest.json").is_file():
        plan = SymbolicPlan.load(source)
        plan.save(directory)
        bundles.seal(directory, "plan", {"imported": True})
    elif (source / "latent.npy").is_file():
        array = bundles.array(source / "latent.npy", "latent")
        np.save(directory / "latent.npy", array, allow_pickle=False)
        bundles.seal(directory, "latents", {"imported": True, "provenance": "User-supplied latent array; no upstream weight identity available"})
    else:
        raise ValueError("Expected native saved song, exact plan, or latent.npy")
    return result_text(directory)


def batch(ctx, params):
    from yue2.protocol import SongRequest
    from yue2.storage import verify_result
    text = input_text(ctx.payload, params, key="requests")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("Provide 1–100 SongRequest objects as JSON array or JSONL")
    requests = [SongRequest(**request_dict(params, lyrics="", overrides=row)) for row in rows]
    ids = [r.id for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("Batch song ids must be unique")
    if params["resume_directory"]:
        directory = absolute_directory(params["resume_directory"], "resume_directory")
        if not (directory / "batch.json").is_file():
            raise ValueError("Resume requires an existing YuE2 batch.json")
        ctx.directory = directory
    else:
        directory = ctx.create("batch")
    with make_pipeline(ctx, params) as pipe:
        identity = canonical_hash({"requests": [r.to_dict() for r in requests], "generation": native_generation_dict(params),
                                   "runtime": {k:v for k,v in params.items() if k not in {"resume_directory", "requests_file"}}, "weights":pipe.weights,
                                   "native_config":pipe.effective_config(requests[0])})
        journal = directory / "batch.json"
        if journal.is_file():
            state = read_json(journal)
            if state["identity"] != identity:
                raise ValueError("Batch request/config/weight identity changed; start a new batch")
        else:
            state = {"identity":identity, "status":"running", "items":[]}
            write_json(journal, state)
        items = {x["id"]:x for x in state["items"]}
        for index, request in enumerate(requests):
            ctx.progress_offset = 5 + 90 * index / len(requests)
            ctx.progress_span = .9 / len(requests)
            ctx.check()
            target = directory / request.id
            if request.id in items and items[request.id].get("status") == "complete":
                bundles.verify(target)
                verify_result(target)
                ctx.log(f"Verified and reused {request.id}")
                continue
            if target.exists():
                # Keep interrupted artifacts, never overwrite a partially written run.
                target.rename(directory / (request.id + ".partial-" + uuid.uuid4().hex[:8]))
            target.mkdir()
            ctx.log(f"Batch item {index+1}/{len(requests)}: {request.id}")
            plan = pipe.plan(request=request, cancelled=ctx.cancelled, on_token=ctx.token)
            result = render_from_plan(ctx, pipe, plan, params, target)
            items[request.id] = dict(id=request.id, status="complete", **result, bundle=bundles.descriptor(target))
            state["items"] = list(items.values())
            write_json(journal, state)
        state["status"] = "complete"
        write_json(journal, state)
    ctx.progress_offset, ctx.progress_span = 0.0, 1.0
    # The batch index references individually verified immutable song bundles.
    return {"text":json.dumps({"batch_directory":str(directory), "index":str(journal), "items":state["items"]},ensure_ascii=False)}


def diagnostics(ctx):
    report = {"extension":EXTENSION_ID, "python":sys.version, "executable":sys.executable,
              "platform":platform.platform(), "machine":platform.machine(),
              "native_runtime_loaded":False, "inference_executed_by_diagnostics":False}
    if STATE.is_file():
        report["setup"] = read_json(STATE)
    for name in ("torch","transformers","yue2-infer","vllm"):
        try:
            report[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report[name] = None
    try:
        report["models_root"] = str(resolve_models_root(ctx.payload))
    except ValueError as exc:
        report["storage_error"] = str(exc)
    return {"text":json.dumps(report,ensure_ascii=False,indent=2)}


def execute(payload: dict, emit, cancelled=lambda: False) -> dict:
    input_data = payload.get("input") or {}
    if not isinstance(input_data, dict):
        raise ValueError("input must be an object")
    node = input_data.get("nodeId") or payload.get("nodeId")
    params = normalize(node, payload.get("params") or {})
    ctx = Context(payload, emit, cancelled)
    ctx.check()
    generation_nodes = {"generate","plan","render-abc","render-plan","semantic","synthesize"}
    if node in generation_nodes:
        result = execute_generation(node, ctx, params)
    else:
        handlers = {"decode":decode, "encode":encode, "inspect-audio":inspect_bundle,
                    "inspect-bundle":inspect_bundle, "import-artifacts":import_artifacts, "batch":batch,
                    "diagnostics":lambda c,p:diagnostics(c)}
        result = handlers[node](ctx, params)
    ctx.check()
    ctx.progress(100, "YuE2 complete")
    return result
