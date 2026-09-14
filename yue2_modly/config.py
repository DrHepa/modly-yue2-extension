"""Schema-driven validation; the manifest is the single source of UI defaults."""
from __future__ import annotations
import json
import math
from .common import ROOT, file_id


def manifest() -> dict:
    return json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))


def node_schema(node_id: str) -> dict:
    for node in manifest()["nodes"]:
        if node["id"] == node_id:
            return node
    raise ValueError(f"Unknown YuE2 nodeId: {node_id}")


def boolean(value) -> bool:
    if value is True or value in ("true", "1", 1):
        return True
    if value is False or value in ("false", "0", 0):
        return False
    raise ValueError("Expected true or false")


def normalize(node_id: str, values: dict) -> dict:
    if not isinstance(values, dict):
        raise TypeError("params must be an object")
    fields = node_schema(node_id)["params_schema"]
    known = {f["id"] for f in fields}
    # Hosts can add bookkeeping fields; never forward unknown fields to native APIs.
    result = {}
    for field in fields:
        key, kind = field["id"], field["type"]
        value = values.get(key, field["default"])
        if kind == "select":
            value = str(value).lower() if isinstance(value, bool) else str(value)
            if value not in {str(x["value"]) for x in field["options"]}:
                raise ValueError(f"Invalid {key}: {value}")
        elif kind in ("int", "float"):
            if isinstance(value, bool):
                raise ValueError(f"{key} must be a number, not boolean")
            if kind == "int":
                number = float(value)
                if not math.isfinite(number) or not number.is_integer():
                    raise ValueError(f"{key} must be a finite integer")
                value = int(number)
            else:
                value = float(value)
            if not math.isfinite(value) or value < field.get("min", -math.inf) or value > field.get("max", math.inf):
                raise ValueError(f"{key} is outside its supported range")
        elif kind == "string":
            if not isinstance(value, str):
                raise ValueError(f"{key} must be text")
        result[key] = value
    for phase in ("abc", "semantic"):
        if f"{phase}_min_tokens" in result and result[f"{phase}_min_tokens"] > result[f"{phase}_max_tokens"]:
            raise ValueError(f"{phase} min_tokens exceeds max_tokens")
    if result.get("quantization") == "fp8" and result.get("backend") == "vllm":
        raise ValueError("YuE2's experimental FP8 adapter is a PyTorch AR feature; select torch or torch-eager")
    return result


def native_generation_dict(params: dict) -> dict:
    result = {"ode_steps": params.get("ode_steps", 32)}
    names = ("temperature", "top_p", "top_k", "repetition_penalty", "penalty_window", "min_tokens", "max_tokens")
    for phase in ("abc", "semantic"):
        result[phase] = {name: params[f"{phase}_{name}"] for name in names if f"{phase}_{name}" in params}
    return result


def request_dict(params: dict, *, lyrics: str, abc: str | None = None, overrides=None) -> dict:
    result = dict(style=params.get("style", ""), lyrics=lyrics, cot=params.get("cot", "full"),
                  seed=params.get("seed", 831001), id=file_id(params.get("song_id", "song")),
                  abc=abc, cfg_scale=None if params.get("cfg_mode", "native") == "native" else params.get("cfg_scale", 1.0))
    if overrides:
        allowed = {"style", "lyrics", "cot", "seed", "id", "abc", "cfg_scale"}
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ValueError("Request JSON contains unsupported fields")
        result.update(overrides)
    file_id(result["id"])
    if result["cot"] not in {"off", "melody", "full"}:
        raise ValueError("cot must be off, melody or full")
    if result["abc"] is not None and (result["cot"] == "off" or not isinstance(result["abc"], str) or not result["abc"].strip()):
        raise ValueError("External ABC needs nonempty text and cot=melody/full")
    if not all(isinstance(result[x], str) for x in ("style", "lyrics")):
        raise ValueError("style and lyrics must be strings")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    cfg = result["cfg_scale"]
    if cfg is not None and (isinstance(cfg, bool) or not isinstance(cfg, (float, int)) or not math.isfinite(cfg) or not 0 <= cfg <= 20):
        raise ValueError("cfg_scale must be null or a finite number in [0,20]")
    return result
