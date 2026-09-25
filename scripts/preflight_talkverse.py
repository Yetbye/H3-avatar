#!/usr/bin/env python3
"""Static TalkVerse/Wan2.2 compatibility checks without loading model tensors."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--lora-ckpt", type=Path, required=True)
    parser.add_argument("--wav2vec-dir", type=Path, required=True)
    parser.add_argument("--generate-py", type=Path, required=True)
    return parser.parse_args()


def require_file(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing file: {path}")


def inspect_constructor_calls(path: Path) -> list[dict[str, object]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "WanS2V_5B":
            continue
        keywords = sorted(keyword.arg for keyword in node.keywords if keyword.arg)
        calls.append(
            {
                "line": node.lineno,
                "keywords": keywords,
                "passes_lora_ckpt": "lora_ckpt" in keywords,
            }
        )
    return sorted(calls, key=lambda item: int(item["line"]))


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    warnings: list[str] = []

    index_path = args.ckpt_dir / "diffusion_pytorch_model.safetensors.index.json"
    require_file(index_path, failures)
    require_file(args.ckpt_dir / "models_t5_umt5-xxl-enc-bf16.pth", failures)
    require_file(args.ckpt_dir / "Wan2.2_VAE.pth", failures)
    require_file(args.lora_ckpt, failures)
    require_file(args.wav2vec_dir / "config.json", failures)
    require_file(args.wav2vec_dir / "preprocessor_config.json", failures)
    require_file(args.wav2vec_dir / "model.safetensors", failures)
    require_file(args.generate_py, failures)

    report: dict[str, object] = {
        "mode": "static_header_only",
        "tensor_payloads_loaded": False,
    }
    if failures:
        report.update(status="FAIL", failures=failures, warnings=warnings)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index.get("weight_map", {})
    shards = sorted(set(weight_map.values()))
    missing_shards = [name for name in shards if not (args.ckpt_dir / name).is_file()]
    if not weight_map:
        failures.append(f"empty weight_map: {index_path}")
    if missing_shards:
        failures.extend(f"missing shard: {args.ckpt_dir / name}" for name in missing_shards)

    with safe_open(args.lora_ckpt, framework="pt", device="cpu") as handle:
        lora_keys = list(handle.keys())
        lora_shapes = {key: list(handle.get_slice(key).get_shape()) for key in lora_keys}

    lora_adapter_keys = [key for key in lora_keys if ".lora_A.weight" in key or ".lora_B.weight" in key]
    target_base_keys = {
        key.replace(".lora_A.weight", ".weight").replace(".lora_B.weight", ".weight")
        for key in lora_adapter_keys
    }
    missing_targets = sorted(target_base_keys - set(weight_map))
    if not lora_adapter_keys:
        failures.append("LoRA checkpoint contains no lora_A/lora_B tensors")
    if missing_targets:
        failures.append(f"{len(missing_targets)} LoRA target tensors are absent from base index")

    constructor_calls = inspect_constructor_calls(args.generate_py)
    valid_calls = [call for call in constructor_calls if call["passes_lora_ckpt"]]
    unsafe_calls = [call for call in constructor_calls if not call["passes_lora_ckpt"]]
    if not valid_calls:
        failures.append("generate.py has no WanS2V_5B constructor that passes lora_ckpt")
    if unsafe_calls:
        lines = ", ".join(str(call["line"]) for call in unsafe_calls)
        warnings.append(
            "WanS2V_5B constructor call(s) without lora_ckpt at line(s) "
            f"{lines}; use the batch-file path until upstream wiring is fixed"
        )

    component_counts = Counter(key.split(".", 1)[0] for key in lora_keys)
    report.update(
        status="FAIL" if failures else ("PASS_WITH_WARNINGS" if warnings else "PASS"),
        base={
            "indexed_tensor_count": len(weight_map),
            "shard_count": len(shards),
            "missing_shards": missing_shards,
        },
        lora={
            "tensor_count": len(lora_keys),
            "adapter_tensor_count": len(lora_adapter_keys),
            "target_base_tensor_count": len(target_base_keys),
            "missing_target_count": len(missing_targets),
            "components": dict(sorted(component_counts.items())),
            "rank_dimensions": sorted(
                {
                    shape[0]
                    for key, shape in lora_shapes.items()
                    if ".lora_A.weight" in key and shape
                }
            ),
        },
        generate_py={"wan_s2v_5b_calls": constructor_calls},
        failures=failures,
        warnings=warnings,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
