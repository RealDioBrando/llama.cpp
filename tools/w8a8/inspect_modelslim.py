#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from safetensors import safe_open
import torch

EXPERT_RE = re.compile(r"\.mlp\.experts\.(?P<expert>\d+)?\.?(?P<projection>gate_proj|up_proj|down_proj|gate_up_proj)\.weight$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory a ModelSlim W8A8_DYNAMIC checkpoint.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_shards(model_dir: Path) -> list[Path]:
    patterns = ("quant_model_weights-*.safetensors", "quant_model_weight_w8a8_dynamic-*.safetensors")
    return sorted({path for pattern in patterns for path in model_dir.glob(pattern) if path.is_file()})


def tensor_locations(model_dir: Path, shards: list[Path]) -> dict[str, Path]:
    index_paths = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(index_paths) > 1:
        raise ValueError(f"ambiguous ModelSlim index files: {[path.name for path in index_paths]}")
    if index_paths:
        index = load_json(index_paths[0])
        weight_map = index.get("weight_map", {})
        return {name: model_dir / shard for name, shard in weight_map.items()}

    locations: dict[str, Path] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                locations[name] = shard
    return locations


def category(name: str) -> str:
    if ".mlp.experts." in name:
        return "moe_expert"
    if name.startswith("mtp."):
        return "mtp"
    if ".mlp." in name:
        return "language_mlp"
    if ".self_attn." in name:
        return "full_attention"
    return "other"


def expected_scale_shapes(weight_shape: list[int]) -> tuple[tuple[int, ...], ...]:
    if len(weight_shape) == 2:
        return ((weight_shape[0], 1), (weight_shape[0],))
    if len(weight_shape) == 3:
        return (
            (weight_shape[0], weight_shape[1]),
            (weight_shape[0], weight_shape[1], 1),
        )
    return ()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    description = load_json(model_dir / "quant_model_description.json")
    config = load_json(model_dir / "config.json")
    text_config = config.get("text_config") or config
    shards = find_shards(model_dir)
    locations = tensor_locations(model_dir, shards)

    type_counts = Counter(
        value if isinstance(value, str) else type(value).__name__
        for value in description.values()
    )
    quantized_weights = sorted(
        name for name, kind in description.items()
        if kind in ("W8A8", "W8A8_DYNAMIC") and name.endswith(".weight")
    )
    static_w8a8_weights = sorted(
        name for name, kind in description.items()
        if kind == "W8A8" and name.endswith(".weight")
    )
    expert_weights = [name for name in quantized_weights if ".mlp.experts." in name]
    categories = Counter(category(name) for name in quantized_weights)
    patterns = Counter(
        re.sub(r"\.experts\.\d+\.", ".experts.N.", name)
        for name in expert_weights
    )
    expert_ids: dict[str, set[int]] = defaultdict(set)
    expert_layers: set[int] = set()
    expert_projection_shapes: dict[str, dict[str, object]] = {}
    for name in expert_weights:
        match = EXPERT_RE.search(name)
        if not match:
            continue
        layer_match = re.search(r"\.layers\.(\d+)\.", name)
        if layer_match:
            expert_layers.add(int(layer_match.group(1)))
        if match.group("expert") is not None:
            expert_ids[match.group("projection")].add(int(match.group("expert")))

    missing: list[str] = []
    nonzero_offsets: list[str] = []
    bad_shapes: list[str] = []
    scale_ranges: dict[str, tuple[float, float]] = {}

    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(safe_open(shard, framework="pt"))
            for shard in shards
        }
        for weight_name in quantized_weights:
            scale_name = weight_name.removesuffix(".weight") + ".weight_scale"
            offset_name = weight_name.removesuffix(".weight") + ".weight_offset"
            missing.extend(name for name in (weight_name, scale_name, offset_name) if name not in locations)
            if any(name not in locations for name in (weight_name, scale_name, offset_name)):
                continue

            weight_file = handles[locations[weight_name]]
            weight_slice = weight_file.get_slice(weight_name)
            weight_shape = list(weight_slice.get_shape())
            weight_dtype = weight_slice.get_dtype()
            scale = handles[locations[scale_name]].get_tensor(scale_name)
            offset = handles[locations[offset_name]].get_tensor(offset_name)
            scale_np = scale.float().numpy() if isinstance(scale, torch.Tensor) else scale
            offset_np = offset.float().numpy() if isinstance(offset, torch.Tensor) else offset

            if len(weight_shape) not in (2, 3) or weight_dtype != "I8":
                bad_shapes.append(f"{weight_name}: shape={tuple(weight_shape)} dtype={weight_dtype}")
            if tuple(scale_np.shape) not in expected_scale_shapes(weight_shape):
                bad_shapes.append(f"{scale_name}: shape={scale_np.shape} weight_shape={tuple(weight_shape)}")
            if tuple(offset_np.shape) != tuple(scale_np.shape):
                bad_shapes.append(f"{offset_name}: shape={offset_np.shape} expected={scale_np.shape}")
            if offset_np.size and offset_np.any():
                nonzero_offsets.append(offset_name)
            if scale_np.size:
                scale_ranges[weight_name] = (float(scale_np.min()), float(scale_np.max()))

            if ".mlp.experts." in weight_name:
                match = EXPERT_RE.search(weight_name)
                projection = match.group("projection") if match else "unknown"
                expert_projection_shapes.setdefault(projection, {
                    "source_name": weight_name,
                    "weight_shape": weight_shape,
                    "scale_shape": list(scale_np.shape),
                    "offset_shape": list(offset_np.shape),
                    "layout": "separate" if len(weight_shape) == 2 else "fused",
                })

    expert_counts = {
        projection: len(ids)
        for projection, ids in sorted(expert_ids.items())
    }
    report = {
        "model_dir": str(model_dir),
        "architecture": config.get("architectures", [None])[0],
        "description_entries": len(description),
        "type_counts": dict(sorted(type_counts.items())),
        "quantized_weights": len(quantized_weights),
        "quantized_categories": dict(sorted(categories.items())),
        "static_w8a8_weights": len(static_w8a8_weights),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "hidden_size": text_config.get("hidden_size"),
        "vocab_size": text_config.get("vocab_size"),
        "expert_count_config": text_config.get("num_experts"),
        "expert_used_count_config": text_config.get("num_experts_per_tok"),
        "moe_intermediate_size": text_config.get("moe_intermediate_size"),
        "expert_weights": len(expert_weights),
        "expert_layers": sorted(expert_layers),
        "expert_projection_counts": expert_counts,
        "expert_projection_shapes": expert_projection_shapes,
        "expert_patterns": dict(sorted(patterns.items())),
        "fused_gate_up": any(".mlp.experts.gate_up_proj.weight" in name for name in expert_weights),
        "safetensors_shards": [path.name for path in shards],
        "safetensors_tensors": len(locations),
        "missing_tensors": missing,
        "bad_shapes": bad_shapes,
        "nonzero_offsets": nonzero_offsets,
        "scale_min": min((value[0] for value in scale_ranges.values()), default=None),
        "scale_max": max((value[1] for value in scale_ranges.values()), default=None),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"model: {report['model_dir']}")
    print(f"architecture: {report['architecture']}")
    print(f"description entries: {report['description_entries']}")
    print(f"type counts: {report['type_counts']}")
    print(f"quantized weights: {report['quantized_weights']}")
    print(f"quantized categories: {report['quantized_categories']}")
    print(f"model config: layers={report['num_hidden_layers']} hidden={report['hidden_size']} vocab={report['vocab_size']}")
    print(f"expert config: count={report['expert_count_config']} used={report['expert_used_count_config']}")
    print(f"expert weights: {report['expert_weights']}")
    print(f"expert layers: {report['expert_layers']}")
    print(f"expert projections: {report['expert_projection_counts']}")
    print(f"fused gate/up: {report['fused_gate_up']}")
    print(f"expert shapes: {json.dumps(report['expert_projection_shapes'], indent=2)}")
    print(f"safetensors shards: {len(shards)}")
    print(f"safetensors tensors: {report['safetensors_tensors']}")
    print(f"scale range: {report['scale_min']} .. {report['scale_max']}")
    print(f"missing tensors: {len(missing)}")
    for name in missing[:20]:
        print(f"  missing: {name}")
    print(f"bad shapes: {len(bad_shapes)}")
    for line in bad_shapes[:20]:
        print(f"  bad: {line}")
    print(f"nonzero offsets: {len(nonzero_offsets)}")
    for name in nonzero_offsets[:20]:
        print(f"  nonzero: {name}")


if __name__ == "__main__":
    main()
