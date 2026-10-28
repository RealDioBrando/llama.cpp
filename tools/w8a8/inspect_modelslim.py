#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory a ModelSlim W8A8_DYNAMIC checkpoint.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_shards(model_dir: Path) -> list[Path]:
    return sorted(path for path in model_dir.glob("quant_model_weights-*.safetensors") if path.is_file())


def tensor_locations(model_dir: Path, shards: list[Path]) -> dict[str, Path]:
    index_path = model_dir / "quant_model_weights.safetensors.index.json"
    if index_path.is_file():
        index = load_json(index_path)
        weight_map = index.get("weight_map", {})
        return {name: model_dir / shard for name, shard in weight_map.items()}

    locations: dict[str, Path] = {}
    for shard in shards:
        with safe_open(shard, framework="np") as handle:
            for name in handle.keys():
                locations[name] = shard
    return locations


def category(name: str) -> str:
    if name.startswith("mtp."):
        return "mtp"
    if ".mlp." in name:
        return "language_mlp"
    if ".self_attn." in name:
        return "full_attention"
    return "other"


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    description = load_json(model_dir / "quant_model_description.json")
    config = load_json(model_dir / "config.json")
    shards = find_shards(model_dir)
    locations = tensor_locations(model_dir, shards)

    type_counts = Counter(
        value if isinstance(value, str) else type(value).__name__
        for value in description.values()
    )
    quantized_weights = sorted(name for name, kind in description.items()
                               if kind == "W8A8_DYNAMIC" and name.endswith(".weight"))
    categories = Counter(category(name) for name in quantized_weights)
    missing: list[str] = []
    nonzero_offsets: list[str] = []
    bad_shapes: list[str] = []
    scale_ranges: dict[str, tuple[float, float]] = {}

    for weight_name in quantized_weights:
        scale_name = weight_name.removesuffix(".weight") + ".weight_scale"
        offset_name = weight_name.removesuffix(".weight") + ".weight_offset"
        for name in (weight_name, scale_name, offset_name):
            if name not in locations:
                missing.append(name)

        weight_loc = locations.get(weight_name)
        scale_loc = locations.get(scale_name)
        offset_loc = locations.get(offset_name)
        if weight_loc is None or scale_loc is None or offset_loc is None:
            continue

        with safe_open(weight_loc, framework="np") as weight_file:
            weight_slice = weight_file.get_slice(weight_name)
            weight_shape = weight_slice.get_shape()
            weight_dtype = weight_slice.get_dtype()
        with safe_open(scale_loc, framework="np") as scale_file:
            scale = scale_file.get_tensor(scale_name)
        with safe_open(offset_loc, framework="np") as offset_file:
            offset = offset_file.get_tensor(offset_name)

        if len(weight_shape) != 2 or weight_dtype != "I8":
            bad_shapes.append(f"{weight_name}: shape={weight_shape} dtype={weight_dtype}")
        if scale.ndim != 2 or scale.shape != (weight_shape[0], 1):
            bad_shapes.append(f"{scale_name}: shape={scale.shape} expected=({weight_shape[0]}, 1)")
        if offset.ndim != 2 or offset.shape != (weight_shape[0], 1):
            bad_shapes.append(f"{offset_name}: shape={offset.shape} expected=({weight_shape[0]}, 1)")
        if offset.size and offset.any():
            nonzero_offsets.append(offset_name)
        if scale.size:
            scale_ranges[weight_name] = (float(scale.min()), float(scale.max()))

    report = {
        "model_dir": str(model_dir),
        "architecture": config.get("architectures", [None])[0],
        "description_entries": len(description),
        "type_counts": dict(sorted(type_counts.items())),
        "quantized_weights": len(quantized_weights),
        "quantized_categories": dict(sorted(categories.items())),
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
