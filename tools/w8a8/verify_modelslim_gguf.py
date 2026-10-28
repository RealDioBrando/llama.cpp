#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from convert_modelslim import w8a8_blocks, w8a8_expert_blocks  # noqa: E402

BLOCK_DTYPE = np.dtype([("d", "<f2"), ("qs", "i1", 32)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a direct ModelSlim W8A8 GGUF against its source shards.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("gguf_path", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def tensor_index(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "quant_model_weights.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    return {name: model_dir / shard for name, shard in index["weight_map"].items()}


def load_source_tensor(locations: dict[str, Path], name: str) -> np.ndarray:
    path = locations[name]
    with safe_open(path, framework="np") as f:
        return f.get_tensor(name)


def sample_records(records: list[dict], count: int) -> list[dict]:
    if count >= len(records):
        return records
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for record in records:
        key = (str(record.get("kind")), str(record.get("projection", "")))
        groups[key].append(record)
    selected: list[dict] = []
    seen: set[int] = set()
    for group in groups.values():
        for record in group[:1]:
            if id(record) not in seen:
                selected.append(record)
                seen.add(id(record))
    remaining = [record for record in records if id(record) not in seen]
    slots = max(count - len(selected), 0)
    if slots:
        indices = {round(i * (len(remaining) - 1) / max(slots - 1, 1)) for i in range(slots)} if len(remaining) > 1 else {0}
        selected.extend(remaining[i] for i in sorted(indices) if i < len(remaining))
    return selected[:max(count, len(groups))]


def block_view(raw: np.ndarray, shape: tuple[int, ...], blocks: int) -> np.ndarray:
    return raw.reshape((*shape, blocks, 34)).view(BLOCK_DTYPE)


def compare_raw(expected: np.ndarray, actual: np.ndarray, blocks: int) -> tuple[bool, float]:
    expected_blocks = block_view(expected, expected.shape[:-1], blocks)
    actual_blocks = block_view(actual, actual.shape[:-1], blocks)
    payload_equal = bool(np.array_equal(expected_blocks["qs"], actual_blocks["qs"]))
    scale_max_delta = float(np.max(np.abs(
        expected_blocks["d"].astype(np.float32) - actual_blocks["d"].astype(np.float32)
    )))
    return payload_equal, scale_max_delta


def verify_record(record: dict, locations: dict[str, Path], tensors: dict[str, Any]) -> dict:
    source_raw = record["source_name_raw"]
    scale_name = source_raw.removesuffix(".weight") + ".weight_scale"
    offset_name = source_raw.removesuffix(".weight") + ".weight_offset"
    weight = load_source_tensor(locations, source_raw)
    scale = load_source_tensor(locations, scale_name)
    offset = load_source_tensor(locations, offset_name)
    gguf_name = record["gguf_name"]
    actual_raw = tensors[gguf_name].data

    kind = record.get("kind", "dense")
    if kind == "expert" and record.get("layout") == "separate":
        expert_id = int(record["expert_id"])
        expected = w8a8_blocks(weight, scale)
        n_experts = int(record["stack_shape"][0])
        n, k = weight.shape
        actual_blocks = actual_raw.reshape(n_experts, n, k // 32, 34).view(BLOCK_DTYPE)
        expected_blocks = expected.reshape(n, k // 32, 34).view(BLOCK_DTYPE)
        actual_blocks = actual_blocks[expert_id]
        payload_equal = bool(np.array_equal(expected_blocks["qs"], actual_blocks["qs"]))
        scale_max_delta = float(np.max(np.abs(
            expected_blocks["d"].astype(np.float32) - actual_blocks["d"].astype(np.float32)
        )))
        shape = [n, k]
        expert_axis = expert_id
    elif kind == "expert" and record.get("layout") == "fused":
        if record.get("source_projection") == "gate_up_proj":
            start, end = record["fused_slice"]
            weight = weight[:, start:end, :]
            scale = scale[:, start:end]
        expected = w8a8_expert_blocks(weight, scale)
        payload_equal, scale_max_delta = compare_raw(expected, actual_raw, weight.shape[2] // 32)
        shape = list(weight.shape)
        expert_axis = "all"
    else:
        expected = w8a8_blocks(weight, scale)
        payload_equal, scale_max_delta = compare_raw(expected, actual_raw, weight.shape[1] // 32)
        shape = list(weight.shape)
        expert_axis = None

    offset_nonzero = int(np.count_nonzero(offset))
    return {
        "kind": kind,
        "layout": record.get("layout"),
        "source_name": source_raw,
        "gguf_name": gguf_name,
        "expert_id": record.get("expert_id"),
        "shape": shape,
        "offset_nonzero": offset_nonzero,
        "payload_equal": payload_equal,
        "scale_max_delta": scale_max_delta,
        "expert_axis": expert_axis,
        "pass": payload_equal and scale_max_delta == 0.0 and offset_nonzero == 0,
    }


def main() -> None:
    args = parse_args()
    locations = tensor_index(args.model_dir)
    records = json.loads(args.mapping.read_text(encoding="utf-8"))
    samples = sample_records(records, args.samples)
    reader = GGUFReader(args.gguf_path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    results = [verify_record(record, locations, tensors) for record in samples]
    type_counts = collections.Counter(tensor.tensor_type.name for tensor in reader.tensors)
    architecture = reader.fields["general.architecture"].contents()
    nextn_field = reader.fields.get(f"{architecture}.nextn_predict_layers")
    nextn_layers = nextn_field.contents() if nextn_field else None
    block_count_field = reader.fields.get(f"{architecture}.block_count")
    block_count = int(block_count_field.contents()) if block_count_field else None
    mtp_prefixes = []
    if block_count is not None and nextn_layers:
        first_mtp_block = block_count - int(nextn_layers)
        mtp_prefixes = [f"blk.{index}." for index in range(first_mtp_block, block_count)]
    expected_w8a8_names = {record["gguf_name"] for record in records}
    actual_w8a8_names = {
        tensor.name for tensor in reader.tensors
        if tensor.tensor_type == gguf.GGMLQuantizationType.W8A8
    }
    count_pass = expected_w8a8_names == actual_w8a8_names
    report = {
        "architecture": architecture,
        "file_type": int(reader.fields["general.file_type"].contents()),
        "tensor_count": len(reader.tensors),
        "type_counts": dict(type_counts),
        "expected_w8a8_tensors": len(expected_w8a8_names),
        "actual_w8a8_tensors": len(actual_w8a8_names),
        "w8a8_names_match": count_pass,
        "nextn_predict_layers": nextn_layers,
        "block_count": block_count,
        "vision_tensors": sum(1 for tensor in reader.tensors if "visual" in tensor.name or "vision" in tensor.name),
        "expert_tensors": sum(1 for tensor in reader.tensors if "_exps." in tensor.name),
        "mtp_tensors": sum(
            1 for tensor in reader.tensors
            if any(tensor.name.startswith(prefix) for prefix in mtp_prefixes)
        ),
        "samples": results,
        "pass": count_pass and all(result["pass"] for result in results),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"architecture: {report['architecture']}")
    print(f"file type: {report['file_type']}")
    print(f"tensor count: {report['tensor_count']}")
    print(f"type counts: {report['type_counts']}")
    print(f"W8A8 tensors: expected={report['expected_w8a8_tensors']} actual={report['actual_w8a8_tensors']}")
    print(f"nextn layers: {report['nextn_predict_layers']}")
    print(f"vision tensors: {report['vision_tensors']}")
    print(f"expert tensors: {report['expert_tensors']}")
    print(f"MTP tensors: {report['mtp_tensors']}")
    print("sample validation:")
    for result in results:
        print(f"  {result['gguf_name']}[{result['expert_id']}]: payload={result['payload_equal']} "
              f"scale_delta={result['scale_max_delta']} offset={result['offset_nonzero']} "
              f"pass={result['pass']}")
    print(f"RESULT {'pass' if report['pass'] else 'fail'}")


if __name__ == "__main__":
    main()
