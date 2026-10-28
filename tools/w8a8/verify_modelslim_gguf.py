#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from convert_modelslim import w8a8_blocks  # noqa: E402

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
    required = {0, len(records) - 1}
    required.update(round(i * (len(records) - 1) / (count - 1)) for i in range(count))
    return [records[i] for i in sorted(required)]


def main() -> None:
    args = parse_args()
    locations = tensor_index(args.model_dir)
    records = json.loads(args.mapping.read_text(encoding="utf-8"))
    samples = sample_records(records, args.samples)
    reader = GGUFReader(args.gguf_path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    results: list[dict] = []
    for record in samples:
        source_raw = record["source_name_raw"]
        weight_name = source_raw
        scale_name = source_raw.removesuffix(".weight") + ".weight_scale"
        offset_name = source_raw.removesuffix(".weight") + ".weight_offset"

        weight = load_source_tensor(locations, weight_name)
        scale = load_source_tensor(locations, scale_name)
        offset = load_source_tensor(locations, offset_name)
        expected_raw = w8a8_blocks(weight, scale)
        gguf_name = record["gguf_name"]
        actual_raw = tensors[gguf_name].data

        n, k = weight.shape
        expected_blocks = expected_raw.reshape(n, k // 32, 34).view(BLOCK_DTYPE)
        actual_blocks = actual_raw.reshape(n, k // 32, 34).view(BLOCK_DTYPE)
        payload_equal = bool(np.array_equal(expected_blocks["qs"], actual_blocks["qs"]))
        scale_max_delta = float(np.max(np.abs(
            expected_blocks["d"].astype(np.float32) - actual_blocks["d"].astype(np.float32)
        )))
        results.append({
            "source_name": weight_name,
            "gguf_name": gguf_name,
            "shape": [n, k],
            "offset_nonzero": int(np.count_nonzero(offset)),
            "payload_equal": payload_equal,
            "scale_max_delta": scale_max_delta,
            "pass": payload_equal and scale_max_delta == 0.0 and not np.any(offset),
        })

    type_counts = collections.Counter(tensor.tensor_type.name for tensor in reader.tensors)
    report = {
        "architecture": reader.fields["general.architecture"].contents(),
        "file_type": int(reader.fields["general.file_type"].contents()),
        "tensor_count": len(reader.tensors),
        "type_counts": dict(type_counts),
        "nextn_predict_layers": reader.fields.get("qwen35.nextn_predict_layers").contents()
            if reader.fields.get("qwen35.nextn_predict_layers") else None,
        "vision_tensors": sum(1 for tensor in reader.tensors if "visual" in tensor.name or "vision" in tensor.name),
        "mtp_tensors": sum(1 for tensor in reader.tensors if tensor.name.startswith("blk.32.")),
        "samples": results,
        "pass": all(result["pass"] for result in results),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"architecture: {report['architecture']}")
    print(f"file type: {report['file_type']}")
    print(f"tensor count: {report['tensor_count']}")
    print(f"type counts: {report['type_counts']}")
    print(f"nextn layers: {report['nextn_predict_layers']}")
    print(f"vision tensors: {report['vision_tensors']}")
    print(f"MTP tensors: {report['mtp_tensors']}")
    print("sample validation:")
    for result in results:
        print(f"  {result['gguf_name']}: payload={result['payload_equal']} "
              f"scale_delta={result['scale_max_delta']} offset={result['offset_nonzero']} "
              f"pass={result['pass']}")
    print(f"RESULT {'pass' if report['pass'] else 'fail'}")


if __name__ == "__main__":
    main()
