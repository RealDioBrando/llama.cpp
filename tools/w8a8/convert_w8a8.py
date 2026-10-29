#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Load the repository-local gguf package when run from a source checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFWriter  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an explicit symmetric W8A8 .npz artifact to a tensor-only GGUF file."
    )
    parser.add_argument("input", type=Path, help="Input .npz containing weight and weight_scale")
    parser.add_argument("output", type=Path, help="Output GGUF path")
    parser.add_argument("--tensor-name", default="weight", help="GGUF tensor name (default: weight)")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"error: {message}")


def main() -> None:
    args = parse_args()
    with np.load(args.input) as artifact:
        require("weight" in artifact, "input must contain 'weight'")
        require("weight_scale" in artifact, "input must contain 'weight_scale'")
        weight = artifact["weight"]
        scale = artifact["weight_scale"]
        offset = artifact["weight_offset"] if "weight_offset" in artifact else None

    require(weight.dtype == np.int8, f"weight must be INT8, got {weight.dtype}")
    require(weight.ndim == 2, "weight must be 2D with shape [N,K]")
    require(scale.dtype in (np.float16, np.float32), f"weight_scale must be FP16 or FP32, got {scale.dtype}")
    require(scale.ndim == 1 and scale.shape[0] == weight.shape[0],
            "weight_scale must have shape [N] and match weight rows")
    require(np.all(np.isfinite(scale)), "weight_scale contains NaN or infinity")
    require(np.all(scale >= 0), "symmetric W8A8 scales must be nonnegative")
    require(offset is None or np.count_nonzero(offset) == 0,
            "the first implementation supports only zero weight offsets")

    n, k = weight.shape
    require(k % 32 == 0, f"K must be divisible by 32, got {k}")

    scale_f16 = scale.astype(np.float16, copy=False)
    blocks = np.empty((n, k // 32), dtype=np.dtype([("d", "<f2"), ("qs", "i1", 32)]))
    blocks["d"] = scale_f16[:, None]
    blocks["qs"] = weight.reshape(n, k // 32, 32)

    raw = np.frombuffer(blocks.tobytes(order="C"), dtype=np.uint8).reshape(n, (k // 32) * 34)
    writer = GGUFWriter(args.output, "w8a8")
    writer.add_string("general.name", args.input.stem)
    writer.add_uint32("w8a8.quantization_version", 1)
    writer.add_tensor(args.tensor_name, raw, raw_dtype=GGMLQuantizationType.W8A8)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {args.output}: tensor={args.tensor_name} shape=({n},{k}) blocks={n * (k // 32)}")


if __name__ == "__main__":
    main()
