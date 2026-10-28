#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402
from conversion.base import LazyTorchTensor, ModelBase, ModelType, get_model_architecture, logger  # noqa: E402
from conversion.qwen import Qwen3_5TextModel  # noqa: E402
from conversion.qwen3vl import Qwen3VLVisionModel  # noqa: E402

W8A8_TYPE = gguf.GGMLQuantizationType.W8A8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a ModelSlim W8A8_DYNAMIC Qwen3.5 checkpoint directly to GGUF.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate and print the tensor mapping without writing GGUF")
    parser.add_argument("--mapping-out", type=Path, help="write the mapping report as JSON")
    parser.add_argument("--mmproj-output", type=Path, help="optionally write the vision projector as a separate mmproj GGUF")
    parser.add_argument("--mmproj-only", action="store_true", help="write only the vision projector")
    parser.add_argument("--no-lazy", action="store_true", help="load tensors eagerly while writing")
    return parser.parse_args()


def w8a8_blocks(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    require(weight.dtype == np.int8, f"weight must be INT8, got {weight.dtype}")
    require(weight.ndim == 2, f"weight must be 2D, got {weight.shape}")
    require(scale.dtype in (np.float16, np.float32), f"weight_scale must be FP16/FP32, got {scale.dtype}")
    require(scale.ndim == 2 and scale.shape == (weight.shape[0], 1),
            f"weight_scale must have shape ({weight.shape[0]}, 1), got {scale.shape}")
    require(weight.shape[1] % 32 == 0, f"K must be divisible by 32, got {weight.shape[1]}")

    n, k = weight.shape
    scale_f16 = scale.reshape(-1).astype(np.float16, copy=False)
    blocks = np.empty((n, k // 32), dtype=np.dtype([("d", "<f2"), ("qs", "i1", 32)]))
    blocks["d"] = scale_f16[:, None]
    blocks["qs"] = weight.reshape(n, k // 32, 32)
    return np.frombuffer(blocks.tobytes(order="C"), dtype=np.uint8).reshape(n, (k // 32) * 34)


def index_modelslim_tensors(model: ModelBase, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], torch.Tensor]]:
    if remote_hf_model_id is not None:
        raise NotImplementedError("remote ModelSlim indexing is not supported")

    shards = sorted(model.dir_model.glob("quant_model_weights-*.safetensors"))
    require(len(shards) > 0, f"no quant_model_weights-*.safetensors files in {model.dir_model}")

    tensors: dict[str, Callable[[], torch.Tensor]] = {}
    index_path = model.dir_model / "quant_model_weights.safetensors.index.json"
    index = None
    if index_path.is_file():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    expected_names = set(index.get("weight_map", {})) if index else set()
    seen_names: set[str] = set()

    for shard in shards:
        logger.info(f"gguf: indexing ModelSlim part '{shard.name}'")
        with gguf.utility.SafetensorsLocal(shard) as model_part:
            for name, data in model_part.items():
                seen_names.add(name)
                if model.lazy:
                    gen = lambda data=data: LazyTorchTensor.from_local_tensor(data)  # noqa: E731
                else:
                    dtype = LazyTorchTensor._dtype_str_map[data.dtype]
                    gen = lambda data=data, dtype=dtype: torch.from_numpy(data.mmap_bytes()).view(dtype).reshape(data.shape)  # noqa: E731
                titem = model.filter_tensors((name, gen))
                if titem is not None:
                    normalized_name, normalized_gen = titem
                    tensors[normalized_name] = normalized_gen

    if expected_names:
        missing = sorted(expected_names - seen_names)
        extra = sorted(seen_names - expected_names)
        require(not missing, f"missing tensors from ModelSlim index: {missing[:10]}")
        require(not extra, f"unexpected ModelSlim tensors: {extra[:10]}")
    return tensors


class ModelslimQwen35Model(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modelslim_description = self._load_description(self.dir_model)
        self.w8a8_mapping: list[dict[str, object]] = []

    @staticmethod
    def _load_description(model_dir: Path) -> dict[str, str]:
        path = model_dir / "quant_model_description.json"
        require(path.is_file(), f"missing {path}")
        with open(path, "r", encoding="utf-8") as f:
            description = json.load(f)
        require(isinstance(description, dict), "quant_model_description.json must contain an object")
        return description

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], torch.Tensor]]:
        hparams = {**self.hparams, **self.hparams.get("text_config", {})}
        key = next((k for k in ["n_layers", "num_hidden_layers", "n_layer", "num_layers"] if k in hparams), None)
        type(self)._original_block_count = hparams.get(key)
        type(self).opt_num_mtp_layers = 0
        return index_modelslim_tensors(self, remote_hf_model_id)

    def _normalize_description_name(self, name: str) -> str | None:
        titem = self.filter_tensors((name, lambda: None))
        return titem[0] if titem is not None else None

    def _quantized_weight_names(self) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for original_name, kind in self.modelslim_description.items():
            if kind != "W8A8_DYNAMIC" or not original_name.endswith(".weight"):
                continue
            normalized = self._normalize_description_name(original_name)
            if normalized is None:
                continue
            result.append((original_name, normalized))
        return result

    def collect_mapping(self, load_payload: bool = True) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for source_name, weight_name in self._quantized_weight_names():
            scale_name = weight_name.removesuffix(".weight") + ".weight_scale"
            offset_name = weight_name.removesuffix(".weight") + ".weight_offset"
            require(weight_name in self.model_tensors, f"missing normalized weight tensor: {weight_name}")
            require(scale_name in self.model_tensors, f"missing scale tensor for {weight_name}")
            require(offset_name in self.model_tensors, f"missing offset tensor for {weight_name}")

            weight_lazy = self.model_tensors[weight_name]()
            scale_lazy = self.model_tensors[scale_name]()
            offset_lazy = self.model_tensors[offset_name]()
            weight_shape = tuple(weight_lazy.shape)
            scale_shape = tuple(scale_lazy.shape)
            offset_shape = tuple(offset_lazy.shape)

            require(len(weight_shape) == 2, f"{weight_name} must be 2D, got {weight_shape}")
            require(scale_shape == (weight_shape[0], 1), f"{scale_name} shape {scale_shape} != ({weight_shape[0]}, 1)")
            require(offset_shape == scale_shape, f"{offset_name} shape {offset_shape} != {scale_shape}")

            offset_eager = LazyTorchTensor.to_eager(offset_lazy).float()
            require(torch.count_nonzero(offset_eager).item() == 0, f"nonzero weight offset in {offset_name}")

            scale_eager = LazyTorchTensor.to_eager(scale_lazy).float()
            require(torch.isfinite(scale_eager).all().item(), f"non-finite scale in {scale_name}")
            require((scale_eager >= 0).all().item(), f"negative scale in {scale_name}")

            gguf_name = self.map_tensor_name(weight_name)
            records.append({
                "source_name_raw": source_name,
                "source_name": weight_name,
                "gguf_name": gguf_name,
                "shape": weight_shape,
                "scale_min": float(scale_eager.min().item()),
                "scale_max": float(scale_eager.max().item()),
                "offset_nonzero": 0,
            })

            if load_payload:
                weight = LazyTorchTensor.to_eager(weight_lazy).cpu().numpy()
                scale = scale_eager.cpu().numpy()
                raw = w8a8_blocks(weight, scale)
                self.gguf_writer.add_tensor(gguf_name, raw, raw_dtype=W8A8_TYPE)
                for name in (weight_name, scale_name, offset_name):
                    self.model_tensors.pop(name, None)
        return records

    def prepare_tensors(self):
        old_ftype = self.ftype
        self.ftype = gguf.LlamaFileType.MOSTLY_F16
        try:
            self.w8a8_mapping = self.collect_mapping(load_payload=True)
            super().prepare_tensors()
        finally:
            self.ftype = old_ftype


class ModelslimQwen35VisionModel(Qwen3VLVisionModel):
    model_arch = gguf.MODEL_ARCH.MMPROJ

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], torch.Tensor]]:
        return index_modelslim_tensors(self, remote_hf_model_id)


def main() -> None:
    args = parse_args()
    require(args.model_dir.is_dir(), f"{args.model_dir} is not a directory")
    hparams = ModelBase.load_hparams(args.model_dir, False)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    require(arch == "Qwen3_5ForConditionalGeneration", f"unsupported architecture: {arch}")

    with torch.inference_mode():
        if args.mmproj_only:
            require(args.mmproj_output is not None, "--mmproj-only requires --mmproj-output")
            vision_model = ModelslimQwen35VisionModel(
                args.model_dir,
                gguf.LlamaFileType.MOSTLY_F16,
                args.mmproj_output,
                eager=args.no_lazy,
            )
            vision_model.write()
            records = []
        else:
            model = ModelslimQwen35Model(
                args.model_dir,
                gguf.LlamaFileType.MOSTLY_W8A8,
                args.output,
                eager=args.no_lazy,
            )
            if args.dry_run:
                records = model.collect_mapping(load_payload=False)
            else:
                model.write()
                records = model.w8a8_mapping
                if args.mmproj_output is not None:
                    vision_model = ModelslimQwen35VisionModel(
                        args.model_dir,
                        gguf.LlamaFileType.MOSTLY_F16,
                        args.mmproj_output,
                        eager=args.no_lazy,
                    )
                    vision_model.write()

    if args.mapping_out:
        with open(args.mapping_out, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    print(f"mapped {len(records)} W8A8 tensors")
    if args.dry_run:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
