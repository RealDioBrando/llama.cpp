#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

import gguf  # noqa: E402
from conversion.base import LazyTorchTensor, ModelBase, ModelType, get_model_architecture, logger  # noqa: E402
from conversion.qwen import Qwen3MoeModel, Qwen3_5MoeTextModel, Qwen3_5TextModel  # noqa: E402
from conversion.qwen3vl import Qwen3VLVisionModel  # noqa: E402

W8A8_TYPE = gguf.GGMLQuantizationType.W8A8
QUANT_TYPES = frozenset({"W8A8_DYNAMIC", "W8A8"})
SEPARATE_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
FUSED_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<projection>gate_up_proj|down_proj)\.weight$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"error: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a ModelSlim W8A8_DYNAMIC Qwen3.5 dense or MoE checkpoint directly to GGUF."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate and print the tensor mapping without writing GGUF")
    parser.add_argument("--mapping-out", type=Path, help="write the mapping report as JSON")
    parser.add_argument("--mmproj-output", type=Path, help="optionally write the vision projector as a separate mmproj GGUF")
    parser.add_argument("--mmproj-only", action="store_true", help="write only the vision projector")
    parser.add_argument("--no-lazy", action="store_true", help="load tensors eagerly while writing")
    parser.add_argument(
        "--allow-static-w8a8",
        action="store_true",
        help="accept static W8A8 weights while omitting their static activation metadata",
    )
    parser.add_argument(
        "--dequant-static-w8a8",
        action="store_true",
        help="diagnostic: dequantize static W8A8 weights to F16 instead of packing them",
    )
    parser.add_argument(
        "--dequant-expert-w8a8",
        action="store_true",
        help="diagnostic: dequantize routed expert weights to F16 instead of packing them",
    )
    parser.add_argument(
        "--no-temp-file",
        action="store_true",
        help="keep converted tensors in memory instead of streaming them through the GGUF temp file",
    )
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


def normalize_expert_scale(scale: np.ndarray) -> np.ndarray:
    require(scale.dtype in (np.float16, np.float32), f"expert weight_scale must be FP16/FP32, got {scale.dtype}")
    if scale.ndim == 3 and scale.shape[-1] == 1:
        scale = scale[..., 0]
    require(scale.ndim == 2, f"expert weight_scale must have shape [E,N] or [E,N,1], got {scale.shape}")
    return scale


def w8a8_expert_blocks(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    require(weight.dtype == np.int8, f"expert weight must be INT8, got {weight.dtype}")
    require(weight.ndim == 3, f"expert weight must be 3D [E,N,K], got {weight.shape}")
    scale = normalize_expert_scale(scale)
    require(
        scale.shape == (weight.shape[0], weight.shape[1]),
        f"expert weight_scale shape {scale.shape} != {(weight.shape[0], weight.shape[1])}",
    )
    require(weight.shape[2] % 32 == 0, f"K must be divisible by 32, got {weight.shape[2]}")

    n_experts, n, k = weight.shape
    raw = np.empty((n_experts, n, (k // 32) * 34), dtype=np.uint8)
    for expert_id in range(n_experts):
        raw[expert_id] = w8a8_blocks(weight[expert_id], scale[expert_id].reshape(n, 1))
    return raw


def index_modelslim_tensors(model: ModelBase, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], torch.Tensor]]:
    if remote_hf_model_id is not None:
        raise NotImplementedError("remote ModelSlim indexing is not supported")

    shard_patterns = ("quant_model_weights-*.safetensors", "quant_model_weight_w8a8_dynamic-*.safetensors")
    shards = sorted({path for pattern in shard_patterns for path in model.dir_model.glob(pattern)})
    require(len(shards) > 0, f"no ModelSlim safetensors shards in {model.dir_model}")
    require(all(path.is_file() for path in shards), f"incomplete ModelSlim shard set in {model.dir_model}")

    tensors: dict[str, Callable[[], torch.Tensor]] = {}
    index_paths = sorted(model.dir_model.glob("*.safetensors.index.json"))
    require(len(index_paths) <= 1, f"ambiguous ModelSlim index files: {[p.name for p in index_paths]}")
    index_path = index_paths[0] if index_paths else None
    index = None
    if index_path is not None:
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


class _ModelslimQwen35Mixin:
    model_arch: gguf.MODEL_ARCH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modelslim_description = self._load_description(self.dir_model)
        self.w8a8_mapping: list[dict[str, object]] = []
        self._accounted_w8a8_sources: set[str] = set()
        self.allow_static_w8a8 = False
        self.dequant_static_w8a8 = False
        self.dequant_expert_w8a8 = False

    @staticmethod
    def _load_description(model_dir: Path) -> dict[str, object]:
        path = model_dir / "quant_model_description.json"
        require(path.is_file(), f"missing {path}")
        with open(path, "r", encoding="utf-8") as f:
            description = json.load(f)
        require(isinstance(description, dict), "quant_model_description.json must contain an object")
        return description

    def _text_hparams(self) -> dict[str, object]:
        return {**self.hparams, **self.hparams.get("text_config", {})}

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], torch.Tensor]]:
        hparams = self._text_hparams()
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
            if kind not in QUANT_TYPES or not original_name.endswith(".weight"):
                continue
            normalized = self._normalize_description_name(original_name)
            if normalized is None:
                continue
            result.append((original_name, normalized))
        return result

    @staticmethod
    def _triple_names(weight_name: str) -> tuple[str, str]:
        base = weight_name.removesuffix(".weight")
        return base + ".weight_scale", base + ".weight_offset"

    def _tensor(self, name: str, description: str) -> torch.Tensor:
        require(name in self.model_tensors, f"missing normalized {description} tensor: {name}")
        return self.model_tensors[name]()

    def _validate_scale(self, scale_lazy: torch.Tensor, name: str) -> torch.Tensor:
        scale = LazyTorchTensor.to_eager(scale_lazy).float()
        require(torch.isfinite(scale).all().item(), f"non-finite scale in {name}")
        require((scale >= 0).all().item(), f"negative scale in {name}")
        return scale

    def _validate_offset(self, offset_lazy: torch.Tensor, name: str) -> int:
        offset = LazyTorchTensor.to_eager(offset_lazy).float()
        nonzero = int(torch.count_nonzero(offset).item())
        require(nonzero == 0, f"nonzero weight offset in {name}")
        return nonzero

    @staticmethod
    def _quant_aux_names(weight_name: str) -> list[str]:
        base = weight_name.removesuffix(".weight")
        return [base + suffix for suffix in (".input_scale", ".input_offset", ".quant_bias", ".deq_scale")]

    def _check_quantization_mode(self, source_name: str) -> str:
        quant_type = str(self.modelslim_description.get(source_name))
        if quant_type == "W8A8":
            require(
                self.allow_static_w8a8,
                f"static W8A8 tensor {source_name} requires --allow-static-w8a8",
            )
        require(quant_type in QUANT_TYPES, f"unsupported quantization type {quant_type} for {source_name}")
        return quant_type

    def _remove_quant_aux(self, weight_name: str) -> list[str]:
        present = [name for name in self._quant_aux_names(weight_name) if name in self.model_tensors]
        for name in present:
            self.model_tensors.pop(name, None)
        return present

    def _pack_dense_weight(self, source_name: str, weight_name: str) -> dict[str, object]:
        quant_type = self._check_quantization_mode(source_name)
        scale_name, offset_name = self._triple_names(weight_name)
        weight_lazy = self._tensor(weight_name, "weight")
        scale_lazy = self._tensor(scale_name, "scale")
        offset_lazy = self._tensor(offset_name, "offset")
        weight_shape = tuple(weight_lazy.shape)
        scale_shape = tuple(scale_lazy.shape)
        offset_shape = tuple(offset_lazy.shape)

        require(len(weight_shape) == 2, f"{weight_name} must be 2D, got {weight_shape}")
        require(scale_shape == (weight_shape[0], 1), f"{scale_name} shape {scale_shape} != ({weight_shape[0]}, 1)")
        require(offset_shape == scale_shape, f"{offset_name} shape {offset_shape} != {scale_shape}")
        self._validate_offset(offset_lazy, offset_name)
        scale = self._validate_scale(scale_lazy, scale_name)

        gguf_name = self.map_tensor_name(weight_name)
        omitted = self._remove_quant_aux(weight_name)
        record = {
            "kind": "dense",
            "source_quant_type": quant_type,
            "omitted_static_metadata": omitted,
            "source_name_raw": source_name,
            "source_name": weight_name,
            "gguf_name": gguf_name,
            "shape": weight_shape,
            "scale_shape": scale_shape,
            "scale_min": float(scale.min().item()),
            "scale_max": float(scale.max().item()),
            "offset_nonzero": 0,
        }

        weight = LazyTorchTensor.to_eager(weight_lazy).cpu().numpy()
        if quant_type == "W8A8" and self.dequant_static_w8a8:
            dequantized = (weight.astype(np.float32) * scale.cpu().numpy().reshape(-1, 1)).astype(np.float16)
            self.gguf_writer.add_tensor(gguf_name, dequantized)
            record["output_type"] = "F16"
        else:
            raw = w8a8_blocks(weight, scale.cpu().numpy())
            self.gguf_writer.add_tensor(gguf_name, raw, raw_dtype=W8A8_TYPE)
            record["output_type"] = "W8A8"
        for name in (weight_name, scale_name, offset_name):
            self.model_tensors.pop(name, None)
        self._accounted_w8a8_sources.add(source_name)
        return record

    def _collect_dense_mapping(self, load_payload: bool) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for source_name, weight_name in self._quantized_weight_names():
            if ".mlp.experts." in weight_name:
                continue
            quant_type = self._check_quantization_mode(source_name)
            if load_payload:
                records.append(self._pack_dense_weight(source_name, weight_name))
                continue

            scale_name, offset_name = self._triple_names(weight_name)
            weight_lazy = self._tensor(weight_name, "weight")
            scale_lazy = self._tensor(scale_name, "scale")
            offset_lazy = self._tensor(offset_name, "offset")
            weight_shape = tuple(weight_lazy.shape)
            scale_shape = tuple(scale_lazy.shape)
            offset_shape = tuple(offset_lazy.shape)
            require(len(weight_shape) == 2, f"{weight_name} must be 2D, got {weight_shape}")
            require(scale_shape == (weight_shape[0], 1),
                    f"{scale_name} shape {scale_shape} != ({weight_shape[0]}, 1)")
            require(offset_shape == scale_shape, f"{offset_name} shape {offset_shape} != {scale_shape}")
            self._validate_offset(offset_lazy, offset_name)
            scale = self._validate_scale(scale_lazy, scale_name)
            records.append({
                "kind": "dense",
                "source_quant_type": quant_type,
                "omitted_static_metadata": [name for name in self._quant_aux_names(weight_name) if name in self.model_tensors],
                "source_name_raw": source_name,
                "source_name": weight_name,
                "gguf_name": self.map_tensor_name(weight_name),
                "shape": weight_shape,
                "scale_shape": scale_shape,
                "scale_min": float(scale.min().item()),
                "scale_max": float(scale.max().item()),
                "offset_nonzero": 0,
            })
            self._accounted_w8a8_sources.add(source_name)
        return records

    def _collect_mapping_impl(self, load_payload: bool) -> list[dict[str, object]]:
        return self._collect_dense_mapping(load_payload)

    def collect_mapping(self, load_payload: bool = True) -> list[dict[str, object]]:
        self._accounted_w8a8_sources.clear()
        records = self._collect_mapping_impl(load_payload)
        described = {
            name for name, kind in self.modelslim_description.items()
            if kind in QUANT_TYPES and name.endswith(".weight")
        }
        missing = sorted(described - self._accounted_w8a8_sources)
        require(not missing, f"unmapped W8A8 weights: {missing[:10]}")
        return records

    def prepare_tensors(self):
        old_ftype = self.ftype
        self.ftype = gguf.LlamaFileType.MOSTLY_F16
        try:
            self.w8a8_mapping = self.collect_mapping(load_payload=True)
            super().prepare_tensors()
        finally:
            self.ftype = old_ftype


class ModelslimQwen35Model(_ModelslimQwen35Mixin, Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35


class _ModelslimMoeMixin:
    def _num_experts(self) -> int:
        hparams = self._text_hparams()
        n_experts = hparams.get("num_experts", hparams.get("num_local_experts"))
        require(isinstance(n_experts, int) and n_experts > 0, f"invalid num_experts/num_local_experts: {n_experts}")
        return n_experts

    @staticmethod
    def _expert_info(weight_name: str) -> dict[str, object] | None:
        match = SEPARATE_EXPERT_RE.match(weight_name)
        if match:
            return {
                "layout": "separate",
                "layer": int(match.group("layer")),
                "expert_id": int(match.group("expert")),
                "projection": match.group("projection"),
            }
        match = FUSED_EXPERT_RE.match(weight_name)
        if match:
            return {
                "layout": "fused",
                "layer": int(match.group("layer")),
                "expert_id": None,
                "projection": match.group("projection"),
            }
        return None

    @staticmethod
    def _merged_expert_name(layer: int, projection: str) -> str:
        return f"model.layers.{layer}.mlp.experts.{projection}.weight"

    def _collect_separate_experts(self, load_payload: bool) -> list[dict[str, object]]:
        n_experts = self._num_experts()
        groups: dict[tuple[int, str], dict[int, tuple[str, str]]] = defaultdict(dict)
        for source_name, weight_name in self._quantized_weight_names():
            if ".mlp.experts." not in weight_name:
                continue
            info = self._expert_info(weight_name)
            require(info is not None, f"unrecognized expert weight name: {weight_name}")
            if info["layout"] != "separate":
                continue
            layer = int(info["layer"])
            expert_id = int(info["expert_id"])
            projection = str(info["projection"])
            require(expert_id not in groups[(layer, projection)],
                    f"duplicate expert {expert_id} for layer {layer} {projection}")
            groups[(layer, projection)][expert_id] = (source_name, weight_name)

        records: list[dict[str, object]] = []
        for (layer, projection), entries in sorted(groups.items()):
            require(set(entries) == set(range(n_experts)),
                    f"layer {layer} {projection} has experts {sorted(entries)}; expected 0..{n_experts - 1}")
            _, first_name = entries[0]
            first_weight = self._tensor(first_name, "expert weight")
            first_scale_name, first_offset_name = self._triple_names(first_name)
            first_scale = self._tensor(first_scale_name, "expert scale")
            first_offset = self._tensor(first_offset_name, "expert offset")
            weight_shape = tuple(first_weight.shape)
            scale_shape = tuple(first_scale.shape)
            offset_shape = tuple(first_offset.shape)
            require(len(weight_shape) == 2, f"{first_name} must be 2D, got {weight_shape}")
            require(scale_shape == (weight_shape[0], 1),
                    f"{first_scale_name} shape {scale_shape} != ({weight_shape[0]}, 1)")
            require(offset_shape == scale_shape, f"{first_offset_name} shape {offset_shape} != {scale_shape}")
            require(weight_shape[1] % 32 == 0, f"K must be divisible by 32, got {weight_shape[1]}")

            gguf_name = self.map_tensor_name(self._merged_expert_name(layer, projection))
            dequant_experts = self.dequant_expert_w8a8
            raw = None
            if load_payload:
                raw = np.empty(
                    (n_experts, *weight_shape) if dequant_experts else
                    (n_experts, weight_shape[0], (weight_shape[1] // 32) * 34),
                    dtype=np.float16 if dequant_experts else np.uint8,
                )

            for expert_id in range(n_experts):
                source_name, weight_name = entries[expert_id]
                scale_name, offset_name = self._triple_names(weight_name)
                weight_lazy = self._tensor(weight_name, "expert weight")
                scale_lazy = self._tensor(scale_name, "expert scale")
                offset_lazy = self._tensor(offset_name, "expert offset")
                require(tuple(weight_lazy.shape) == weight_shape,
                        f"{weight_name} shape {tuple(weight_lazy.shape)} != {weight_shape}")
                require(tuple(scale_lazy.shape) == scale_shape,
                        f"{scale_name} shape {tuple(scale_lazy.shape)} != {scale_shape}")
                require(tuple(offset_lazy.shape) == offset_shape,
                        f"{offset_name} shape {tuple(offset_lazy.shape)} != {offset_shape}")
                self._validate_offset(offset_lazy, offset_name)
                scale = self._validate_scale(scale_lazy, scale_name)
                if load_payload:
                    weight = LazyTorchTensor.to_eager(weight_lazy).cpu().numpy()
                    if dequant_experts:
                        raw[expert_id] = (weight.astype(np.float32) * scale.cpu().numpy().reshape(-1, 1)).astype(np.float16)
                    else:
                        raw[expert_id] = w8a8_blocks(weight, scale.cpu().numpy())
                records.append({
                    "kind": "expert",
                    "output_type": "F16" if dequant_experts else "W8A8",
                    "layout": "separate",
                    "layer": layer,
                    "projection": projection,
                    "expert_id": expert_id,
                    "source_name_raw": source_name,
                    "source_name": weight_name,
                    "gguf_name": gguf_name,
                    "shape": weight_shape,
                    "scale_shape": scale_shape,
                    "stack_shape": (n_experts, *weight_shape),
                    "stack_scale_shape": (n_experts, weight_shape[0]),
                    "scale_min": float(scale.min().item()),
                    "scale_max": float(scale.max().item()),
                    "offset_nonzero": 0,
                })
                self._accounted_w8a8_sources.add(source_name)
                for name in (weight_name, scale_name, offset_name):
                    self.model_tensors.pop(name, None)

            if load_payload:
                assert raw is not None
                if dequant_experts:
                    self.gguf_writer.add_tensor(gguf_name, raw)
                    logger.info(f"gguf: wrote {gguf_name} as F16 [{n_experts},{weight_shape[0]},{weight_shape[1]}]")
                else:
                    self.gguf_writer.add_tensor(gguf_name, raw, raw_dtype=W8A8_TYPE)
                    logger.info(f"gguf: wrote {gguf_name} as W8A8 [{n_experts},{weight_shape[0]},{weight_shape[1]}]")
            else:
                logger.info(f"gguf: validated {n_experts} experts for {gguf_name}")
        return records

    def _collect_fused_experts(self, load_payload: bool) -> list[dict[str, object]]:
        n_experts = self._num_experts()
        records: list[dict[str, object]] = []
        for source_name, weight_name in self._quantized_weight_names():
            if ".mlp.experts." not in weight_name:
                continue
            info = self._expert_info(weight_name)
            require(info is not None, f"unrecognized expert weight name: {weight_name}")
            if info["layout"] != "fused":
                continue

            layer = int(info["layer"])
            projection = str(info["projection"])
            scale_name, offset_name = self._triple_names(weight_name)
            weight_lazy = self._tensor(weight_name, "expert weight")
            scale_lazy = self._tensor(scale_name, "expert scale")
            offset_lazy = self._tensor(offset_name, "expert offset")
            weight_shape = tuple(weight_lazy.shape)
            raw_scale_shape = tuple(scale_lazy.shape)
            raw_offset_shape = tuple(offset_lazy.shape)
            require(len(weight_shape) == 3, f"{weight_name} must be 3D, got {weight_shape}")
            require(weight_shape[0] == n_experts,
                    f"{weight_name} expert count {weight_shape[0]} != {n_experts}")
            if raw_scale_shape == (*weight_shape[:2], 1):
                scale_shape = weight_shape[:2]
            else:
                scale_shape = raw_scale_shape
            require(scale_shape == weight_shape[:2],
                    f"{scale_name} shape {raw_scale_shape} incompatible with weight shape {weight_shape}")
            require(raw_offset_shape in (scale_shape, (*scale_shape, 1)),
                    f"{offset_name} shape {raw_offset_shape} incompatible with scale shape {scale_shape}")
            require(weight_shape[2] % 32 == 0, f"K must be divisible by 32, got {weight_shape[2]}")
            self._validate_offset(offset_lazy, offset_name)
            scale = self._validate_scale(scale_lazy, scale_name)

            if load_payload:
                weight = LazyTorchTensor.to_eager(weight_lazy).cpu().numpy()
                if projection == "gate_up_proj":
                    require(weight_shape[1] % 2 == 0, f"{weight_name} N is not divisible by 2: {weight_shape[1]}")
                    half = weight_shape[1] // 2
                    targets = [
                        ("gate_proj", weight[:, :half, :], scale[:, :half], [0, half]),
                        ("up_proj", weight[:, half:, :], scale[:, half:], [half, weight_shape[1]]),
                    ]
                else:
                    targets = [(projection, weight, scale, None)]
            else:
                if projection == "gate_up_proj":
                    require(weight_shape[1] % 2 == 0, f"{weight_name} N is not divisible by 2: {weight_shape[1]}")
                    half = weight_shape[1] // 2
                    targets = [
                        ("gate_proj", None, scale[:, :half], [0, half]),
                        ("up_proj", None, scale[:, half:], [half, weight_shape[1]]),
                    ]
                else:
                    targets = [(projection, None, scale, None)]

            for target_projection, target_weight, target_scale, fused_slice in targets:
                target_name = self.map_tensor_name(self._merged_expert_name(layer, target_projection))
                target_scale_np = target_scale.cpu().numpy() if hasattr(target_scale, "cpu") else target_scale
                target_shape = tuple(target_weight.shape) if target_weight is not None else (
                    n_experts,
                    half if projection == "gate_up_proj" else weight_shape[1],
                    weight_shape[2],
                )
                records.append({
                    "kind": "expert",
                    "layout": "fused",
                    "layer": layer,
                    "projection": target_projection,
                    "source_projection": projection,
                    "expert_id": None,
                    "fused_slice": fused_slice,
                    "source_name_raw": source_name,
                    "source_name": weight_name,
                    "gguf_name": target_name,
                    "shape": target_shape,
                    "scale_shape": tuple(target_scale_np.shape),
                    "stack_shape": target_shape,
                    "stack_scale_shape": tuple(target_scale_np.shape),
                    "scale_min": float(target_scale_np.min()),
                    "scale_max": float(target_scale_np.max()),
                    "offset_nonzero": 0,
                })
                if load_payload:
                    assert target_weight is not None
                    raw = w8a8_expert_blocks(target_weight, target_scale_np)
                    self.gguf_writer.add_tensor(target_name, raw, raw_dtype=W8A8_TYPE)
                    logger.info(f"gguf: wrote fused {target_name} as W8A8 {target_shape}")

            self._accounted_w8a8_sources.add(source_name)
            for name in (weight_name, scale_name, offset_name):
                self.model_tensors.pop(name, None)
        return records

    def _collect_mapping_impl(self, load_payload: bool) -> list[dict[str, object]]:
        records = self._collect_separate_experts(load_payload)
        records.extend(self._collect_fused_experts(load_payload))
        records.extend(self._collect_dense_mapping(load_payload))
        return records


class ModelslimQwen3MoeModel(_ModelslimMoeMixin, _ModelslimQwen35Mixin, Qwen3MoeModel):
    model_arch = gguf.MODEL_ARCH.QWEN3MOE


class ModelslimQwen35MoeModel(_ModelslimMoeMixin, _ModelslimQwen35Mixin, Qwen3_5MoeTextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35MOE


class ModelslimQwen35VisionModel(Qwen3VLVisionModel):
    model_arch = gguf.MODEL_ARCH.MMPROJ

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], torch.Tensor]]:
        return index_modelslim_tensors(self, remote_hf_model_id)


MODEL_CLASSES = {
    "Qwen3_5ForConditionalGeneration": ModelslimQwen35Model,
    "Qwen3_5ForCausalLM": ModelslimQwen35Model,
    "Qwen3_5MoeForConditionalGeneration": ModelslimQwen35MoeModel,
    "Qwen3_5MoeForCausalLM": ModelslimQwen35MoeModel,
    "Qwen3MoeForCausalLM": ModelslimQwen3MoeModel,
}


def main() -> None:
    args = parse_args()
    require(args.model_dir.is_dir(), f"{args.model_dir} is not a directory")
    hparams = ModelBase.load_hparams(args.model_dir, False)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    require(arch in MODEL_CLASSES, f"unsupported architecture: {arch}")

    use_temp_file = not args.dry_run and not args.no_temp_file
    with torch.inference_mode():
        if args.mmproj_only:
            require(args.mmproj_output is not None, "--mmproj-only requires --mmproj-output")
            vision_model = ModelslimQwen35VisionModel(
                args.model_dir,
                gguf.LlamaFileType.MOSTLY_F16,
                args.mmproj_output,
                eager=args.no_lazy,
                use_temp_file=use_temp_file,
            )
            vision_model.write()
            records = []
        else:
            model_class = MODEL_CLASSES[arch]
            model = model_class(
                args.model_dir,
                gguf.LlamaFileType.MOSTLY_W8A8,
                args.output,
                eager=args.no_lazy,
                use_temp_file=use_temp_file,
            )
            model.allow_static_w8a8 = args.allow_static_w8a8
            model.dequant_static_w8a8 = args.dequant_static_w8a8
            model.dequant_expert_w8a8 = args.dequant_expert_w8a8
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
                        use_temp_file=use_temp_file,
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
