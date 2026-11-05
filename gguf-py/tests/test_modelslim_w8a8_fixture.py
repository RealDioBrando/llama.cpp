#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "convert_hf_to_gguf.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402
from conversion.qwen import Qwen3_5TextModel  # noqa: E402


def base_config(architecture: str = "Qwen3_5ForConditionalGeneration") -> dict:
    return {
        "architectures": [architecture],
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": False,
            "attn_output_gate": True,
            "full_attention_interval": 4,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 64,
            "intermediate_size": 128,
            "layer_types": ["linear_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 64,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 64,
            "max_position_embeddings": 128,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [1, 1, 1],
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
            },
            "vocab_size": 128,
        },
    }


def make_fixture(
    model_dir: Path,
    *,
    architecture: str = "Qwen3_5ForConditionalGeneration",
    missing: str | None = None,
    nonzero_offset: bool = False,
    weight_dtype: np.dtype = np.dtype(np.int8),
    k: int = 64,
    static_metadata: bool = False,
    embedding_w8a8: bool = False,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(base_config(architecture)), encoding="utf-8")
    weight_name = "model.language_model.layers.0.mlp.gate_proj.weight"
    scale_name = weight_name.removesuffix(".weight") + ".weight_scale"
    offset_name = weight_name.removesuffix(".weight") + ".weight_offset"
    norm_name = "model.language_model.layers.0.input_layernorm.weight"
    embedding_name = "model.language_model.embed_tokens.weight"
    tensors = {
        weight_name: np.arange(32 * k, dtype=np.int16).astype(weight_dtype).reshape(32, k),
        scale_name: np.full((32, 1), 0.25, dtype=np.float32),
        offset_name: np.full((32, 1), 1.0 if nonzero_offset else 0.0, dtype=np.float32),
        norm_name: np.ones((64,), dtype=np.float32),
    }
    if embedding_w8a8:
        embedding_scale = embedding_name.removesuffix(".weight") + ".weight_scale"
        embedding_offset = embedding_name.removesuffix(".weight") + ".weight_offset"
        tensors[embedding_name] = np.zeros((128, 64), dtype=np.int8)
        tensors[embedding_scale] = np.ones((128, 1), dtype=np.float32)
        tensors[embedding_offset] = np.zeros((128, 1), dtype=np.float32)
    for name in (missing,) if missing else ():
        tensors.pop(name, None)

    description = {
        "version": "1.0.0",
        "model_quant_type": "W8A8",
        weight_name: "W8A8_DYNAMIC",
        scale_name: "W8A8_DYNAMIC",
        offset_name: "W8A8_DYNAMIC",
        norm_name: "FLOAT",
    }
    if missing:
        description.pop(missing, None)
    if embedding_w8a8:
        description[embedding_name] = "W8A8_DYNAMIC"
        description[embedding_name.removesuffix(".weight") + ".weight_scale"] = "W8A8_DYNAMIC"
        description[embedding_name.removesuffix(".weight") + ".weight_offset"] = "W8A8_DYNAMIC"
    if static_metadata:
        description["model.language_model.layers.0.self_attn.q_proj.input_scale"] = "W8A8"
    (model_dir / "quant_model_description.json").write_text(json.dumps(description), encoding="utf-8")
    shard = "quant_model_weights-00001-of-00001.safetensors"
    save_file(tensors, str(model_dir / shard))
    index = {"weight_map": {name: shard for name in tensors}}
    (model_dir / "quant_model_weights.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def run_tool(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "gguf-py")))
    return subprocess.run([sys.executable, str(script), *args], text=True, capture_output=True, env=env)


class ModelslimFixtureTest(unittest.TestCase):
    def test_valid_fixture_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir)
            model = Qwen3_5TextModel(model_dir, gguf.LlamaFileType.MOSTLY_W8A8,
                                     Path(tmp) / "out.gguf", dry_run=True)
            model.prepare_tensors()
            self.assertEqual(len(model._modelslim_mapping), 1)
            self.assertEqual(model._modelslim_mapping[0]["gguf_name"], "blk.0.ffn_gate.weight")

    def test_w8a8_token_embedding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, embedding_w8a8=True)
            result = run_tool(CONVERTER, str(model_dir), "--outfile", str(Path(tmp) / "out.gguf"),
                              "--outtype", "w8a8", "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("token embedding", result.stderr.lower())

    def test_missing_scale_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, missing="model.language_model.layers.0.mlp.gate_proj.weight_scale")
            result = run_tool(CONVERTER, str(model_dir), "--outfile", str(Path(tmp) / "out.gguf"),
                              "--outtype", "w8a8", "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("scale", result.stderr.lower())

    def test_missing_offset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, missing="model.language_model.layers.0.mlp.gate_proj.weight_offset")
            result = run_tool(CONVERTER, str(model_dir), "--outfile", str(Path(tmp) / "out.gguf"),
                              "--outtype", "w8a8", "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("offset", result.stderr.lower())

    def test_nonzero_offset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, nonzero_offset=True)
            result = run_tool(CONVERTER, str(model_dir), "--outfile", str(Path(tmp) / "out.gguf"),
                              "--outtype", "w8a8", "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("nonzero", result.stderr.lower())

    def test_invalid_weight_dtype_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, weight_dtype=np.dtype(np.float32))
            result = run_tool(CONVERTER, str(model_dir), "--outfile", str(Path(tmp) / "out.gguf"),
                              "--outtype", "w8a8", "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("int8", result.stderr.lower())

    def test_invalid_k_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, k=63)
            result = run_tool(CONVERTER, str(model_dir), "--outfile", str(Path(tmp) / "out.gguf"),
                              "--outtype", "w8a8", "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("divisible", result.stderr.lower())

    def test_static_w8a8_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, static_metadata=True)
            result = run_tool(CONVERTER, str(model_dir), "--outfile", str(Path(tmp) / "out.gguf"),
                              "--outtype", "w8a8", "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only w8a8_dynamic", result.stderr.lower())

    def test_unsupported_architecture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, architecture="UnsupportedForCausalLM")
            result = run_tool(CONVERTER, str(model_dir), "--outfile", str(Path(tmp) / "out.gguf"),
                              "--outtype", "w8a8", "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not supported", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
