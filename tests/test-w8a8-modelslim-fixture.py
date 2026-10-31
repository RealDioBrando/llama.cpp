#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "tools" / "w8a8" / "convert_modelslim.py"
INSPECTOR = ROOT / "tools" / "w8a8" / "inspect_modelslim.py"


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
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(base_config(architecture)), encoding="utf-8")
    weight_name = "model.language_model.layers.0.mlp.gate_proj.weight"
    scale_name = weight_name.removesuffix(".weight") + ".weight_scale"
    offset_name = weight_name.removesuffix(".weight") + ".weight_offset"
    norm_name = "model.language_model.layers.0.input_layernorm.weight"
    tensors = {
        weight_name: np.arange(32 * k, dtype=np.int16).astype(weight_dtype).reshape(32, k),
        scale_name: np.full((32, 1), 0.25, dtype=np.float32),
        offset_name: np.full((32, 1), 1.0 if nonzero_offset else 0.0, dtype=np.float32),
        norm_name: np.ones((64,), dtype=np.float32),
    }
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
    (model_dir / "quant_model_description.json").write_text(json.dumps(description), encoding="utf-8")
    shard = "quant_model_weights-00001-of-00001.safetensors"
    save_file(tensors, str(model_dir / shard))
    index = {"weight_map": {name: shard for name in tensors}}
    (model_dir / "quant_model_weights.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def run_tool(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT) + ":" + str(ROOT / "gguf-py")
    return subprocess.run([sys.executable, str(script), *args], text=True, capture_output=True, env=env)


class ModelslimFixtureTest(unittest.TestCase):
    def test_valid_fixture_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir)
            mapping = Path(tmp) / "mapping.json"
            result = run_tool(CONVERTER, str(model_dir), str(Path(tmp) / "out.gguf"), "--dry-run", "--mapping-out", str(mapping))
            self.assertEqual(result.returncode, 0, result.stderr)
            records = json.loads(mapping.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["gguf_name"], "blk.0.ffn_gate.weight")
            self.assertEqual(records[0]["offset_nonzero"], 0)

    def test_inspector_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir)
            result = run_tool(INSPECTOR, str(model_dir), "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["quantized_weights"], 1)
            self.assertEqual(report["nonzero_offsets"], [])

    def test_missing_scale_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, missing="model.language_model.layers.0.mlp.gate_proj.weight_scale")
            result = run_tool(CONVERTER, str(model_dir), str(Path(tmp) / "out.gguf"), "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("scale", result.stderr.lower())

    def test_missing_offset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, missing="model.language_model.layers.0.mlp.gate_proj.weight_offset")
            result = run_tool(CONVERTER, str(model_dir), str(Path(tmp) / "out.gguf"), "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("offset", result.stderr.lower())

    def test_nonzero_offset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, nonzero_offset=True)
            result = run_tool(CONVERTER, str(model_dir), str(Path(tmp) / "out.gguf"), "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("nonzero", result.stderr.lower())

    def test_invalid_weight_dtype_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, weight_dtype=np.dtype(np.float32))
            result = run_tool(CONVERTER, str(model_dir), str(Path(tmp) / "out.gguf"), "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("int8", result.stderr.lower())

    def test_invalid_k_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, k=63)
            result = run_tool(CONVERTER, str(model_dir), str(Path(tmp) / "out.gguf"), "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("divisible", result.stderr.lower())

    def test_unsupported_architecture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model"
            make_fixture(model_dir, architecture="UnsupportedForCausalLM")
            result = run_tool(CONVERTER, str(model_dir), str(Path(tmp) / "out.gguf"), "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsupported architecture", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
