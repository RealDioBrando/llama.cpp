"""Offline metadata contracts; optional real-weight conversion uses local fixtures."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))

import gguf
from conversion.qwen import DFlashModel

REPO = Path(__file__).resolve().parents[2]


def field_contents(reader, name):
    field = reader.get_field(name)
    assert field is not None, f"missing GGUF field: {name}"
    return field.contents()


@pytest.mark.parametrize("target_key", ["rope_parameters", "rope_scaling"])
@pytest.mark.parametrize("layout", ["interleaved", "contiguous", "unspecified", "legacy", "text"])
def test_dflash_rope_metadata(tmp_path, target_key, layout):
    draft = tmp_path / "draft"
    target = tmp_path / "target"
    draft.mkdir()
    target.mkdir()
    rope = {"rope_theta": 5000000.0}
    if layout not in ("legacy", "text"):
        rope["mrope_section"] = [24, 20, 20]
        if layout != "unspecified":
            rope["mrope_interleaved"] = layout == "interleaved"
    hparams = {
        "architectures": ["DFlashDraftModel"], "num_hidden_layers": 1,
        "hidden_size": 256, "intermediate_size": 384, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 128, "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512, "rope_parameters": rope,
        "dflash_config": {"target_layer_ids": [3, 9, 17, 25, 33]},
    }
    # Qwen3Model also reads the on-disk config while initializing its tensor map.
    (draft / "config.json").write_text(json.dumps(hparams))
    target_config = {"text_config": {target_key: {"mrope_section": [24, 20, 20]}}}
    if layout == "text":
        target_config = {target_key: {"rope_theta": 10000.0}}
    (target / "config.json").write_text(json.dumps(target_config))
    path = tmp_path / "metadata.gguf"
    model = DFlashModel(draft, gguf.LlamaFileType.MOSTLY_F16, path,
                        hparams=hparams, target_model_dir=target)
    model.set_gguf_parameters()
    writer = model.gguf_writer
    writer.write_header_to_file(path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader = gguf.GGUFReader(path)
    sections = reader.get_field("dflash.rope.dimension_sections")
    flag = reader.get_field("dflash.rope.mrope_interleaved")
    if layout == "text":
        assert sections is None
    else:
        assert sections is not None
        assert sections.contents() == ([64, 0, 0, 0] if layout == "legacy" else [24, 20, 20, 0])
    if layout in ("legacy", "text"):
        assert flag is None
    else:
        assert flag is not None
        assert flag.types == [gguf.GGUFValueType.BOOL]
        assert flag.contents() == (layout == "interleaved")
    assert field_contents(reader, "dflash.target_layers") == [4, 10, 18, 26, 34]


def local_fixture(name):
    value = os.environ.get(name)
    if not value:
        raise pytest.skip.Exception(f"set {name} to enable the local model conversion test")
    path = Path(value)
    assert path.exists(), f"explicit fixture does not exist: {name}={path}"
    return path


def test_dflash_full_conversion(tmp_path):
    source = local_fixture("LLAMA_TEST_DFLASH_HF")
    target = local_fixture("LLAMA_TEST_VL_TARGET_HF")
    reference = local_fixture("LLAMA_TEST_DFLASH_REFERENCE")
    output = tmp_path / "converted.gguf"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    result = subprocess.run(
        [sys.executable, str(REPO / "convert_hf_to_gguf.py"), str(source),
         "--target-model-dir", str(target), "--outfile", str(output), "--outtype", "f16"],
        env=env, capture_output=True, text=True, timeout=600,
    )
    (tmp_path / "conversion.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr[-6000:]
    actual, expected = gguf.GGUFReader(output), gguf.GGUFReader(reference)
    for key in ("dflash.rope.dimension_sections", "dflash.rope.mrope_interleaved",
                "dflash.rope.freq_base", "dflash.target_layers"):
        assert field_contents(actual, key) == field_contents(expected, key), key
    assert field_contents(actual, "dflash.rope.dimension_sections") == [24, 20, 20, 0]
    assert field_contents(actual, "dflash.rope.mrope_interleaved") is True
    tensors = {tensor.name: tensor for tensor in actual.tensors}
    assert len(tensors) == len(expected.tensors) > 0
    for ref in expected.tensors:
        got = tensors[ref.name]
        assert tuple(got.shape) == tuple(ref.shape), ref.name
        assert got.tensor_type == ref.tensor_type, ref.name
        assert hashlib.sha256(got.data.tobytes()).digest() == hashlib.sha256(ref.data.tobytes()).digest(), ref.name
