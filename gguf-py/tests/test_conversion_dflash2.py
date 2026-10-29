from copy import deepcopy
import json
import os
from pathlib import Path
import sys

# Necessary to load the local converter and gguf packages
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gguf
import pytest

from conversion.qwen import DFlashModel, DSparkModel, Qwen3Model


@pytest.fixture
def dflash2_config():
    """Provide a synthetic Speculators configuration without any model weights."""
    return {
        "architectures": ["DFlash2DraftModel"],
        "transformer_layer_config": {
            "model_type": "qwen3",
            "num_hidden_layers": 5,
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "vocab_size": 151936,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 40960,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000},
        },
        "block_size": 9,
        "mask_token_id": 151669,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "aux_hidden_state_layer_ids": [1, 9, 17, 25, 33],
    }


@pytest.fixture
def capture_parent_hparams(monkeypatch):
    """Capture the real draft constructor's parent call without loading a model."""
    def init_parent(self, *args, hparams, **kwargs):
        self.hparams = hparams

    monkeypatch.setattr(Qwen3Model, "__init__", init_parent)


def test_dflash_nested_config(dflash2_config, capture_parent_hparams, tmp_path):
    """Normalize a nested Speculators configuration before parent initialization.

    Pass nested backbone settings and top-level draft options to DFlashModel with an empty dflash_config.
    Capture the parent's hparams and check the dimensions, all copied draft options and converted layer IDs.
    Compare against a deep copy to ensure normalization does not mutate the caller's configuration.
    """
    dflash2_config["dflash_config"] = {}
    before = deepcopy(dflash2_config)

    model = DFlashModel(tmp_path, hparams=dflash2_config)

    assert model.hparams["num_hidden_layers"] == 5
    assert model.hparams["hidden_size"] == 2560
    assert model.hparams["intermediate_size"] == 9728
    assert model.hparams["vocab_size"] == 151936
    assert model.hparams["dflash_config"] == {
        "block_size": 9,
        "mask_token_id": 151669,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [0, 8, 16, 24, 32],
    }
    assert dflash2_config == before


def test_dflash_explicit_config_wins(dflash2_config, capture_parent_hparams, tmp_path):
    """Resolve conflicting backbone and draft settings without overwriting explicit options.

    Set outer hidden_size to 4096 and nested hidden_size to 2560, with explicit nested block_size and target_layer_ids.
    Capture the parent's hparams; expect hidden_size=2560, block_size=4 and target_layer_ids=[2, 4].
    Verify that the original configuration is unchanged.
    """
    dflash2_config["hidden_size"] = 4096
    dflash2_config["dflash_config"] = {"block_size": 4, "target_layer_ids": [2, 4]}
    before = deepcopy(dflash2_config)

    model = DFlashModel(tmp_path, hparams=dflash2_config)

    assert model.hparams["hidden_size"] == 2560
    assert model.hparams["dflash_config"]["block_size"] == 4
    assert model.hparams["dflash_config"]["target_layer_ids"] == [2, 4]
    assert dflash2_config == before


@pytest.mark.parametrize("model_class", [DFlashModel, DSparkModel])
def test_dflash_legacy_config_unchanged(model_class, capture_parent_hparams, tmp_path):
    """Keep the existing flat configuration compatible with both draft classes.

    Run once with DFlashModel and once with DSparkModel, using existing nested draft options and no transformer_layer_config.
    Capture the parent's hparams and verify that both the effective and caller-owned configurations are unchanged.
    For DSparkModel, also check its existing anchor default of True and draft vocabulary size of 32.
    """
    config = {
        "num_hidden_layers": 2,
        "hidden_size": 16,
        "vocab_size": 32,
        "dflash_config": {"block_size": 4, "mask_token_id": 3, "target_layer_ids": [0, 1]},
    }
    before = deepcopy(config)

    model = model_class(tmp_path, hparams=config)

    assert model.hparams == before
    assert config == before
    if model_class is DSparkModel:
        assert model._sample_from_anchor is True
        assert model._n_vocab_draft == 32


@pytest.mark.parametrize("model_class", [DFlashModel, DSparkModel])
def test_dflash_top_level_target_layers(model_class, dflash2_config, capture_parent_hparams, tmp_path):
    """Preserve already-normalized top-level layer IDs in DFlash and DSpark.

    Remove aux_hidden_state_layer_ids and supply target_layer_ids=[0, 8, 16, 24, 32].
    Construct each draft class while capturing its parent call, then check that the IDs enter dflash_config unchanged.
    This prevents applying the auxiliary-layer decrement to IDs that already use the internal numbering.
    """
    del dflash2_config["aux_hidden_state_layer_ids"]
    dflash2_config["target_layer_ids"] = [0, 8, 16, 24, 32]

    model = model_class(tmp_path, hparams=dflash2_config)

    assert model.hparams["dflash_config"]["target_layer_ids"] == [0, 8, 16, 24, 32]


@pytest.mark.parametrize("anchor", [None, False, True])
def test_dspark_nested_config(dflash2_config, capture_parent_hparams, tmp_path, anchor):
    """Keep DSpark's nested export compatible with shared DFlash normalization.

    Remove DFlash2 convolution and selector options, then test omitted, explicit False and explicit True anchor settings.
    Capture the parent call and verify converted layer IDs, the mask token and the full draft vocabulary size.
    An omitted setting must retain DSpark's nested-export default of False; explicit values must be preserved.
    """
    for key in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"):
        del dflash2_config[key]
    if anchor is not None:
        dflash2_config["sample_from_anchor"] = anchor

    model = DSparkModel(tmp_path, hparams=dflash2_config)

    assert model.hparams["dflash_config"]["target_layer_ids"] == [0, 8, 16, 24, 32]
    assert model.hparams["dflash_config"]["mask_token_id"] == 151669
    assert model._n_vocab_draft == 151936
    assert model._sample_from_anchor is (False if anchor is None else anchor)


def test_dflash_metadata_round_trip(dflash2_config, tmp_path):
    """Round-trip DFlash2 metadata through the real constructor, writer and reader.

    Write only a synthetic config.json, construct DFlashModel without replacing its parent, and serialize the metadata.
    Read the file with GGUFReader and check dimensions, RoPE, draft options and the final extraction layer IDs.
    Verify that the fix does not add sample_from_anchor metadata; this file has no weight tensors and is not an inference model.
    """
    (tmp_path / "config.json").write_text(json.dumps(dflash2_config), encoding="utf-8")
    output = tmp_path / "metadata.gguf"
    model = DFlashModel(tmp_path, gguf.LlamaFileType.MOSTLY_F16, output)
    model.set_gguf_parameters()
    model.gguf_writer.write_header_to_file(path=output)
    model.gguf_writer.write_kv_data_to_file()
    model.gguf_writer.close()

    reader = gguf.GGUFReader(output)
    expected = {
        "dflash.block_count": 5,
        "dflash.embedding_length": 2560,
        "dflash.feed_forward_length": 9728,
        "dflash.attention.head_count_kv": 8,
        "dflash.rope.freq_base": 1000000,
        "dflash.block_size": 9,
        "dflash.conv_kernel_size": 2,
        "dflash.conv_group_size": 16,
        "dflash.selector_rank": 256,
        "dflash.selector_top_k": 16,
        "dflash.target_layers": [1, 9, 17, 25, 33],
    }
    for key, value in expected.items():
        assert reader.fields[key].contents() == value
    assert "dflash.sample_from_anchor" not in reader.fields
