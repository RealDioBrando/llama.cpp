"""Opt-in, offline Qwen3-VL/DFlash regression. Run this module with -n 0."""

import base64
import json
import os
from pathlib import Path
import re
import socket
import struct
import zlib

import numpy as np
import pytest

from utils import ServerProcess

pytestmark = [pytest.mark.slow, pytest.mark.serial]


@pytest.fixture(scope="module", autouse=True)
def load_server_presets():
    # This module uses explicit local models and must not preload/download all presets.
    yield


@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # The module fixture owns its servers across the assertion-only tests below.
    yield


def fixture_path(name, required=True):
    value = os.environ.get(name)
    if not value:
        if required:
            raise pytest.skip.Exception(f"set {name} to enable local VL speculation tests")
        return None
    path = Path(value)
    assert path.is_file(), f"explicit fixture does not exist: {name}={path}"
    return path


def color_image(width, height, reverse=False):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    left, right = ((0, 0, 255), (255, 0, 0)) if reverse else ((255, 0, 0), (0, 0, 255))
    row = b"\x00" + bytes(left)*(width // 2) + bytes(right)*(width - width // 2)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(row*height)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def request_cases():
    a, b = color_image(256, 256), color_image(256, 256, True)
    question = "State the left and right colors of each image, in image order. Be brief."
    text = [
        ("text_zh", "\u7528\u4e2d\u6587\u4ecb\u7ecd\u4e00\u4e0b\u4ec0\u4e48\u662f\u6295\u673a\u89e3\u7801\u3002", [], False, 64),
        ("text_en", "Explain speculative decoding in three short sentences.", [], False, 64),
        ("text_code", "Write a Python function that returns the sum of a list of integers.", [], False, 64),
    ]
    images = [
        ("image_a", question, [a], False, 32), ("image_b", question, [b], False, 32),
        ("shape_wide", question, [color_image(640, 384)], False, 32),
        ("shape_tall", question, [color_image(256, 768, True)], False, 32),
        ("shape_square", question, [a], False, 32),
        ("images_ab", question, [a, b], False, 48), ("images_ba", question, [b, a], False, 48),
        ("mixed_text_before", "Count from one to ten.", [], False, 32),
        ("mixed_image", question, [b], False, 32),
        ("mixed_text_after", "Write the names of five common fruits.", [], False, 32),
        ("cache_a_first", question, [a], True, 32),
        ("cache_a_repeat", question, [a], True, 32),
        ("cache_b_changed", question, [b], True, 32),
    ]
    return text + images


def post(server, endpoint, data):
    result = server.make_request("POST", endpoint, data, timeout=360)
    assert result.status_code == 200, result.body
    return result.body


def counters(server):
    response = server.make_request("GET", "/metrics")
    assert response.status_code == 200
    result = {}
    for name, key in (("drafted", "num_draft_tokens"), ("accepted", "num_accepted_tokens"), ("rounds", "num_drafts")):
        values = re.findall(r"^[^#\s]*spec_decode_" + key + r"_total(?:\{[^\n]*\})?\s+([0-9.eE+-]+)\s*$", response.body, re.M)
        assert len(values) == 1, (name, values)
        result[name] = int(float(values[0]))
    return result


def run_requests(server, cases):
    results = {}
    for name, question, images, cached, limit in cases:
        common = {"temperature": 0, "seed": 1234, "cache_prompt": cached, "return_tokens": True, "id_slot": 0}
        if images:
            content = [{"type": "text", "text": question}]
            content += [{"type": "image_url", "image_url": {"url": url}} for url in images]
            payload = {**common, "messages": [{"role": "user", "content": content}], "max_tokens": limit, "verbose": True}
            endpoint = "/v1/chat/completions"
        else:
            prompt = post(server, "/apply-template", {"messages": [{"role": "user", "content": question}]})["prompt"]
            tokens = post(server, "/tokenize", {"content": prompt, "add_special": True, "parse_special": True})["tokens"]
            payload = {**common, "prompt": tokens, "n_predict": limit}
            endpoint = "/completion"
        before = counters(server)
        response = post(server, endpoint, payload)
        after = counters(server)
        generated = response.get("__verbose", response)
        assert generated.get("tokens"), f"missing generated token IDs: {name}"
        assert generated.get("stop_type"), f"missing stop type: {name}"
        results[name] = {
            "tokens": generated["tokens"], "stop": generated["stop_type"],
            "content": response["choices"][0]["message"]["content"] if images else response["content"],
            "counters": {key: after[key] - before[key] for key in before},
            "timings": response["timings"], "response": response,
        }
    return results


@pytest.fixture(scope="module")
def vl_results(tmp_path_factory, request):
    if getattr(request.config, "workerinput", {}).get("workerid", "master") != "master":
        raise pytest.fail.Exception("run the VL module with -n 0; it shares one accelerator")
    target = fixture_path("LLAMA_TEST_VL_TARGET")
    projector = fixture_path("LLAMA_TEST_VL_MMPROJ")
    draft = fixture_path("LLAMA_TEST_VL_DFLASH")
    legacy = fixture_path("LLAMA_TEST_VL_LEGACY", required=False)
    observer = fixture_path("LLAMA_TEST_VL_ROPE_OBSERVER", required=False)
    batch_observer = fixture_path("LLAMA_TEST_VL_BATCH_OBSERVER", required=False)
    output = Path(os.environ.get("LLAMA_TEST_VL_RESULTS", tmp_path_factory.mktemp("vl-speculation")))
    output.mkdir(parents=True, exist_ok=True)
    env = pytest.MonkeyPatch()
    env.setenv("LLAMA_ARG_MMPROJ", str(projector))
    env.setenv("LLAMA_ARG_FIT", "off")
    env.setenv("LLAMA_ARG_N_GPU_LAYERS_DRAFT", os.environ.get("LLAMA_TEST_VL_NGL", "999"))
    env.setenv("LLAMA_ARG_SPEC_DRAFT_P_MIN", "0")
    env.setenv("LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING", "0")
    if device := os.environ.get("LLAMA_TEST_VL_DEVICE"):
        env.setenv("LLAMA_ARG_DEVICE", device)
    results = {}
    cases = request_cases()
    try:
        phases = [("baseline", None, 3, cases), ("dflash_k3", draft, 3, cases),
                  ("dflash_k1", draft, 1, cases[:3])]
        if legacy:
            phases.append(("legacy", legacy, 3, [cases[0], cases[3], cases[4]]))
        for name, path, length, samples in phases:
            if observer or batch_observer:
                if name == "dflash_k3":
                    graph_dir = output / "runtime-graph"
                    graph_dir.mkdir(exist_ok=True)
                    (graph_dir / "ACTIVE").write_text("Diagnostic capture; not a performance run.\n")
                    env.setenv("LD_PRELOAD", ":".join(str(item) for item in (observer, batch_observer) if item))
                    if observer:
                        env.setenv("ROPE_TRACE_DIR", str(graph_dir))
                        results["graph_dir"] = str(graph_dir)
                    if batch_observer:
                        env.setenv("BATCH_OBSERVER_DIR", str(graph_dir))
                        results["batch_dir"] = str(graph_dir)
                else:
                    env.delenv("LD_PRELOAD", raising=False)
                    env.delenv("ROPE_TRACE_DIR", raising=False)
                    env.delenv("BATCH_OBSERVER_DIR", raising=False)
            server = ServerProcess()
            assert not server.external_server, "DEBUG_EXTERNAL is incompatible with this local-model test"
            server.model_hf_repo = server.model_hf_file = None
            server.model_file = str(target)
            server.model_draft = str(path) if path else None
            server.spec_type = (os.environ.get("LLAMA_TEST_VL_LEGACY_TYPE", "draft-dspark") if name == "legacy"
                                else "draft-dflash") if path else "none"
            server.spec_draft_n_min, server.spec_draft_n_max = 1, length
            server.n_ctx, server.n_batch, server.n_ubatch, server.n_slots = 2048, 128, 128, 1
            server.n_threads, server.n_gpu_layer = 4, int(os.environ.get("LLAMA_TEST_VL_NGL", "999"))
            server.temperature, server.seed, server.fa = 0, 1234, "off"
            server.jinja, server.reasoning, server.offline = True, "off", True
            server.server_metrics, server.cache_ram = True, 512
            server.log_path = str(output / f"{name}.server.log")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                server.server_port = sock.getsockname()[1]
            try:
                server.start(timeout_seconds=240)
                results[name] = run_requests(server, samples)
                (output / f"{name}.json").write_text(json.dumps(results[name], ensure_ascii=False, indent=2))
            finally:
                server.stop()
    finally:
        env.undo()
    return results


def assert_pair(results, names, phase="dflash_k3"):
    for name in names:
        baseline, actual = results["baseline"][name], results[phase][name]
        assert actual["tokens"] == baseline["tokens"], (phase, name)
        assert actual["stop"] == baseline["stop"], (phase, name)
        assert actual["content"].strip()
        counts = actual["counters"]
        assert counts["drafted"] > 0 and counts["rounds"] > 0, (phase, name, counts)
        assert 0 <= counts["accepted"] <= counts["drafted"]
        assert actual["timings"]["draft_n"] == counts["drafted"]
        assert actual["timings"]["draft_n_accepted"] == counts["accepted"]


@pytest.mark.parametrize("phase", ["dflash_k1", "dflash_k3"])
def test_text_exact(vl_results, phase):
    assert_pair(vl_results, ["text_zh", "text_en", "text_code"], phase)


def test_image_exact(vl_results):
    assert_pair(vl_results, ["image_a", "image_b"])
    for name in ("image_a", "image_b"):
        text = vl_results["dflash_k3"][name]["content"].lower()
        colors = re.findall(r"\b(?:red|blue)\b", text)
        assert colors[:2] == (["red", "blue"] if name == "image_a" else ["blue", "red"])
    assert vl_results["dflash_k3"]["image_a"]["tokens"] != vl_results["dflash_k3"]["image_b"]["tokens"]


def test_shape_changes(vl_results):
    assert_pair(vl_results, ["shape_wide", "shape_tall", "shape_square"])


def test_multiple_images(vl_results):
    assert_pair(vl_results, ["images_ab", "images_ba"])
    for name, expected in (("images_ab", ["red", "blue", "blue", "red"]),
                           ("images_ba", ["blue", "red", "red", "blue"])):
        colors = re.findall(r"\b(?:red|blue)\b", vl_results["dflash_k3"][name]["content"].lower())
        assert colors[:4] == expected
    assert vl_results["dflash_k3"]["images_ab"]["tokens"] != vl_results["dflash_k3"]["images_ba"]["tokens"]


def test_cached_image_changes(vl_results):
    assert_pair(vl_results, ["cache_a_first", "cache_a_repeat", "cache_b_changed"])
    for phase in ("baseline", "dflash_k3"):
        assert vl_results[phase]["cache_a_repeat"]["timings"]["cache_n"] > 0
        assert vl_results[phase]["cache_a_first"]["tokens"] == vl_results[phase]["cache_a_repeat"]["tokens"]
        assert vl_results[phase]["cache_b_changed"]["tokens"] != vl_results[phase]["cache_a_repeat"]["tokens"]


def test_text_image_text(vl_results):
    assert_pair(vl_results, ["mixed_text_before", "mixed_image", "mixed_text_after"])


def test_legacy_temporal_draft(vl_results):
    if "legacy" not in vl_results:
        raise pytest.skip.Exception("set LLAMA_TEST_VL_LEGACY to enable the matching legacy draft")
    assert_pair(vl_results, ["text_zh", "image_a", "image_b"], "legacy")


def test_runtime_mrope_graph(vl_results):
    if "graph_dir" not in vl_results:
        raise pytest.skip.Exception("set LLAMA_TEST_VL_ROPE_OBSERVER to enable the Linux graph diagnostic")
    directory = Path(vl_results["graph_dir"])
    axes = np.zeros(64, dtype=np.int32)
    axes[1:60:3], axes[2:60:3] = 1, 2
    errors = {}
    for bucket in ("inject_text", "inject_image", "generate_Q", "generate_K"):
        metadata = json.loads((directory / f"{bucket}.json").read_text())
        assert metadata["model_rope_type"] == metadata["mode"] == 40
        assert metadata["sections"] == [24, 20, 20, 0]
        assert metadata["n_dims"] == 128 and metadata["freq_base"] == 5000000.0
        assert metadata["input_saved_before_rope"]
        assert metadata["freq_scale"] == metadata["attn_factor"] == 1 and metadata["ext_factor"] == 0
        assert not metadata["has_freq_factors"] and metadata["offset"] == 0
        arrays = {}
        for kind in ("input", "output"):
            desc = metadata[kind]
            assert desc["type"] == "f32"
            arrays[kind] = np.ndarray(tuple(desc["ne"])[::-1], dtype=np.float32,
                                      buffer=(directory / f"{bucket}.{kind}.bin").read_bytes(),
                                      strides=tuple(desc["nb"])[::-1])
        count = metadata["input"]["ne"][2]
        positions = np.frombuffer((directory / f"{bucket}.positions.bin").read_bytes(), dtype=np.int32).reshape(4, count)
        angle = positions[axes].T * np.power(5000000.0, -2*np.arange(64)/128.0)
        cosine, sine = np.cos(angle)[None, :, None, :], np.sin(angle)[None, :, None, :]
        left, right = arrays["input"][..., :64], arrays["input"][..., 64:]
        reference = np.concatenate((left*cosine - right*sine, left*sine + right*cosine), axis=-1)
        errors[bucket] = float(np.max(np.abs(arrays["output"] - reference)))
        assert errors[bucket] < 1e-4, (bucket, errors[bucket])
        if bucket == "inject_image":
            assert np.any(positions[0] != positions[1]) or np.any(positions[0] != positions[2])
    (directory / "scalar-check.json").write_text(json.dumps(errors, indent=2))


def test_target_feature_injection(vl_results):
    if "batch_dir" not in vl_results:
        raise pytest.skip.Exception("set LLAMA_TEST_VL_BATCH_OBSERVER to enable the Linux injection diagnostic")
    path = Path(vl_results["batch_dir"]) / "batch-observer.jsonl"
    batches = [json.loads(line) for line in path.read_text().splitlines()]
    assert batches and any(batch["target_image"] for batch in batches)
    assert any(batch["spatial_rows"] > 0 for batch in batches)
    for batch in batches:
        assert batch["features_match"] and batch["positions_match"], batch
        assert batch["draft_window"] == 0 and batch["layers"] == 5 and batch["target_width"] == 2560
