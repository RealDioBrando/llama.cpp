import base64
import os
import struct
import zlib

import pytest
from utils import *

# Functional acceptance for llama-server with the paged scheduler (--kv-paged).
#
# The paged path requires a single-device full offload and is validated on
# Ascend 310P. The models are provided via the environment so CI stays
# unaffected:
#
#   PAGED_TEST_MODEL=/path/to/model.gguf \
#   PAGED_TEST_MMPROJ=/path/to/mmproj.gguf \
#   ASCEND_RT_VISIBLE_DEVICES=1 \
#   ./tests.sh unit/test_paged_server.py
#
# The multimodal tests (mm_*) additionally require PAGED_TEST_MMPROJ to point
# at an mmproj GGUF matching a Qwen3-VL text model.

MODEL_PATH = os.environ.get("PAGED_TEST_MODEL")
MMPROJ_PATH = os.environ.get("PAGED_TEST_MMPROJ")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set PAGED_TEST_MODEL to a local GGUF file to run the paged server tests",
)

mm_required = pytest.mark.skipif(
    not MMPROJ_PATH or not os.path.exists(MMPROJ_PATH),
    reason="set PAGED_TEST_MMPROJ to a matching mmproj GGUF to run the multimodal tests",
)

# 64x64 solid red PNG, the test image for the multimodal cases
def _make_test_png(width=64, height=64, rgb=(200, 30, 30)):
    raw = b""
    for _ in range(height):
        raw += b"\x00" + bytes(rgb) * width

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


TEST_PNG_B64 = _make_test_png()


def create_server(**overrides):
    server = ServerProcess()
    server.model_file = MODEL_PATH
    server.model_hf_repo = None
    server.model_hf_file = None
    server.mmproj = MMPROJ_PATH
    server.kv_paged = True
    server.cache_ram = 0  # paged mode rejects the cache-ram default
    server.n_batch = 2048
    server.n_ubatch = 2048
    server.n_ctx = 4096
    server.n_slots = 4
    server.n_gpu_layer = 99
    server.server_slots = True
    server.jinja = True
    server.temperature = 0.0
    server.seed = 42
    server.log_path = "/tmp/paged_mm_server.log"
    for key, value in overrides.items():
        setattr(server, key, value)
    return server


@pytest.fixture()
def paged_server():
    server = create_server()
    yield server
    server.stop()


def mm_chat_request(png_b64, prompt, max_tokens):
    return {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_b64}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


@mm_required
def test_mm_completion_matches_default_scheduler():
    # greedy equivalence: the paged path must produce the same tokens as the
    # default slot scheduler for an image + text request
    request = mm_chat_request(TEST_PNG_B64, "Describe this image in one sentence.", 48)

    paged = create_server()
    paged.start()
    res = paged.make_request("POST", "/v1/chat/completions", data=request)
    assert res.status_code == 200
    paged_content = res.body["choices"][0]["message"]["content"]
    assert len(paged_content) > 0
    paged.stop()

    default = create_server(kv_paged=False)
    default.start()
    res = default.make_request("POST", "/v1/chat/completions", data=request)
    assert res.status_code == 200
    default_content = res.body["choices"][0]["message"]["content"]
    default.stop()

    assert default_content == paged_content


@mm_required
def test_mm_text_after_image(paged_server):
    # the text rows after the media segment must carry the image context
    paged_server.start()
    res = paged_server.make_request("POST", "/v1/chat/completions", data=mm_chat_request(
        TEST_PNG_B64, "What is the dominant color of this image? Answer with one word.", 16))
    assert res.status_code == 200
    content = res.body["choices"][0]["message"]["content"]
    assert len(content) > 0
    assert "red" in content.lower()


@mm_required
def test_mm_prompt_tokens_tracked(paged_server):
    # the media rows must be accounted as prompt tokens
    paged_server.start()
    res = paged_server.make_request("POST", "/v1/chat/completions", data=mm_chat_request(
        TEST_PNG_B64, "Describe this image.", 16))
    assert res.status_code == 200
    assert res.body["usage"]["prompt_tokens"] > 16
