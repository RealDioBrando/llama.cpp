import pytest
from utils import *

# We use a F16 MOE gguf as main model, and q4_0 as draft model

server = ServerPreset.stories15m_moe()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # set default values
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"


@pytest.fixture(autouse=True)
def fixture_create_server():
    return create_server()


def test_with_and_without_draft():
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 16,
        "return_tokens": True,
    }

    server.model_draft = None  # disable draft model
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    tokens_no_draft = res.body["tokens"]
    server.stop()

    # create new server with draft model
    create_server()
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    tokens_draft = res.body["tokens"]

    assert tokens_no_draft == tokens_draft

    server.stop()
    create_server()
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.0] * server.spec_draft_n_max
    server.start()
    res = server.make_request("POST", "/completion", data=request)

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == 0
    assert res.body["tokens"] == tokens_no_draft


def test_different_draft_min_draft_max():
    global server
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
        (4, 12),
        (8, 16),
    ]
    last_content = None
    for draft_min, draft_max in test_values:
        server.stop()
        server.spec_draft_n_min = draft_min
        server.spec_draft_n_max = draft_max
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        })
        assert res.status_code == 200
        if last_content is not None:
            assert last_content == res.body["content"]
        last_content = res.body["content"]


def test_synth_is_deterministic():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.75 ** (i + 1) for i in range(server.spec_draft_n_max)]
    server.start()

    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 32,
    }
    responses = [server.make_request("POST", "/completion", data=request) for _ in range(2)]

    for res in responses:
        assert res.status_code == 200
        assert res.body["timings"]["draft_n"] > 0
    assert responses[0].body["timings"]["draft_n"] == responses[1].body["timings"]["draft_n"]
    assert responses[0].body["timings"]["draft_n_accepted"] == responses[1].body["timings"]["draft_n_accepted"]


def test_synth_ignores_target_tokens():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [1.0] * server.spec_draft_n_max
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 32,
    })

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == res.body["timings"]["draft_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 6,
        "grammar": 'root ::= "a"{5,5}',
    })
    assert res.status_code == 200, res.body

    res = server.make_request("POST", "/completion", data={
        "prompt": "Respond with only: OK",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 64,
        "ignore_eos": True,
    })
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 64
    assert res.body["stop_type"] == "limit"


def test_slot_ctx_not_exceeded():
    global server
    server.n_ctx = 256
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_with_ctx_shift():
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
        })))
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])


@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.skipif(
    not all(os.environ.get(name) for name in ("LLAMA_TEST_VL_TARGET", "LLAMA_TEST_VL_MMPROJ", "LLAMA_TEST_VL_DRAFT")),
    reason="set LLAMA_TEST_VL_TARGET, LLAMA_TEST_VL_MMPROJ and LLAMA_TEST_VL_DRAFT",
)
class TestVLSpeculative:
    """Local Qwen3-VL regression; run with -n 0. LLAMA_TEST_VL_SPEC_TYPE selects DFlash or DSpark."""

    @pytest.fixture(scope="class", autouse=True)
    def load_server_presets(self):
        """Keep the existing preset downloads outside this local-model class."""

    @pytest.fixture(autouse=True)
    def fixture_create_server(self):
        """Use the local server fixture without changing the older test cases."""

    @pytest.fixture
    def vl_case(self, request, monkeypatch, tmp_path):
        import base64
        from pathlib import Path
        import socket
        import struct
        import zlib

        assert not hasattr(request.config, "workerinput"), "run the local VL tests with -n 0"
        paths = [Path(os.environ[name]).resolve() for name in
                 ("LLAMA_TEST_VL_TARGET", "LLAMA_TEST_VL_MMPROJ", "LLAMA_TEST_VL_DRAFT")]
        assert all(path.is_file() for path in paths), paths
        spec_type = os.environ.get("LLAMA_TEST_VL_SPEC_TYPE", "draft-dflash")
        assert spec_type in ("draft-dflash", "draft-dspark"), spec_type
        monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(paths[1]))
        monkeypatch.setenv("LLAMA_ARG_FIT", "off")
        monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_BACKEND_SAMPLING", "0")
        monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS_DRAFT", os.environ.get("N_GPU_LAYERS", "999"))
        local = ServerProcess()
        assert not local.external_server, "DEBUG_EXTERNAL is incompatible with this test"
        local.model_hf_repo = local.model_hf_file = None
        local.model_file = str(paths[0])
        local.n_gpu_layer = int(os.environ.get("N_GPU_LAYERS", "999"))
        local.n_ctx, local.n_batch, local.n_ubatch, local.n_slots = 4096, 128, 128, 1
        local.n_threads, local.seed, local.temperature = 4, 1234, 0
        local.fa, local.ctk, local.ctv = "off", "f16", "f16"
        local.jinja, local.reasoning, local.offline = True, "off", True
        local.cache_ram, local.no_cache_idle_slots = 0, True
        local.spec_draft_n_min, local.spec_draft_n_max = 1, 3
        local.log_path = str(tmp_path / "server.log")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            local.server_port = sock.getsockname()[1]

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

        row = b"\x00" + bytes((255, 0, 0)) * 128 + bytes((0, 0, 255)) * 128
        image = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0))
        image += chunk(b"IDAT", zlib.compress(row * 256)) + chunk(b"IEND", b"")
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "State the left and right colors. Answer only: left=..., right=..."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image).decode("ascii"),
                }},
            ]}],
            "max_tokens": 32, "temperature": 0, "seed": 1234,
            "cache_prompt": False, "return_tokens": True, "verbose": True,
        }
        try:
            yield local, payload, str(paths[2]), spec_type
        finally:
            local.stop()

    def test_end_to_end(self, vl_case):
        local, payload, draft, spec_type = vl_case
        local.model_draft, local.spec_type = draft, spec_type
        local.start(timeout_seconds=240)
        response = local.make_request("POST", "/v1/chat/completions", payload)
        assert response.status_code == 200, response.body
        message = response.body["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert re.findall(r"\b(?:red|blue)\b", message["content"].lower())[:2] == ["red", "blue"]

    def test_matches_target(self, vl_case, tmp_path):
        local, payload, draft, spec_type = vl_case
        responses = []
        for enabled in (False, True):
            local.model_draft = draft if enabled else None
            local.spec_type = spec_type if enabled else "none"
            local.log_path = str(tmp_path / ("draft.log" if enabled else "baseline.log"))
            try:
                local.start(timeout_seconds=240)
                response = local.make_request("POST", "/v1/chat/completions", payload)
                assert response.status_code == 200, response.body
                responses.append(response.body)
            finally:
                local.stop()
        baseline, speculative = [body.get("__verbose", body) for body in responses]
        assert baseline.get("tokens") and speculative.get("tokens"), "missing generated token IDs"
        assert baseline["tokens"] == speculative["tokens"]
        assert baseline.get("stop_type") and baseline["stop_type"] == speculative.get("stop_type")
        timings = responses[1]["timings"]
        assert timings["draft_n"] > 0
        assert 0 <= timings["draft_n_accepted"] <= timings["draft_n"]
