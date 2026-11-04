import os
import pytest
from utils import *

# Functional acceptance for llama-server with the paged scheduler (--kv-paged).
#
# The paged path requires a single-device full offload and is validated on
# Ascend 310P. The model is provided via the environment so CI stays unaffected:
#
#   PAGED_TEST_MODEL=/path/to/model.gguf \
#   ASCEND_RT_VISIBLE_DEVICES=1 \
#   ./tests.sh unit/test_paged_server.py

MODEL_PATH = os.environ.get("PAGED_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.exists(MODEL_PATH),
    reason="set PAGED_TEST_MODEL to a local GGUF file to run the paged server tests",
)


def create_server(**overrides):
    server = ServerProcess()
    server.model_file = MODEL_PATH
    server.model_hf_repo = None
    server.model_hf_file = None
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
    for key, value in overrides.items():
        setattr(server, key, value)
    return server


@pytest.fixture()
def paged_server():
    server = create_server()
    yield server
    server.stop()


def test_completion_matches_default_scheduler(paged_server):
    # greedy equivalence: the paged scheduler must produce the same tokens as
    # the default slot scheduler for a fixed prompt and seed
    request = {
        "prompt": "Once upon a time there was a lovely",
        "n_predict": 16,
        "temperature": 0.0,
    }

    paged_server.start()
    res = paged_server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    paged_content = res.body["content"]
    assert len(paged_content) > 0
    paged_server.stop()

    default_server = create_server(kv_paged=False)
    default_server.start()
    res = default_server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["content"] == paged_content
    default_server.stop()


def test_completion_deterministic_and_stream(paged_server):
    paged_server.start()
    request = {
        "prompt": "The capital of France is",
        "n_predict": 8,
        "temperature": 0.0,
    }
    first = paged_server.make_request("POST", "/completion", data=request)
    assert first.status_code == 200
    second = paged_server.make_request("POST", "/completion", data=request)
    assert second.status_code == 200
    assert first.body["content"] == second.body["content"]

    chunks = list(paged_server.make_stream_request("POST", "/completion", data={
        **request,
        "stream": True,
    }))
    assert len(chunks) >= 2
    assert chunks[-1]["stop"] is True
    assert chunks[-1]["stop_type"] in ("limit", "eos", "word")


def test_stop_string_and_n_predict(paged_server):
    paged_server.start()
    res = paged_server.make_request("POST", "/completion", data={
        "prompt": "Write the word banana then stop: banana",
        "n_predict": 32,
        "temperature": 0.0,
        "stop": ["banana"],
    })
    assert res.status_code == 200
    assert res.body["stop_type"] == "word"
    assert res.body["stopping_word"] == "banana"

    res = paged_server.make_request("POST", "/completion", data={
        "prompt": "hello",
        "n_predict": 5,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert res.body["tokens_predicted"] == 5


def test_prefix_cache_hit(paged_server):
    paged_server.start()
    request = {
        "prompt": "The history of the Roman Empire is long and complex. " * 40
        + "\n\nQuestion: which city was the capital? Answer:",
        "n_predict": 8,
        "temperature": 0.0,
    }
    res = paged_server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    res = paged_server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0


def test_concurrent_requests_queue(paged_server):
    paged_server.start()

    def call(index):
        return paged_server.make_request("POST", "/completion", data={
            "prompt": f"Request number {index} says: the sky is",
            "n_predict": 8,
            "temperature": 0.0,
        })

    # 8 requests on 4 slots: the overflow must queue and finish after slots free up
    results = parallel_function_calls([(call, (i,)) for i in range(8)])
    for res in results:
        assert res.status_code == 200
        assert "blue" in res.body["content"]


def test_client_disconnect_releases_slot(paged_server):
    paged_server.start()
    # open a stream and drop it mid-generation, like an aborted client
    response = requests.post(
        f"http://{paged_server.server_host}:{paged_server.server_port}/completion",
        json={
            "prompt": "Tell me a very long detailed story about robots.",
            "n_predict": 256,
            "stream": True,
            "temperature": 0.8,
            "seed": 1,
        },
        stream=True,
        timeout=30,
    )
    for _, _line in zip(range(3), response.iter_lines()):
        pass
    response.close()

    # allow the scheduler to reap the cancelled request and its KV blocks
    time.sleep(5)

    res = paged_server.make_request("POST", "/completion", data={
        "prompt": "after cancel say ok:",
        "n_predict": 4,
        "temperature": 0.0,
    })
    assert res.status_code == 200

    slots = paged_server.make_request("GET", "/slots")
    assert slots.status_code == 200
    assert all(not slot["is_processing"] for slot in slots.body)


def test_chat_completion_and_tool_call(paged_server):
    paged_server.start()
    res = paged_server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 16,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    message = res.body["choices"][0]["message"]
    assert message["content"] or message.get("reasoning_content")

    res = paged_server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "What is the weather in Paris today? Use the provided function."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": "auto",
        "max_tokens": 120,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    assert res.body["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = res.body["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert "Paris" in tool_call["function"]["arguments"]


def test_parallel_sampling_n2(paged_server):
    paged_server.start()
    res = paged_server.make_request("POST", "/v1/completions", data={
        "prompt": "My favorite color is",
        "n": 2,
        "max_tokens": 12,
        "temperature": 1.0,
        "seed": 42,
    })
    assert res.status_code == 200
    assert len(res.body["choices"]) == 2


def test_degraded_endpoints_report_clear_errors(paged_server):
    paged_server.start()
    res = paged_server.make_request("POST", "/infill", data={
        "prompt": "a",
        "input_prefix": "b",
        "input_suffix": "c",
    })
    assert res.status_code == 501
    assert "--kv-paged" in res.body["error"]["message"]

    res = paged_server.make_request("POST", "/v1/embeddings", data={"input": "hello"})
    assert res.status_code == 501


def test_batch_ubatch_mismatch_rejected():
    # the paged scheduler requires -b == -ub and must fail fast at startup
    server = create_server(n_ubatch=1024)
    with pytest.raises(RuntimeError):
        server.start()
    server.stop()
