# End-to-end service test for the pluggable KV eviction policy
# (docs/kvcache-manager/prefix-cache-evict-policy.md, section 13).
#
# The server installs its built-in protected-tag policy through the
# test/grayscale hook:
#
#   LLAMA_SERVER_PAGED_EVICT_PROTECT_TAGS=keep llama-server --kv-paged ...
#
# The scenario keeps a long prompt under tag `keep`, floods the (deliberately
# tiny) paged pool with differently tagged requests, then checks that:
#   * the `keep` prefix is still cached and reused on a new request
#     (timings.cache_n >= 64, i.e. reuse across a checkpoint boundary),
#   * /tag_usage?tag=keep still reports cache-resident full blocks,
#   * the pressure really did evict something: at least one pressure tag lost
#     all of its blocks (or, if the picker happened to spare every tag, the
#     owner totals demonstrably dropped).
#
# Run:
#   TAG_TEST_MODEL=/path/to/model.gguf \
#   LLAMA_SERVER_BIN=/path/to/llama-server \
#   pytest -v tests/tag_kvcache_manger/test_paged_evict_policy_e2e.py

import os
import sys

import pytest

# pytest puts this test's directory on sys.path (prepend import mode); add it
# explicitly so a direct `pytest path/to/test_...py` also resolves the import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_tag_kv_server_e2e import LONG_PROMPT, MODEL, ServerProc  # noqa: E402

PROTECTED_TAG = "keep"
N_PRESSURE = 8
PRESSURE_TAGS = [f"p{i}" for i in range(N_PRESSURE)]

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL),
    reason="set TAG_TEST_MODEL to a local GGUF file (default: /home/qwen35-2b-f16.gguf)",
)


@pytest.fixture(scope="module")
def evict_server():
    # Tiny pool: 20 full blocks of 16 tokens (~19 usable after the watermark).
    # On hybrid models the paged budget is otherwise derived from device
    # memory (n_ctx alone does not bound it), so --n-gpu-blocks is what
    # actually makes the pool small. The scheduler warmup prefills 256 tokens
    # (16 blocks), so the pool must stay above that, but one protected prompt
    # (~6 blocks) plus the pressure traffic cannot fit.
    srv = ServerProc(
        ["--kv-paged", "--cache-ram", "0", "--n-gpu-blocks", "20"],
        ctx_len=256,
        n_batch=256,
        n_parallel=2,
        env_extra={"LLAMA_SERVER_PAGED_EVICT_PROTECT_TAGS": PROTECTED_TAG},
    )
    srv.start()
    try:
        # The server logs the installed policy during init. Assert it so a
        # mistyped env var fails with a clear message instead of a confusing
        # cache_n=0 later.
        with open(srv.log_path, "r", errors="replace") as f:
            startup_log = f.read()
        assert "paged eviction policy: protecting prefix-cache tags" in startup_log, (
            "server did not install the protected-tag eviction policy\n" + srv.log_tail()
        )
        yield srv
    finally:
        srv.stop()


def _completion(srv, prompt, tag, n_predict=1):
    r = srv.post("/completion", {
        "prompt": prompt,
        "tag": tag,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,  # the paged scheduler decides reuse itself
    })
    assert r.status_code == 200, (
        f"completion for tag {tag!r} failed: {r.status_code} {r.text}\n"
        + srv.log_tail()
    )
    body = r.json()
    assert "content" in body, f"unexpected completion payload: {body}\n" + srv.log_tail()
    return body


def _tag_usage(srv, tag):
    r = srv.get("/tag_usage", params={"tag": tag}, timeout=120)
    assert r.status_code == 200, (
        f"/tag_usage?tag={tag} failed: {r.status_code} {r.text}\n" + srv.log_tail()
    )
    return r.json()


def _kv_stats(srv):
    r = srv.get("/kv_stats", timeout=120)
    assert r.status_code == 200, f"/kv_stats failed: {r.status_code} {r.text}\n" + srv.log_tail()
    return r.json()


def test_protected_tag_survives_eviction_pressure(evict_server):
    srv = evict_server

    # 1) populate the cache under the protected tag
    _completion(srv, LONG_PROMPT, PROTECTED_TAG)
    keep_before = _tag_usage(srv, PROTECTED_TAG)
    keep_blocks = keep_before["npu"]["full_blocks"]
    assert keep_blocks >= 4, (
        f"expected the 110-token prompt to leave reusable full blocks, got {keep_blocks}\n"
        + srv.log_tail()
    )

    stats = _kv_stats(srv)
    pool_total = stats["npu"]["full"]["total_blocks"]
    assert keep_blocks * (1 + N_PRESSURE) > pool_total, (
        f"pool of {pool_total} full blocks is too large for the pressure scenario "
        f"({keep_blocks} blocks per prompt x {1 + N_PRESSURE} requests); "
        "lower --n-gpu-blocks in the fixture or send more pressure requests\n"
        + srv.log_tail()
    )

    # 2) flood the pool with unique prompts under unique tags; each one adds
    # more cache-resident blocks than the pool can hold, forcing evictions
    blocks_after_request = {}
    for i, tag in enumerate(PRESSURE_TAGS):
        _completion(srv, f"Request {i}: " + LONG_PROMPT, tag)
        blocks_after_request[tag] = _tag_usage(srv, tag)["npu"]["full_blocks"]
        assert blocks_after_request[tag] > 0, (
            f"pressure tag {tag!r} left no full blocks right after its request\n"
            + srv.log_tail()
        )

    # Snapshot the pressure tags here, BEFORE the keep re-run: the re-run itself
    # allocates fresh blocks and can evict pressure blocks, which would make the
    # "eviction happened" check pass for the wrong reason.
    pressure_after_eviction = {tag: _tag_usage(srv, tag) for tag in PRESSURE_TAGS}

    # 3) the protected prefix must still be reusable through a checkpoint
    # boundary; 64 tokens is the first reusable checkpoint on the default
    # --kv-prefix-ckpt-interval 4 / block 16 lattice
    keep_again = _completion(srv, LONG_PROMPT, PROTECTED_TAG)
    cache_n = int((keep_again.get("timings") or {}).get("cache_n", 0))
    assert cache_n >= 64, (
        f"protected prefix was not reused after eviction pressure: cache_n={cache_n} "
        f"(expected >= 64)\n" + srv.log_tail()
    )

    # 4) the protected blocks are still cache-resident and still owned by `keep`
    keep_after = _tag_usage(srv, PROTECTED_TAG)
    assert keep_after["npu"]["full_blocks"] > 0, (
        "protected tag has no cache-resident full blocks after eviction pressure\n"
        + srv.log_tail()
    )

    # 5) eviction actually happened during the pressure phase (snapshot taken
    # before the keep re-run): a pressure tag that owned blocks right after its
    # request must have lost all of them, or at least the per-tag totals dropped
    evicted_tags = [
        tag for tag, usage in pressure_after_eviction.items()
        if usage["npu"]["full_blocks"] == 0
    ]
    if not evicted_tags:
        before_total = sum(blocks_after_request.values())
        after_total = sum(u["npu"]["full_blocks"] for u in pressure_after_eviction.values())
        assert after_total < before_total, (
            "eviction pressure did not evict any pressure block: "
            f"all {N_PRESSURE} tags still own their blocks (before={before_total}, "
            f"after={after_total})\n" + srv.log_tail()
        )
