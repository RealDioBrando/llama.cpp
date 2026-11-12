# End-to-end service tests for the tag -> KV block accounting
# (docs/kvcache-manager/tag-kv-block-usage.md).
#
# These tests drive a real `llama-server --kv-paged` over HTTP and only look
# at the public service surface:
#
#   POST /completion | /v1/completions | /v1/chat/completions  with "tag"
#   GET  /tag_usage            [&tag=<uint64>] [&blocks=1]
#   GET  /kv_stats
#   GET  /slots                (params.tag visibility)
#
# They assert the accounting contract, not internal state:
#   * only reusable blocks are counted (live GPU prefix blocks / GPU
#     checkpoints), so counts are lower bounds of the physical footprint;
#   * owned ids are exactly the counted set and are sorted ascending;
#   * owned is a set-membership count: a block reused by several tags is
#     counted for every one of them, so per-tag counts may double count and
#     sum(tag.full_blocks) can exceed npu.full.used_blocks; `shared` is only
#     the fallback for references whose tag is not in the block's tag set;
#   * full and linear id spaces are independent and cpu.* is always zero;
#   * unknown tags are all-zero, not an error;
#   * a non-paged server reports the endpoints as not supported (501).
#
# Run:
#   TAG_TEST_MODEL=/path/to/model.gguf \
#   LLAMA_SERVER_BIN=/path/to/llama-server \
#   pytest -v tests/tag_kvcache_manger/test_tag_kv_server_e2e.py
#
# Notes:
#   * paged mode requires a single-device full offload (-sm none -mg 0) and
#     rejects the default --cache-ram (pass --cache-ram 0).
#   * on hybrid (full + linear attention) models the prefix cache can only
#     reuse up to the deepest *live linear checkpoint*. With the default
#     --kv-prefix-ckpt-interval 4 and a 16-token block, a prompt must cover at
#     least 64 tokens before any full block becomes reusable. All prompts
#     below are long enough for that, and the linear assertions are skipped
#     for pure full-attention models.

import json
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import threading
import time

import pytest
import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", os.path.join(REPO_ROOT, "build", "bin", "llama-server"))

MODEL = (
    os.environ.get("TAG_TEST_MODEL")
    or os.environ.get("PAGED_TEST_MODEL")
    or "/home/qwen35-2b-f16.gguf"
)

# Numeric owner tags used by the tests (0 = no tag).
TAG_TENANT_A   = 1001
TAG_OWNER1     = 1002
TAG_BORROWER1  = 1003
TAG_OAI_CMPL   = 1004
TAG_OAI_CHAT   = 1005
TAG_SLOT       = 1006
TAG_UNKNOWN    = 999999

# ~110 tokens with the Qwen tokenizer: comfortably past 6 blocks (96 tokens)
# so at least one checkpoint on the default lattice (block 4 = 64 tokens)
# and several reusable full blocks exist.
LONG_PROMPT = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
    "liquor jugs. How vexingly quick daft zebras jump! Sphinx of black quartz, "
    "judge my vow. The five boxing wizards jump quickly. Bright vixens jump; "
    "dozy fowl quack. Jackdaws love my big sphinx of quartz. We promptly judged "
    "antique ivory buckles for the next prize. A very bad quack might jinx zippy "
    "fowls. Sixty zippers were quickly picked from the woven jute bag."
)

# A second, different long prompt used by the sharing test so it does not
# depend on block contents produced by other tests.
SHARED_PROMPT = (
    "Mount Everest rises above the Khumbu valley where climbers gather each "
    "spring. The monsoon season brings heavy snow and sudden storms. Sherpas "
    "carry supplies through the icefall while the weather holds. Every camp "
    "along the route stores food, fuel, and oxygen for the final push. Teams "
    "watch the forecast closely before they leave the last camp."
)

# A third long prompt, only used by the OAI tests: an unseen prompt cannot
# hit the prefix cache, so the request owns every reusable block it creates.
OAI_PROMPT = (
    "Photosynthesis converts light energy into chemical energy inside the "
    "chloroplast. Chlorophyll absorbs red and blue light while reflecting "
    "green wavelengths. Water molecules split to release oxygen as a byproduct. "
    "Carbon dioxide enters through stomata and is fixed into sugars. The "
    "resulting glucose supports growth and is stored as starch. Stomata open "
    "in response to light and close when water becomes scarce. Oxygen escapes "
    "into the atmosphere while carbon dioxide diffuses inward. Enzymes in the "
    "stroma assemble triose phosphates that later form sucrose and cellulose."
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL),
    reason="set TAG_TEST_MODEL to a local GGUF file (default: /home/qwen35-2b-f16.gguf)",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ascend_env_wrapper(cmd):
    """Source the CANN environment when it is available, then exec `cmd`."""
    toolkit = os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/cann-9.1.0")
    set_env = os.path.join(toolkit, "set_env.sh")
    if os.path.exists(set_env):
        quoted = " ".join(shlex.quote(c) for c in cmd)
        return ["bash", "-c", f"source {shlex.quote(set_env)} >/dev/null 2>&1; exec {quoted}"]
    return cmd


class ServerProc:
    """One llama-server child process with helpers for HTTP + lifecycle."""

    def __init__(self, extra_args, wait_timeout=600, ctx_len=8192, n_batch=2048,
                 n_parallel=4, env_extra=None):
        self.extra_args = list(extra_args)
        self.wait_timeout = wait_timeout
        self.ctx_len = ctx_len
        self.n_batch = n_batch
        self.n_parallel = n_parallel
        # merged over the inherited environment when starting the process
        self.env_extra = dict(env_extra or {})
        self.port = _free_port()
        self.proc = None
        self.log_path = None

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def start(self):
        assert os.path.exists(SERVER_BIN), f"llama-server not found: {SERVER_BIN}"
        self.log_path = tempfile.mkstemp(prefix="tag-kv-e2e-", suffix=".log")[1]
        cmd = [
            SERVER_BIN,
            "-m", MODEL,
            "-ngl", "99",
            "-sm", "none",
            "-mg", "0",
            "-b", str(self.n_batch),
            "-ub", str(self.n_batch),
            "-np", str(self.n_parallel),
            "-c", str(self.ctx_len),
            "-kvbls", "16",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            *self.extra_args,
        ]
        log = open(self.log_path, "wb")
        env = None
        if self.env_extra:
            env = dict(os.environ)
            env.update(self.env_extra)
        self.proc = subprocess.Popen(
            _ascend_env_wrapper(cmd),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        log.close()

        deadline = time.time() + self.wait_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with {self.proc.returncode} during startup\n"
                    + self.log_tail()
                )
            try:
                r = requests.get(self.url("/health"), timeout=3)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return self
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError(f"llama-server did not become ready in {self.wait_timeout}s\n" + self.log_tail())

    def log_tail(self, n=80):
        if not self.log_path or not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, "r", errors="replace") as f:
            return "".join(f.readlines()[-n:])

    def stop(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=10)

    # --- HTTP helpers -----------------------------------------------------

    def get(self, path, params=None, timeout=60):
        return requests.get(self.url(path), params=params, timeout=timeout)

    def post(self, path, payload, timeout=900):
        return requests.post(self.url(path), json=payload, timeout=timeout)


@pytest.fixture(scope="session")
def paged_server():
    srv = ServerProc(["--kv-paged", "--cache-ram", "0", "--slots"])
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture(scope="module")
def default_server():
    # A server without --kv-paged: the accounting endpoints must report
    # "not supported" instead of failing or returning bogus data.
    srv = ServerProc([])
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


# --- assertion helpers ----------------------------------------------------


def tag_usage(srv, tag, timeout=120):
    r = srv.get("/tag_usage", params={"tag": tag}, timeout=timeout)
    assert r.status_code == 200, r.text
    return r.json()


def tag_listing(srv, blocks=False, timeout=120):
    params = {"blocks": "1"} if blocks else None
    r = srv.get("/tag_usage", params=params, timeout=timeout)
    assert r.status_code == 200, r.text
    return r.json()


def kv_stats(srv, timeout=60):
    r = srv.get("/kv_stats", timeout=timeout)
    assert r.status_code == 200, r.text
    return r.json()


def completion(srv, prompt=LONG_PROMPT, tag=None, n_predict=2):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,  # the paged scheduler decides reuse itself
    }
    if tag is not None:
        payload["tag"] = tag
    r = srv.post("/completion", payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "content" in body
    return body


STATS_FIELDS = {
    "total_blocks",
    "used_blocks",
    "free_blocks",
    "cached_blocks",
    "allocatable_blocks",
}


def assert_stats_consistent(stats):
    npu = stats["npu"]
    for pool in ("full", "linear"):
        b = npu[pool]
        assert set(b) == STATS_FIELDS
        assert b["total_blocks"] >= 0
        assert b["used_blocks"] >= 0
        assert b["free_blocks"] >= 0
        assert b["cached_blocks"] >= 0
        # used = in-use + cache-resident; free = free and not cache-resident
        assert b["used_blocks"] + b["free_blocks"] == b["total_blocks"]
        # allocation may also consume cache-resident blocks
        assert b["free_blocks"] + b["cached_blocks"] == b["allocatable_blocks"]
        assert b["cached_blocks"] <= b["used_blocks"]
        assert b["allocatable_blocks"] <= b["total_blocks"]
    cpu = stats["cpu"]
    for pool in ("full", "linear"):
        assert cpu[pool] == {
            "total_blocks": 0,
            "used_blocks": 0,
            "free_blocks": 0,
            "cached_blocks": 0,
            "allocatable_blocks": 0,
        }
    assert npu["full"]["total_blocks"] > 0


def assert_usage_shape(usage, tag):
    assert usage["tag"] == tag
    npu = usage["npu"]
    assert set(npu) == {"full_blocks", "linear_blocks"}
    assert npu["full_blocks"] >= 0 and npu["linear_blocks"] >= 0
    assert usage["cpu"] == {
        "full_blocks": 0,
        "linear_blocks": 0,
    }
    blocks = usage["blocks"]
    assert blocks["full"]["shared"] == [] or all(isinstance(i, int) for i in blocks["full"]["shared"])
    assert blocks["linear"]["shared"] == []


def assert_sorted_ids(ids):
    assert isinstance(ids, list)
    assert all(isinstance(i, int) and i >= 0 for i in ids)
    assert ids == sorted(ids), f"ids must be sorted ascending: {ids}"
    assert len(ids) == len(set(ids)), f"ids must be unique: {ids}"


def wait_until(predicate, timeout, interval=1.0, what="condition"):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {what}; last value: {last!r}")


# --- tests ----------------------------------------------------------------


def test_kv_stats_initial_consistency(paged_server):
    stats = kv_stats(paged_server)
    assert_stats_consistent(stats)
    # warmup releases everything it allocates
    assert stats["npu"]["full"]["used_blocks"] == 0


def test_tag_counts_owned_reusable_blocks(paged_server):
    tag = TAG_TENANT_A
    completion(paged_server, tag=tag, n_predict=2)

    usage = tag_usage(paged_server, tag)
    assert_usage_shape(usage, tag)

    owned = usage["blocks"]["full"]["owned"]
    assert_sorted_ids(owned)
    # 110-token prompt: at least 6 full blocks are registered
    assert usage["npu"]["full_blocks"] >= 4
    assert usage["npu"]["full_blocks"] == len(owned)
    # nothing was shared with this tag
    assert usage["blocks"]["full"]["shared"] == []
    # a finished request keeps its blocks cache-resident: the counts stay
    assert len(owned) > 0

    stats = kv_stats(paged_server)
    assert_stats_consistent(stats)
    assert stats["npu"]["full"]["used_blocks"] >= usage["npu"]["full_blocks"]


def test_tag_linear_gpu_checkpoint(paged_server):
    stats = kv_stats(paged_server)
    if stats["npu"]["linear"]["total_blocks"] == 0:
        pytest.skip("model has no linear-attention layers")

    usage = tag_usage(paged_server, TAG_TENANT_A)
    linear = usage["blocks"]["linear"]["owned"]
    assert_sorted_ids(linear)
    assert usage["npu"]["linear_blocks"] >= 1
    assert usage["npu"]["linear_blocks"] == len(linear)
    # linear and full id spaces are independent
    full_ids = set(usage["blocks"]["full"]["owned"])
    assert not (full_ids & set(linear))


def test_unknown_tag_is_zero(paged_server):
    usage = tag_usage(paged_server, TAG_UNKNOWN)
    assert_usage_shape(usage, TAG_UNKNOWN)
    assert usage["npu"]["full_blocks"] == 0
    assert usage["npu"]["linear_blocks"] == 0
    assert usage["blocks"]["full"]["owned"] == []
    assert usage["blocks"]["full"]["shared"] == []
    assert usage["blocks"]["linear"]["owned"] == []


def test_tag_listing_and_blocks_flag(paged_server):
    listing = tag_listing(paged_server)
    assert "tags" in listing
    assert str(TAG_TENANT_A) in listing["tags"]

    entry = listing["tags"][str(TAG_TENANT_A)]
    assert set(entry) == {"npu", "cpu"}
    assert "blocks" not in entry, "the listing must not carry ids unless ?blocks=1"

    listing_blocks = tag_listing(paged_server, blocks=True)
    entry_blocks = listing_blocks["tags"][str(TAG_TENANT_A)]
    assert "blocks" in entry_blocks

    single = tag_usage(paged_server, TAG_TENANT_A)
    assert entry_blocks["blocks"] == single["blocks"]
    assert entry_blocks["npu"] == single["npu"]

    # Block sets may overlap across tags (membership accounting), but no single
    # tag can count more blocks than the pool holds.
    stats = kv_stats(paged_server)
    assert_stats_consistent(stats)
    for entry in listing_blocks["tags"].values():
        assert entry["npu"]["full_blocks"] <= stats["npu"]["full"]["used_blocks"]


def test_shared_prefix_is_counted_for_both_tags(paged_server):
    owner_tag = TAG_OWNER1
    borrower_tag = TAG_BORROWER1

    # the owner runs to completion first, leaving reusable blocks and a
    # checkpoint (block 4 on the default lattice) behind
    completion(paged_server, prompt=SHARED_PROMPT, tag=owner_tag, n_predict=2)
    owner_before = tag_usage(paged_server, owner_tag)
    owner_ids = set(owner_before["blocks"]["full"]["owned"])
    assert len(owner_ids) >= 4
    assert owner_before["npu"]["linear_blocks"] >= 1 or \
        kv_stats(paged_server)["npu"]["linear"]["total_blocks"] == 0

    result = {}

    def run_borrower():
        try:
            result["body"] = completion(paged_server, prompt=SHARED_PROMPT, tag=borrower_tag, n_predict=8)
        except Exception as exc:  # surfaced by the main thread below
            result["error"] = exc

    thread = threading.Thread(target=run_borrower, daemon=True)
    thread.start()

    def borrower_reuses_owner_blocks():
        borrower = tag_usage(paged_server, borrower_tag)
        # Multi-tag model: acquire_blocks()/add_ref_blocks() union the
        # borrower's tag into the block set, so a reused owner block is
        # attributed to both tags rather than reported as borrower "shared".
        if set(borrower["blocks"]["full"]["owned"]) & owner_ids:
            return borrower
        return None

    try:
        borrower = wait_until(borrower_reuses_owner_blocks, timeout=180,
                              what="the borrower to reuse the owner's prefix blocks")
    finally:
        thread.join(timeout=600)

    if "error" in result:
        raise result["error"]

    owned = borrower["blocks"]["full"]["owned"]
    shared = borrower["blocks"]["full"]["shared"]
    assert_sorted_ids(owned)
    assert_sorted_ids(shared)
    reused = set(owned) & owner_ids
    assert reused, "the borrower must reuse the owner's reusable blocks"
    # Normal acquisition writes the tag into the block set, so the fallback
    # shared list stays empty.
    assert shared == []
    assert borrower["npu"]["full_blocks"] == len(owned)

    owner_now = set(tag_usage(paged_server, owner_tag)["blocks"]["full"]["owned"])
    assert reused.issubset(owner_now), "reused blocks stay attributed to the owner too"

    stats = kv_stats(paged_server)
    assert_stats_consistent(stats)
    for entry in tag_listing(paged_server)["tags"].values():
        assert entry["npu"]["full_blocks"] <= stats["npu"]["full"]["used_blocks"]

    # Once the borrower finishes its block_table is cleared. The tag set is a
    # history union while the blocks stay cache-resident, so the borrower's
    # attribution may persist (unlike the old shared-only accounting); the
    # owner must keep its own attribution.
    after = tag_usage(paged_server, borrower_tag)
    assert after["npu"]["full_blocks"] == len(after["blocks"]["full"]["owned"])
    owner_after = set(tag_usage(paged_server, owner_tag)["blocks"]["full"]["owned"])
    assert reused.issubset(owner_after)


def test_oai_endpoints_accept_tag(paged_server):
    cmpl_tag = TAG_OAI_CMPL
    chat_tag = TAG_OAI_CHAT
    r = paged_server.post("/v1/completions", {
        "prompt": OAI_PROMPT,
        "max_tokens": 2,
        "temperature": 0.0,
        "tag": cmpl_tag,
    })
    assert r.status_code == 200, r.text
    cmpl_usage = tag_usage(paged_server, cmpl_tag)
    assert cmpl_usage["npu"]["full_blocks"] >= 4
    assert cmpl_usage["npu"]["full_blocks"] == len(cmpl_usage["blocks"]["full"]["owned"])

    r = paged_server.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": OAI_PROMPT}],
        "max_tokens": 2,
        "temperature": 0.0,
        "tag": chat_tag,
    })
    assert r.status_code == 200, r.text
    chat_usage = tag_usage(paged_server, chat_tag)
    assert chat_usage["npu"]["full_blocks"] >= 1
    assert chat_usage["npu"]["full_blocks"] == len(chat_usage["blocks"]["full"]["owned"])


def test_slots_expose_request_tag(paged_server):
    tag = TAG_SLOT
    result = {}

    def run():
        try:
            result["body"] = completion(paged_server, prompt=SHARED_PROMPT, tag=tag, n_predict=24)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def slot_with_tag():
        r = paged_server.get("/slots", timeout=30)
        assert r.status_code == 200, r.text
        for slot in r.json():
            if slot.get("params", {}).get("tag") == tag:
                return slot
        return None

    try:
        wait_until(slot_with_tag, timeout=120, what="/slots to expose params.tag")
    finally:
        thread.join(timeout=600)

    if "error" in result:
        raise result["error"]


def test_non_uint64_tag_is_rejected(paged_server):
    # A string tag is no longer accepted: the tag is a uint64 identifier.
    r = paged_server.post("/completion", {
        "prompt": "hello",
        "n_predict": 1,
        "tag": "not-a-number",
    })
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "tag" in body["error"]["message"].lower()

    # The /tag_usage query parameter must also parse as uint64.
    r = paged_server.get("/tag_usage", params={"tag": "not-a-number"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_kv_stats_consistency_after_activity(paged_server):
    stats = kv_stats(paged_server)
    assert_stats_consistent(stats)
    assert stats["npu"]["full"]["used_blocks"] > 0


def test_non_paged_server_reports_not_supported(default_server):
    r = default_server.get("/kv_stats")
    assert r.status_code == 501, r.text
    assert r.json()["error"]["type"] == "not_supported_error"

    r = default_server.get("/tag_usage", params={"tag": "1"})
    assert r.status_code == 501, r.text
    assert r.json()["error"]["type"] == "not_supported_error"
