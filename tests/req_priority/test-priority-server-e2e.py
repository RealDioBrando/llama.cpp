#!/usr/bin/env python3
"""请求优先级端到端测试：真实拉起 llama-server，再发 HTTP 请求。

测两层：
  服务端队列      -np 1 时溢出请求停在 server 队列，测按 priority 出队（P1-P7）
  调度器等待队列  -np 给足、KV block 卡紧，请求进 scheduler 的 waiting（P8）

运行（NPU，默认就是 Qwen3-1.7B F16）：
  source /usr/local/Ascend/cann-9.1.0/set_env.sh
  ASCEND_RT_VISIBLE_DEVICES=0 python3 tests/req_priority/test-priority-server-e2e.py \
      --model /home/wenjing/models/qwen3-1.7b-f16.gguf

  换 CPU：加 --ngl 0
  跑 P8：加 --with-scheduler-case
  全部参数：--help

用例一览（策略 -> 预期）：
  P1 优先级排序   blocker 占住唯一 slot，依次排 R0(0),R1(5),R2(-5),R3(-2)
                  -> 首 token 顺序 R2,R3,R0,R1
  P2 FCFS 对照    不开 priority，同样请求顺序
                  -> 首 token 顺序 R0,R1,R2,R3
  P3 同优先级保序 T0,T1,T2 都是 0
                  -> T0,T1,T2
  P4 输出不变     两个 server 用同一 prompt 和 seed
                  -> content 完全相同
  P5 OAI 端点     /v1/completions，V0(0),V1(3),V2(-3)
                  -> 首 token 顺序 V2,V0,V1
  P5b chat 字段   /v1/chat/completions 带 priority
                  -> 200（模型没有 chat template 就 SKIP）
  P6 参数校验     priority = ±1000001、字符串、±1000000
                  -> 越界/类型错 400，边界 200
  P7 非 paged     --enable-priority 但不加 --kv-paged
                  -> 启动日志有警告，请求仍正常 200
  P8 waiting 队列 到达顺序固定 S0,S1,S2；blocker 长 prompt 占 20/32 block，
                  3 个测试请求各需 17 个，只能进 waiting
                  -> 优先级模式 S1(-5),S0(0),S2(5)；FCFS 对照 S0,S1,S2

P8 调参：
  每个请求 block = ceil((prompt_tokens + n_predict) / kvbls)
  进 waiting：ngpub - blocker_block < 请求 block
  能串行：    2 * 请求 block > ngpub
  默认 --sched-ngpub 32 --sched-kvbls 32 -b 256；换模型或改 prompt 要重算。

注意：
  paged 要求 -b == -ub、--cache-ram 0，脚本已自动设置。
  日志留在结尾打印的临时目录。
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/home/wenjing/models/qwen3-1.7b-f16.gguf"
DEFAULT_SERVER = REPO_ROOT / "build" / "bin" / "llama-server"

BLOCKER_N_PREDICT = 512
QUEUED_N_PREDICT = 16
PRIORITY_LIMIT = 1000000

BLOCKER_PROMPT = "Write a very long detailed story about a brave knight and a dragon:"

# 到达顺序 -> (名字, 优先级, prompt)
ORDER_ITEMS = [
    ("R0", 0, "The quick brown fox jumps over the"),
    ("R1", 5, "A small boat sailed across the"),
    ("R2", -5, "In the dense forest there lived a"),
    ("R3", -2, "The old lighthouse stood beside the"),
]
TIE_ITEMS = [
    ("T0", 0, "The first train arrived at the"),
    ("T1", 0, "A curious cat watched the"),
    ("T2", 0, "The market opened early on the"),
]
OAI_ITEMS = [
    ("V0", 0, "The history of the Roman Empire began in"),
    ("V1", 3, "The chemistry of water is defined by"),
    ("V2", -3, "A gentle breeze moved across the"),
]
SCHED_TAIL = (
    "The students wrote down every detail they could see through the telescope while the "
    "teacher explained how the distant star moved slowly across the dark night sky. "
)
# P8 用长 prompt：blocker 准入时占 20/32 个 block，
# 每个测试请求要 17 个，blocker 释放后一次只能放行一个。
SCHED_ITEMS = [
    ("S0", 0, "The astronomy lesson started with an unexpected question. " + SCHED_TAIL * 18),
    ("S1", -5, "A mountain climber reached the summit just before dawn. " + SCHED_TAIL * 18),
    ("S2", 5, "The bakery smelled of fresh bread on a quiet morning. " + SCHED_TAIL * 18),
]
SCHED_BLOCKER_PROMPT = "Continue: " + SCHED_TAIL * 22
SCHED_BLOCKER_N_PREDICT = 200
P4_PROMPT = "The capital of France is"


# --------------------------------------------------------------------------
# server process
# --------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, name: str, args, flags: list[str], n_parallel: int,
                 ngpub: int | None, log_dir: Path,
                 batch: int | None = None, ubatch: int | None = None) -> None:
        self.name = name
        self.port = free_port()
        self.log_path = log_dir / f"server-{name}.log"
        self.proc: subprocess.Popen | None = None
        self._log = None

        # paged warmup 会解码 min(n_batch, n_ctx-1) 个 token，
        # P8 用小 batch，保证 warmup 不撑爆 block 预算
        batch = batch if batch is not None else args.batch
        ubatch = ubatch if ubatch is not None else args.ubatch

        cmd = [
            str(args.server),
            "-m", str(args.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-ngl", str(args.ngl),
            "-sm", args.split_mode,
            "-mg", str(args.main_gpu),
            "-np", str(n_parallel),
            "-c", str(args.ctx),
            "-b", str(batch),
            "-ub", str(ubatch),
            "--cache-ram", "0",
            "--metrics",
            "--no-ui",
            "--threads", str(args.threads),
            "--jinja",
        ]
        if ngpub is not None:
            cmd += ["-ngpub", str(ngpub)]
        cmd += flags
        cmd += args.server_arg
        self.cmd = cmd

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float) -> None:
        self._log = open(self.log_path, "w")
        self._log.write(" ".join(self.cmd) + "\n")
        self._log.flush()
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = self.log_tail()
                raise RuntimeError(
                    f"server '{self.name}' exited with {self.proc.returncode}\n{tail}")

            try:
                res = requests.get(self.base + "/health", timeout=2)
                if res.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)

        tail = self.log_tail()
        raise RuntimeError(
            f"server '{self.name}' did not become healthy in {timeout}s\n{tail}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
                else:
                    self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    else:
                        self.proc.kill()
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
        if self._log is not None:
            self._log.close()
            self._log = None

    def log_tail(self, n: int = 80) -> str:
        try:
            with open(self.log_path, "r", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except OSError:
            return "<no server log>"

    def log_contains(self, needle: str) -> bool:
        try:
            return needle in self.log_path.read_text(errors="replace")
        except OSError:
            return False


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

@dataclass
class StreamResult:
    name: str
    priority: int
    first_token_time: float | None = None
    finish_time: float | None = None
    text: str = ""
    error: str | None = None


def make_payload(endpoint: str, prompt: str, n_predict: int, priority: int,
                 stream: bool, seed: int = 42) -> dict:
    if endpoint == "/v1/chat/completions":
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "temperature": 0.0,
            "seed": seed,
            "stream": stream,
            "priority": priority,
        }
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": seed,
        "stream": stream,
        "priority": priority,
    }


def iter_sse(resp):
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def chunk_text(obj: dict, endpoint: str) -> str:
    if endpoint == "/completion":
        text = obj.get("content")
        return text if isinstance(text, str) else ""
    choices = obj.get("choices") or []
    if not choices:
        return ""
    if endpoint == "/v1/chat/completions":
        delta = choices[0].get("delta") or {}
        return (delta.get("content") or "") + (delta.get("reasoning_content") or "")
    return choices[0].get("text") or ""


def stream_worker(server: Server, endpoint: str, name: str, prompt: str,
                  priority: int, n_predict: int, timeout: float,
                  results: dict[str, StreamResult]) -> None:
    res = StreamResult(name=name, priority=priority)
    try:
        resp = requests.post(
            server.base + endpoint,
            json=make_payload(endpoint, prompt, n_predict, priority, stream=True),
            stream=True,
            timeout=(5, timeout),
        )
        if resp.status_code != 200:
            res.error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            results[name] = res
            return
        with resp:
            for obj in iter_sse(resp):
                text = chunk_text(obj, endpoint)
                if text:
                    res.text += text
                    if res.first_token_time is None:
                        res.first_token_time = time.monotonic()
    except Exception as exc:  # noqa: BLE001 - reported as a test failure
        res.error = repr(exc)
    res.finish_time = time.monotonic()
    results[name] = res


def post_json(server: Server, endpoint: str, payload: dict, timeout: float):
    return requests.post(server.base + endpoint, json=payload, timeout=(5, timeout))


def plain_completion(server: Server, prompt: str, n_predict: int, timeout: float) -> str:
    resp = post_json(server, "/completion", make_payload(
        "/completion", prompt, n_predict, 0, stream=False), timeout)
    resp.raise_for_status()
    return resp.json().get("content", "")


# --------------------------------------------------------------------------
# metrics based synchronization
# --------------------------------------------------------------------------

def fetch_metrics(server: Server, timeout: float = 5) -> dict[str, float] | None:
    try:
        resp = requests.get(server.base + "/metrics", timeout=timeout)
        if resp.status_code != 200:
            return None
    except requests.RequestException:
        return None

    values: dict[str, float] = {}
    for line in resp.text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            values[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return values


def fetch_slots(server: Server, timeout: float = 5):
    try:
        resp = requests.get(server.base + "/slots", timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def processing_slot_count(server: Server):
    slots = fetch_slots(server)
    if slots is None:
        return None
    return sum(1 for slot in slots if slot.get("is_processing"))


def wait_blocker_admitted(server: Server, timeout: float,
                          results: dict[str, StreamResult]) -> bool:
    # n_prompt_tokens_processed > 0 说明 blocker 已准入，
    # 整个 prompt 的 block 已被 cache manager 预留
    deadline = time.time() + timeout
    while time.time() < deadline:
        if "blocker" in results and results["blocker"].error:
            raise RuntimeError(f"blocker request failed: {results['blocker'].error}")
        slots = fetch_slots(server)
        if slots:
            for slot in slots:
                if slot.get("is_processing") and \
                        slot.get("n_prompt_tokens_processed", 0) > 0:
                    return True
        time.sleep(0.01)
    return False


def wait_metric(server: Server, name: str, predicate, timeout: float) -> bool:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        metrics = fetch_metrics(server)
        if metrics is None:
            return False
        last = metrics.get(name, 0.0)
        if predicate(last):
            return True
        time.sleep(0.01)
    raise RuntimeError(f"metric {name} did not reach the expected value, last={last}")


# --------------------------------------------------------------------------
# test cases
# --------------------------------------------------------------------------

def run_order_case(server: Server, endpoint: str, items, metrics_ok: bool,
                   timeout: float) -> tuple[list[str], dict[str, StreamResult]]:
    results: dict[str, StreamResult] = {}

    blocker = threading.Thread(
        target=stream_worker,
        args=(server, endpoint, "blocker", BLOCKER_PROMPT, 0,
              BLOCKER_N_PREDICT, timeout, results),
        daemon=True,
    )
    blocker.start()

    if metrics_ok:
        wait_metric(server, "llamacpp:requests_processing", lambda v: v >= 1, 60)
    else:
        time.sleep(0.5)

    # 一次只放一个请求，并用指标等到它真的进了延迟队列，
    # 这样到达顺序是确定的
    workers = []
    for index, (name, priority, prompt) in enumerate(items):
        worker = threading.Thread(
            target=stream_worker,
            args=(server, endpoint, name, prompt, priority,
                  QUEUED_N_PREDICT, timeout, results),
            daemon=True,
        )
        worker.start()
        workers.append(worker)

        if metrics_ok:
            ok = wait_metric(server, "llamacpp:requests_deferred",
                             lambda v, k=index + 1: v >= k, 30)
            if not ok:
                time.sleep(0.5)
        else:
            time.sleep(0.5)

    for worker in workers:
        worker.join(timeout=timeout)
    blocker.join(timeout=timeout)

    ordered = sorted(
        (results[name] for name, _, _ in items if name in results),
        key=lambda r: (r.first_token_time is None, r.first_token_time or 0.0),
    )
    return [r.name for r in ordered], results


def run_scheduler_case(server: Server, items,
                       timeout: float) -> tuple[list[str], dict[str, StreamResult]]:
    # P8 策略：4 个 server slot 让所有请求直接进调度器。
    # blocker 长 prompt 准入时就占住大部分 block，
    # 3 个测试请求只能进 waiting；
    # blocker 结束释放 block 后，按优先级逐个放行。
    results: dict[str, StreamResult] = {}

    blocker = threading.Thread(
        target=stream_worker,
        args=(server, "/completion", "blocker", SCHED_BLOCKER_PROMPT,
              -PRIORITY_LIMIT, SCHED_BLOCKER_N_PREDICT, timeout, results),
        daemon=True,
    )
    blocker.start()
    if not wait_blocker_admitted(server, 30, results):
        time.sleep(0.3)

    workers = []
    for index, (name, priority, prompt) in enumerate(items):
        worker = threading.Thread(
            target=stream_worker,
            args=(server, "/completion", name, prompt, priority,
                  QUEUED_N_PREDICT, timeout, results),
            daemon=True,
        )
        worker.start()
        workers.append(worker)

        # 固定到达顺序：等 blocker + 已提交请求占满预期 slot 数，再发下一个
        expected = index + 2
        deadline = time.time() + 10
        while time.time() < deadline:
            count = processing_slot_count(server)
            if count is None or count >= expected:
                break
            time.sleep(0.005)

    # 给最后一个请求一点时间进入 waiting
    time.sleep(0.1)

    blocker.join(timeout=timeout)
    for worker in workers:
        worker.join(timeout=timeout)

    ordered = sorted(
        (results[name] for name, _, _ in items if name in results),
        key=lambda r: (r.first_token_time is None, r.first_token_time or 0.0),
    )
    return [r.name for r in ordered], results


class FailFast(Exception):
    pass


class Runner:
    def __init__(self, fail_fast: bool = False) -> None:
        self.failures: list[str] = []
        self.fail_fast = fail_fast

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        if condition:
            print(f"PASS {label}")
        else:
            print(f"FAIL {label} {detail}")
            self.failures.append(label)
            if self.fail_fast:
                raise FailFast(label)

    def skip(self, label: str, reason: str) -> None:
        print(f"SKIP {label}: {reason}")


def describe_order(results: dict[str, StreamResult], order: list[str]) -> str:
    origin = min((r.first_token_time for r in results.values()
                  if r.first_token_time is not None), default=0.0)
    parts = []
    for name in order:
        res = results.get(name)
        if res is None or res.first_token_time is None:
            parts.append(f"{name}=none")
        else:
            parts.append(f"{name}={res.first_token_time - origin:.3f}s")
    return "observed order: " + ", ".join(parts)


def summarize_results(results: dict[str, StreamResult]) -> str:
    errors = [f"{n}: {r.error}" for n, r in results.items() if r.error]
    empty = [n for n, r in results.items() if not r.error and not r.text]
    return f"errors={errors} empty_text={empty}"


def check_stream_results(runner: Runner, label: str,
                         results: dict[str, StreamResult]) -> bool:
    bad = {n: r.error for n, r in results.items() if r.error}
    empty = {n: r.text for n, r in results.items()
             if not r.error and not r.text and n != "blocker"}
    ok = not bad and not empty
    runner.check(ok, label, f"errors={bad} empty_text={list(empty)}")
    return ok


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end request priority tests for llama-server")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--server", default=str(DEFAULT_SERVER))
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--split-mode", default="none")
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--with-scheduler-case", action="store_true",
                        help="also run P8 (needs -ngpub tuned for this model)")
    parser.add_argument("--sched-ngpub", type=int, default=32,
                        help="GPU block budget for P8")
    parser.add_argument("--sched-kvbls", type=int, default=32,
                        help="KV block size for P8")
    parser.add_argument("--server-arg", action="append", default=[],
                        help="extra llama-server argument, repeatable")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop after the first failed case")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Runner(fail_fast=args.fail_fast)

    if not Path(args.model).is_file():
        print(f"ERROR model not found: {args.model}")
        return 2
    if not Path(args.server).is_file():
        print(f"ERROR llama-server not found: {args.server}")
        return 2
    if os.environ.get("ASCEND_TOOLKIT_HOME") is None and \
            os.environ.get("ASCEND_HOME_PATH") is None:
        print("WARN Ascend environment is not sourced; run "
              "'source /usr/local/Ascend/cann-9.1.0/set_env.sh' first")

    log_dir = Path(tempfile.mkdtemp(prefix="req-priority-e2e-"))
    print(f"server logs: {log_dir}")

    prio_content = None
    fcfs_content = None

    try:
        # ---- priority server: P1, P3, P5, P6, P4a ----
        prio = Server("priority", args, ["--kv-paged", "--enable-priority"],
                      n_parallel=1, ngpub=None, log_dir=log_dir)
        prio.start(args.startup_timeout)
        try:
            metrics_ok = fetch_metrics(prio) is not None
            if not metrics_ok:
                print("WARN /metrics unavailable, falling back to sleep based sync")

            # P1 优先级排序：blocker 占住唯一 slot，依次排 R0,R1,R2,R3
            # -> 预期首 token 顺序 R2,R3,R0,R1
            order, results = run_order_case(
                prio, "/completion", ORDER_ITEMS, metrics_ok, args.request_timeout)
            detail = describe_order(results, order) + " | " + summarize_results(results)
            if check_stream_results(runner, "P1 all requests completed", results):
                runner.check(order == ["R2", "R3", "R0", "R1"],
                             "P1 priority order", f"expected R2,R3,R0,R1; {detail}")

            # P3 同优先级保持到达顺序 -> 预期 T0,T1,T2
            order, results = run_order_case(
                prio, "/completion", TIE_ITEMS, metrics_ok, args.request_timeout)
            detail = describe_order(results, order) + " | " + summarize_results(results)
            if check_stream_results(runner, "P3 all requests completed", results):
                runner.check(order == ["T0", "T1", "T2"],
                             "P3 equal priority keeps arrival order",
                             f"expected T0,T1,T2; {detail}")

            # P5 OAI 端点同样生效 -> 预期 V2(-3),V0(0),V1(3)
            order, results = run_order_case(
                prio, "/v1/completions", OAI_ITEMS, metrics_ok, args.request_timeout)
            detail = describe_order(results, order) + " | " + summarize_results(results)
            if check_stream_results(runner, "P5 all requests completed", results):
                runner.check(order == ["V2", "V0", "V1"],
                             "P5 /v1/completions priority order",
                             f"expected V2,V0,V1; {detail}")

            # P6 参数校验：越界/类型错 400，±1000000 边界通过
            for value, expected_status in [
                (PRIORITY_LIMIT + 1, 400),
                (-PRIORITY_LIMIT - 1, 400),
                ("high", 400),
                (PRIORITY_LIMIT, 200),
                (-PRIORITY_LIMIT, 200),
            ]:
                payload = make_payload("/completion", "Hello", 1, 0, stream=False)
                payload["priority"] = value
                resp = post_json(prio, "/completion", payload, args.request_timeout)
                ok = resp.status_code == expected_status
                detail = f"priority={value!r} -> {resp.status_code}, body={resp.text[:200]}"
                if expected_status == 400:
                    ok = ok and "priority" in resp.text.lower()
                runner.check(ok, f"P6 priority={value!r}", detail)

            # P5b chat 端点接受 priority -> 预期 200，无模板则 SKIP
            chat_payload = make_payload("/v1/chat/completions", "Hello", 4, -1,
                                        stream=False)
            chat_resp = post_json(prio, "/v1/chat/completions", chat_payload,
                                  args.request_timeout)
            if chat_resp.status_code == 200:
                runner.check(True, "P5b /v1/chat/completions accepts priority")
            elif chat_resp.status_code == 400 and "chat template" in chat_resp.text.lower():
                runner.skip("P5b /v1/chat/completions accepts priority",
                            "model has no chat template")
            else:
                runner.check(False, "P5b /v1/chat/completions accepts priority",
                             f"status={chat_resp.status_code}, body={chat_resp.text[:200]}")

            prio_content = plain_completion(prio, P4_PROMPT, 16, args.request_timeout)
        finally:
            prio.stop()

        # ---- FCFS baseline server: P2, P4b ----
        fcfs = Server("fcfs", args, ["--kv-paged"],
                      n_parallel=1, ngpub=None, log_dir=log_dir)
        fcfs.start(args.startup_timeout)
        try:
            metrics_ok = fetch_metrics(fcfs) is not None
            # P2 FCFS 对照：不开 priority -> 预期到达顺序 R0,R1,R2,R3
            order, results = run_order_case(
                fcfs, "/completion", ORDER_ITEMS, metrics_ok, args.request_timeout)
            detail = describe_order(results, order) + " | " + summarize_results(results)
            if check_stream_results(runner, "P2 all requests completed", results):
                runner.check(order == ["R0", "R1", "R2", "R3"],
                             "P2 FCFS keeps arrival order",
                             f"expected R0,R1,R2,R3; {detail}")

            fcfs_content = plain_completion(fcfs, P4_PROMPT, 16, args.request_timeout)
        finally:
            fcfs.stop()

        # P4 开不开 priority 输出应一致 -> 预期 content 完全相同
        if prio_content is not None and fcfs_content is not None:
            runner.check(prio_content == fcfs_content,
                         "P4 output independent of priority mode",
                         f"priority={prio_content!r} fcfs={fcfs_content!r}")

        # ---- P7 不加 --kv-paged 时 flag 被忽略 ----
        # 预期：启动日志有警告，请求仍返回 200
        no_paged = Server("no-paged", args, ["--enable-priority"],
                          n_parallel=1, ngpub=None, log_dir=log_dir)
        no_paged.start(args.startup_timeout)
        try:
            runner.check(
                no_paged.log_contains("only applies to the paged scheduler"),
                "P7 warning logged when --kv-paged is missing",
                no_paged.log_tail(20))

            results: dict[str, StreamResult] = {}
            workers = []
            for name, priority, prompt in ORDER_ITEMS[:2]:
                worker = threading.Thread(
                    target=stream_worker,
                    args=(no_paged, "/completion", name, prompt, priority,
                          QUEUED_N_PREDICT, args.request_timeout, results),
                    daemon=True,
                )
                worker.start()
                workers.append(worker)
            for worker in workers:
                worker.join(timeout=args.request_timeout)
            check_stream_results(runner, "P7 server still serves requests", results)
        finally:
            no_paged.stop()

        # ---- P8（可选）：调度器 waiting 队列 ----
        # 到达顺序固定为 S0,S1,S2；优先级模式应变成 S1,S0,S2，FCFS 对照应保持到达顺序。
        if args.with_scheduler_case:
            base_flags = ["--kv-paged",
                          "-kvbls", str(args.sched_kvbls),
                          "--kv-prefix-ckpt-interval", "0",
                          "--slots"]
            sched_cases = [
                ("sched", ["--enable-priority"], ["S1", "S0", "S2"],
                 "P8 waiting 队列按优先级出队"),
                ("sched-fcfs", [], ["S0", "S1", "S2"],
                 "P8 对照 FCFS 保持到达顺序"),
            ]
            for name, extra_flags, expected, label in sched_cases:
                sched = Server(name, args, base_flags + extra_flags,
                               n_parallel=len(SCHED_ITEMS) + 1,
                               ngpub=args.sched_ngpub, log_dir=log_dir,
                               batch=256, ubatch=256)
                sched.start(args.startup_timeout)
                try:
                    if sched.log_contains("paged scheduler warmup failed"):
                        raise RuntimeError(
                            "P8 paged scheduler warmup failed; increase --sched-ngpub "
                            "or reduce -b/-ub")
                    order, results = run_scheduler_case(
                        sched, SCHED_ITEMS, args.request_timeout)
                    detail = describe_order(results, order) + " | " + summarize_results(results)
                    if check_stream_results(runner, f"{label} 请求完成", results):
                        runner.check(order == expected, label,
                                     f"expected {expected}; {detail}")
                finally:
                    sched.stop()
        else:
            runner.skip("P8 waiting 队列优先级顺序",
                        "加 --with-scheduler-case 并调好 --sched-ngpub")
    except FailFast as exc:
        print(f"FAIL-FAST: stopping after first failure: {exc}")
    finally:
        print(f"server logs kept in: {log_dir}")

    if runner.failures:
        print(f"\nFAILED {len(runner.failures)} check(s): {runner.failures}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
