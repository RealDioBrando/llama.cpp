#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

SCENARIOS = {
    "text_10c": {
        "model": "qwen3_1_7b",
        "models": {"llama": "llama_qwen3_1_7b", "vllm": "vllm_qwen3_1_7b"},
        "dataset": "text_4k_2k_10",
        "targets": {"TTFT": 500.0, "TPOT": 14.0},
        "server_port": 18112,
    },
    "vl_small": {
        "model": "qwen3_vl_4b",
        "models": {"llama": "llama_qwen3_vl_4b", "vllm": "vllm_qwen3_vl_4b"},
        "dataset": "vl_image_small",
        "targets": {"E2EL": 30.0},
        "server_port": 18113,
    },
    "vl_large": {
        "model": "qwen3_vl_4b",
        "models": {"llama": "llama_qwen3_vl_4b", "vllm": "vllm_qwen3_vl_4b"},
        "dataset": "vl_image_large",
        "targets": {"E2EL": 1000.0},
        "server_port": 18113,
    },
}

VLLM_CONTAINER = "z00575061-vllm-v0230-cann910"
VLLM_NPU_ID = "5"
VLLM_SERVERS = {
    "text_10c": {
        "model_path": "/home/common/model/qwen3/Qwen3-1.7B",
        "port": 18110,
        "gpu_mem_util": 0.5,
        "max_model_len": 8192,
    },
    "vl_small": {
        "model_path": "/home/common/model/qwen3/Qwen3-VL-4B-Instruct",
        "port": 18120,
        "gpu_mem_util": 0.9,
        "max_model_len": 8192,
    },
    "vl_large": {
        "model_path": "/home/common/model/qwen3/Qwen3-VL-4B-Instruct",
        "port": 18120,
        "gpu_mem_util": 0.9,
        "max_model_len": 24000,
    },
}

VLLM_COMPILATION_CONFIG = (
    '{"cudagraph_mode":"FULL_DECODE_ONLY",'
    '"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10]}'
)


def wait_server(port: int, timeout: int = 120) -> bool:
    import time
    import urllib.request
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_server(scenario: str):
    import signal
    port = SCENARIOS[scenario]["server_port"]
    if os.environ.get("LLAMA_REUSE_SERVER") == "1" and wait_server(port, 2):
        return None
    env = os.environ.copy()
    env["PERF_REGRESSION_ROOT"] = str(ROOT)
    proc = subprocess.Popen(
        ["bash", str(SCRIPTS / "run_llama_server.sh"), scenario],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    if not wait_server(port):
        proc.kill()
        return None
    return proc


def stop_server(proc) -> None:
    import signal
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _docker_exec(cmd: str, background: bool = False) -> subprocess.Popen | None:
    full = ["docker", "exec"]
    if background:
        full.append("-d")
    full += [VLLM_CONTAINER, "bash", "-c", cmd]
    return subprocess.Popen(full, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_vllm_server(scenario: str) -> bool:
    cfg = VLLM_SERVERS[scenario]
    _docker_exec("pkill -9 -f vllm 2>/dev/null; sleep 3")
    import time
    time.sleep(5)
    cmd = (
        f"source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
        f"ASCEND_RT_VISIBLE_DEVICES={VLLM_NPU_ID} vllm serve "
        f"{cfg['model_path']} --port {cfg['port']} "
        f"--enable-prefix-caching --dtype float16 "
        f"--gpu-memory-utilization {cfg['gpu_mem_util']} "
        f"--max-model-len {cfg['max_model_len']} "
        f"--compilation-config '{VLLM_COMPILATION_CONFIG}' "
        f"> /tmp/vllm_{scenario}.log 2>&1"
    )
    _docker_exec(cmd, background=True)
    return wait_server(cfg["port"], timeout=180)


def stop_vllm_server() -> None:
    _docker_exec("pkill -9 -f vllm 2>/dev/null")
    import time
    time.sleep(3)


def run_benchmark(engine: str, scenario: str, config_dir: Path) -> None:
    spec = SCENARIOS[scenario]
    work_dir = ROOT / "results" / engine / scenario
    work_dir.mkdir(parents=True, exist_ok=True)
    ais_bench_bin = os.environ.get("AIS_BENCH_BIN", "/home/z00575061/venv-aisbench/bin/ais_bench")
    command = [
        ais_bench_bin,
        "--models",
        spec["models"][engine],
        "--datasets",
        spec["dataset"],
        "--config-dir",
        str(config_dir),
        "--mode",
        "perf",
        "--work-dir",
        str(work_dir),
    ]
    env = os.environ.copy()
    config_dir = str(ROOT / "configs")
    env["PYTHONPATH"] = config_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    cann_env = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    if os.path.exists(cann_env):
        # source CANN env so libascendcl etc. are on LD_LIBRARY_PATH
         import subprocess as _sp
         _r = _sp.run(["bash", "-c", f"source {cann_env} && env"], capture_output=True, text=True)
         for line in _r.stdout.splitlines():
             if "=" in line:
                 k, v = line.split("=", 1)
                 if k in ("LD_LIBRARY_PATH", "PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
                     env[k] = v + (os.pathsep + env[k] if k in env and env[k] else "")
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def latest_run(work_dir: Path) -> Path | None:
    runs = [p for p in work_dir.iterdir() if p.is_dir() and len(p.name) >= 12] if work_dir.exists() else []
    return max(runs, key=lambda p: p.name) if runs else None


def number(value: str) -> float:
    return float(value.replace("ms", "").replace("token/s", "").strip())


def parse_run(run_dir: Path) -> dict:
    result: dict = {"run_dir": str(run_dir)}
    csv_files = list(run_dir.glob("performances/**/*.csv"))
    json_files = [p for p in run_dir.glob("performances/**/*.json") if not p.name.endswith("_details.json")]
    if csv_files:
        with csv_files[0].open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Stage") != "total":
                    continue
                metric = row.get("Performance Parameters", "")
                value = row.get("Average", "")
                if metric in {"TTFT", "TPOT", "E2EL"} and value:
                    result[metric] = number(value)
                if metric == "OutputTokenThroughput" and value:
                    result["OutputTokenThroughput"] = number(value)
    if json_files:
        with json_files[0].open(encoding="utf-8") as f:
            common = json.load(f)
        for key in {"Total Requests", "Success Requests", "Output Token Throughput", "Total Token Throughput"}:
            if key in common and "total" in common[key]:
                result[key] = common[key]["total"]
    return result


def collect(engine: str, scenario: str) -> dict:
    run_dir = latest_run(ROOT / "results" / engine / scenario)
    if not run_dir:
        return {}
    data = parse_run(run_dir)
    data["targets"] = SCENARIOS[scenario]["targets"]
    return data


def write_summary(results: dict) -> None:
    summary_path = ROOT / "results" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["| engine | scenario | TTFT ms | TPOT ms | E2E ms | output tok/s | requests |", "|---|---|---:|---:|---:|---:|---:|"]
    for engine, scenarios in results.items():
        for scenario, data in scenarios.items():
            lines.append(
                "| {engine} | {scenario} | {ttft} | {tpot} | {e2e} | {tps} | {requests} |".format(
                    engine=engine,
                    scenario=scenario,
                    ttft=data.get("TTFT", "N/A"),
                    tpot=data.get("TPOT", "N/A"),
                    e2e=data.get("E2EL", "N/A"),
                    tps=data.get("Output Token Throughput", "N/A"),
                    requests=data.get("Total Requests", "N/A"),
                )
            )
    (ROOT / "results" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["llama", "vllm", "both"], default="llama")
    parser.add_argument("--scenario", choices=["all", "text_10c", "vl_small", "vl_large"], default="all")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()

    scenarios = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]
    engines = ['llama', 'vllm'] if args.engine == 'both' else [args.engine]

    if not args.collect_only:
        for engine in engines:
            if engine == 'vllm':
                vllm_started = set()
                for scenario in scenarios:
                    cfg = VLLM_SERVERS[scenario]
                    port = cfg['port']
                    # skip if server already running on same port with same config
                    need_restart = port not in vllm_started
                    # vl_small and vl_large share port 18120 but different configs
                    if port in vllm_started and scenario in ('vl_large',):
                        need_restart = True
                        vllm_started.discard(port)
                    if need_restart:
                        if not start_vllm_server(scenario):
                            print(f'ERROR: failed to start vllm server for {scenario}')
                            continue
                        vllm_started.add(port)
                    run_benchmark(engine, scenario, ROOT / 'configs')
                stop_vllm_server()
            else:
                for scenario in scenarios:
                    proc = start_server(scenario)
                    if proc is None and os.environ.get("LLAMA_REUSE_SERVER") != "1":
                        print(f'ERROR: failed to start llama server for {scenario}')
                        continue
                    try:
                        run_benchmark(engine, scenario, ROOT / 'configs')
                    finally:
                        if proc is not None:
                            stop_server(proc)
    results = {engine: {scenario: collect(engine, scenario) for scenario in scenarios} for engine in engines}
    write_summary(results)

    if args.save_baseline:
        baseline_path = ROOT / "results" / "vllm_baseline.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
        for scenario, data in results.get("vllm", {}).items():
            if data:
                baseline[scenario] = data
        baseline_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
