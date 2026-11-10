#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
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
        "server_port": 18114,
    },
    "vl_large": {
        "model": "qwen3_vl_4b",
        "models": {"llama": "llama_qwen3_vl_4b", "vllm": "vllm_qwen3_vl_4b"},
        "dataset": "vl_image_large",
        "targets": {"E2EL": 1000.0},
        "server_port": 18114,
    },
}

ACCURACY_DATASETS = {
    "text_10c": "gsm8k_accuracy_30",
    "vl_small": "textvqa_accuracy_30",
    "vl_large": "textvqa_accuracy_30",
}

ACCURACY_DATASETS_FULL = {
    "text_10c": "gsm8k_accuracy_full",
    "vl_small": "textvqa_accuracy_full",
    "vl_large": "textvqa_accuracy_full",
}

VLLM_SERVERS = {
    "text_10c": {
        "port": 18110,
        "gpu_mem_util": 0.5,
        "max_model_len": 8192,
    },
    "vl_small": {
        "port": 18120,
        "gpu_mem_util": 0.9,
        "max_model_len": 8192,
    },
    "vl_large": {
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
    port = int(os.environ.get("PERF_SERVER_PORT") or SCENARIOS[scenario]["server_port"])
    if os.environ.get("LLAMA_REUSE_SERVER") == "1" and wait_server(port, 2):
        return None
    env = os.environ.copy()
    env["PERF_REGRESSION_ROOT"] = str(ROOT)
    env["PERF_SERVER_PORT"] = str(port)
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
    container = os.environ.get("VLLM_CONTAINER")
    if not container:
        raise RuntimeError("VLLM_CONTAINER is not set")
    full = ["docker", "exec"]
    if background:
        full.append("-d")
    full += [container, "bash", "-c", cmd]
    return subprocess.Popen(full, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generated_model_config(engine: str, scenario: str) -> str | Path:
    model_name = SCENARIOS[scenario]["models"][engine]
    server_port = os.environ.get("PERF_SERVER_PORT") or SCENARIOS[scenario]["server_port"]
    source = ROOT / "configs" / "models" / f"{model_name}.py"
    if scenario == "text_10c":
        tokenizer_path = os.environ.get("QWEN3_1_7B_PATH") or os.environ.get("TOKENIZER_PATH")
    else:
        tokenizer_path = os.environ.get("QWEN3_VL_4B_PATH") or os.environ.get("TOKENIZER_PATH")
    if not tokenizer_path:
        return model_name

    text = source.read_text(encoding="utf-8")
    text = re.sub(r"path\s*=\s*\"\.\"", f'path="{tokenizer_path}"', text, count=1)
    text = re.sub(r"host_ip\s*=\s*\"localhost\"", 'host_ip="127.0.0.1"', text, count=1)
    text = re.sub(r"host_port\s*=\s*\d+", f"host_port={server_port}", text, count=1)
    output_name = f"generated_{engine}_{scenario}"
    output = ROOT / "configs" / "models" / f"{output_name}.py"
    output.write_text(text, encoding="utf-8")
    return output_name


def start_vllm_server(scenario: str) -> bool:
    cfg = VLLM_SERVERS[scenario]
    model_key = "VLLM_TEXT_MODEL" if scenario == "text_10c" else "VLLM_VL_MODEL"
    model_path = os.environ.get(model_key) or os.environ.get("VLLM_MODEL")
    if not model_path:
        raise RuntimeError(f"{model_key} or VLLM_MODEL is not set")
    npu_id = os.environ.get("VLLM_NPU_ID", "5")
    _docker_exec("pkill -9 -f vllm 2>/dev/null; sleep 3")
    import time
    time.sleep(5)
    cmd = (
        f'source "${{ASCEND_TOOLKIT_HOME}}/set_env.sh" && '
        f"ASCEND_RT_VISIBLE_DEVICES={npu_id} vllm serve "
        f"{model_path} --port {cfg['port']} "
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


def run_benchmark(
    engine: str,
    scenario: str,
    config_dir: Path,
    dataset: str | None = None,
    work_dir_name: str | None = None,
    mode: str = "perf",
    num_prompts: int | None = None,
) -> None:
    spec = SCENARIOS[scenario]
    dataset = dataset or spec["dataset"]
    work_dir = ROOT / "results" / engine / (work_dir_name or scenario)
    work_dir.mkdir(parents=True, exist_ok=True)
    ais_bench_bin = os.environ.get("AIS_BENCH_BIN") or shutil.which("ais_bench")
    if not ais_bench_bin:
        raise RuntimeError("ais_bench was not found; set AIS_BENCH_BIN or put it on PATH")
    command = [ais_bench_bin]
    if Path(ais_bench_bin).name.startswith("python"):
        ais_bench_module = os.environ.get("AIS_BENCH_MODULE", "ais_bench.benchmark.cli.main")
        command.extend(["-m", ais_bench_module])
    command.extend(
        [
            "--models",
            str(generated_model_config(engine, scenario)),
            "--datasets",
            dataset,
            "--config-dir",
            str(config_dir),
            "--mode",
            mode,
            "--work-dir",
            str(work_dir),
        ]
    )
    env = os.environ.copy()
    python_paths = [str(ROOT / "configs")]
    ais_bench_stubs = os.environ.get("AISBENCH_STUBS")
    if ais_bench_stubs:
        python_paths.insert(0, ais_bench_stubs)
    ais_bench_root = Path(ais_bench_bin).resolve().parents[1]
    python_paths.extend(str(p) for p in sorted(ais_bench_root.glob("lib/python*/site-packages")))
    ais_bench_home = os.environ.get("AIS_BENCH_HOME")
    if ais_bench_home:
        python_paths.insert(0, ais_bench_home)
    env["PYTHONPATH"] = os.pathsep.join(python_paths + [env.get("PYTHONPATH", "")])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    cann_env = os.environ.get("CANN_SET_ENV")
    if not cann_env and os.environ.get("ASCEND_TOOLKIT_HOME"):
        cann_env = str(Path(os.environ["ASCEND_TOOLKIT_HOME"]) / "set_env.sh")
    if cann_env and os.path.exists(cann_env):
        # source CANN env so libascendcl etc. are on LD_LIBRARY_PATH
         import subprocess as _sp
         _r = _sp.run(["bash", "-c", f"source {cann_env} && env"], capture_output=True, text=True)
         for line in _r.stdout.splitlines():
             if "=" in line:
                 k, v = line.split("=", 1)
                 if k in ("LD_LIBRARY_PATH", "PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
                     env[k] = v + (os.pathsep + env[k] if k in env and env[k] else "")
    if num_prompts is not None:
        command.extend(["--num-prompts", str(num_prompts)])
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def run_accuracy_benchmark(
    engine: str,
    scenario: str,
    config_dir: Path,
    num_prompts: int,
    full_dataset: bool,
) -> None:
    dataset_map = ACCURACY_DATASETS_FULL if full_dataset else ACCURACY_DATASETS
    dataset = dataset_map.get(scenario)
    if dataset is None:
        return
    if dataset.startswith("textvqa"):
        stem = "full" if full_dataset else "30"
        subset_path = ROOT / "assets" / "textvqa" / "textvqa_json" / f"textvqa_val_{stem}.jsonl"
        if not subset_path.exists():
            command = [sys.executable, str(SCRIPTS / "prepare_accuracy_assets.py")]
            if full_dataset or num_prompts <= 0:
                command.append("--full")
            subprocess.run(command, check=True, cwd=ROOT)
    if full_dataset or num_prompts <= 0:
        num_prompts = None
    run_benchmark(
        engine,
        scenario,
        config_dir,
        dataset=dataset,
        work_dir_name=f"{scenario}_accuracy",
        mode="all",
        num_prompts=None if full_dataset else num_prompts,
    )


def latest_run(work_dir: Path) -> Path | None:
    runs = [p for p in work_dir.iterdir() if p.is_dir() and len(p.name) >= 12] if work_dir.exists() else []
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


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


def find_accuracy(obj) -> float | None:
    if isinstance(obj, dict):
        value = obj.get("accuracy")
        if isinstance(value, (int, float)):
            return float(value)
        for child in obj.values():
            found = find_accuracy(child)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for child in obj:
            found = find_accuracy(child)
            if found is not None:
                return found
    return None


def collect_accuracy(engine: str, scenario: str) -> dict:
    work_dir = ROOT / "results" / engine / f"{scenario}_accuracy"
    run_dir = latest_run(work_dir)
    if not run_dir:
        return {}
    result: dict = {"run_dir": str(run_dir)}
    for json_file in run_dir.rglob("*.json"):
        try:
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        accuracy = find_accuracy(data)
        if accuracy is not None:
            result["accuracy"] = accuracy
            break
    return result


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

    accuracy_lines = ["| engine | scenario | accuracy percent | run dir |", "|---|---|---:|---|"]
    for engine, scenarios in results.items():
        for scenario, data in scenarios.items():
            if "accuracy" not in data:
                continue
            accuracy_lines.append(
            f"| {engine} | {scenario} | {data['accuracy']:.2f} | {data.get('run_dir', '')} |"
            )
    (ROOT / "results" / "accuracy.md").write_text("\n".join(accuracy_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["llama", "vllm", "both"], default="llama")
    parser.add_argument("--scenario", choices=["all", "text_10c", "vl_small", "vl_large"], default="all")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--accuracy-mode", choices=["both", "off", "only"], default="both")
    parser.add_argument("--accuracy-num-prompts", type=int, default=30)
    parser.add_argument("--accuracy-full", action="store_true")
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
                    if args.accuracy_mode != 'only':
                        run_benchmark(engine, scenario, ROOT / 'configs')
                    if args.accuracy_mode != 'off':
                        run_accuracy_benchmark(
                            engine,
                            scenario,
                            ROOT / 'configs',
                            args.accuracy_num_prompts,
                            args.accuracy_full,
                        )
                stop_vllm_server()
            else:
                for scenario in scenarios:
                    proc = start_server(scenario)
                    if proc is None and os.environ.get("LLAMA_REUSE_SERVER") != "1":
                        print(f'ERROR: failed to start llama server for {scenario}')
                        continue
                    try:
                        if args.accuracy_mode != 'only':
                            run_benchmark(engine, scenario, ROOT / 'configs')
                        if args.accuracy_mode != 'off':
                            run_accuracy_benchmark(
                                engine,
                                scenario,
                                ROOT / 'configs',
                                args.accuracy_num_prompts,
                                args.accuracy_full,
                            )
                    finally:
                        if proc is not None:
                            stop_server(proc)
    results = {}
    for engine in engines:
        results[engine] = {}
        for scenario in scenarios:
            results[engine][scenario] = collect(engine, scenario)
            if args.accuracy_mode != 'off':
                results[engine][scenario].update(collect_accuracy(engine, scenario))
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
