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
from dataclasses import dataclass

from test_config import config_value, load_config, resolved_path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG_PATH = ""
CONFIG = {}
SCENARIOS = {}
ACCURACY_DATASETS = {}
ACCURACY_DATASETS_FULL = {}
VLLM_SERVERS = {}


def apply_config(config: dict, path: str) -> None:
    global CONFIG_PATH, CONFIG, SCENARIOS, ACCURACY_DATASETS, ACCURACY_DATASETS_FULL, VLLM_SERVERS
    CONFIG_PATH = str(Path(path).expanduser().resolve())
    CONFIG = config
    SCENARIOS = {}
    for scenario_name, scenario_spec in CONFIG["scenarios"].items():
        spec = dict(scenario_spec)
        spec["models"] = scenario_spec["model_configs"]
        spec["dataset"] = scenario_spec["datasets"]["perf"]
        spec["server_port"] = CONFIG["servers"]["llama"]["scenarios"][scenario_name]["port"]
        SCENARIOS[scenario_name] = spec

    ACCURACY_DATASETS = {
        name: spec["datasets"]["accuracy_30"] for name, spec in SCENARIOS.items()
    }
    ACCURACY_DATASETS_FULL = {
        name: spec["datasets"]["accuracy_full"] for name, spec in SCENARIOS.items()
    }
    VLLM_SERVERS = {
        name: {**CONFIG["servers"]["vllm"]["defaults"], **spec}
        for name, spec in CONFIG["servers"]["vllm"]["scenarios"].items()
    }


@dataclass(frozen=True)
class ModelPaths:
    llama_text_model: Path
    llama_vl_model: Path
    llama_vl_mmproj: Path
    qwen3_text_hf_dir: Path
    qwen3_vl_hf_dir: Path
    vllm_text_model: Path
    vllm_vl_model: Path


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


def start_server(scenario: str, paths: ModelPaths):
    import signal
    port = int(os.environ.get("PERF_SERVER_PORT") or SCENARIOS[scenario]["server_port"])
    if os.environ.get("LLAMA_REUSE_SERVER") == "1" and wait_server(port, 2):
        return None
    env = os.environ.copy()
    env["PERF_REGRESSION_ROOT"] = str(ROOT)
    env["PERF_SERVER_PORT"] = str(port)
    command = [
        "bash",
        str(SCRIPTS / "run_llama_server.sh"),
        scenario,
        "--config",
        CONFIG_PATH,
    ]
    if scenario == "text_10c":
        command.extend(["--model", str(paths.llama_text_model)])
    else:
        command.extend(["--model", str(paths.llama_vl_model)])
        if paths.llama_vl_mmproj:
            command.extend(["--mmproj", str(paths.llama_vl_mmproj)])
    proc = subprocess.Popen(
        command,
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
    container = os.environ.get("VLLM_CONTAINER") or config_value(CONFIG, "runtime.vllm_container")
    if not container:
        raise RuntimeError("VLLM_CONTAINER is not set")
    full = ["docker", "exec"]
    if background:
        full.append("-d")
    full += [container, "bash", "-c", cmd]
    return subprocess.Popen(full, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generated_model_config(engine: str, scenario: str, paths: ModelPaths) -> str | Path:
    model_name = SCENARIOS[scenario]["models"][engine]
    server_port = os.environ.get("PERF_SERVER_PORT") or SCENARIOS[scenario]["server_port"]
    workload = SCENARIOS[scenario].get("workload", {})
    source = ROOT / "configs" / "models" / f"{model_name}.py"
    tokenizer_path = paths.qwen3_text_hf_dir if scenario == "text_10c" else paths.qwen3_vl_hf_dir

    text = source.read_text(encoding="utf-8")
    tokenizer_path_text = str(tokenizer_path).replace('"', '\\"')
    text = re.sub(r"path\s*=\s*\"\.\"", f'path="{tokenizer_path_text}"', text, count=1)
    text = re.sub(r"host_ip\s*=\s*\"localhost\"", 'host_ip="127.0.0.1"', text, count=1)
    text = re.sub(r"host_port\s*=\s*\d+", f"host_port={server_port}", text, count=1)
    if "max_output_tokens" in workload:
        text = re.sub(r"max_out_len\s*=\s*\d+", f"max_out_len={workload['max_output_tokens']}", text, count=1)
    if "batch_size" in workload:
        text = re.sub(r"batch_size\s*=\s*\d+", f"batch_size={workload['batch_size']}", text, count=1)
    output_name = f"generated_{engine}_{scenario}"
    output = ROOT / "configs" / "models" / f"{output_name}.py"
    output.write_text(text, encoding="utf-8")
    return output_name


def start_vllm_server(scenario: str, paths: ModelPaths) -> bool:
    cfg = VLLM_SERVERS[scenario]
    model_path = paths.vllm_text_model if scenario == "text_10c" else paths.vllm_vl_model
    npu_id = os.environ.get("VLLM_NPU_ID") or config_value(CONFIG, "runtime.vllm_npu_id", "5")
    cann_env = os.environ.get("CANN_SET_ENV") or config_value(CONFIG, "runtime.cann.set_env")
    compilation_config = json.dumps(cfg.get("compilation_config", {}), separators=(",", ":"))
    prefix_caching = "--enable-prefix-caching" if cfg.get("enable_prefix_caching", False) else ""
    _docker_exec("pkill -9 -f vllm 2>/dev/null; sleep 3")
    import time
    time.sleep(5)
    cmd = (
        f'source "{cann_env}" && '
        f"ASCEND_RT_VISIBLE_DEVICES={npu_id} vllm serve "
        f"{model_path} --port {cfg['port']} "
        f"{prefix_caching} --dtype {cfg.get('dtype', 'float16')} "
        f"--gpu-memory-utilization {cfg['gpu_memory_utilization']} "
        f"--max-model-len {cfg['max_model_len']} "
        f"--max-num-seqs {cfg['max_num_seqs']} "
        f"--compilation-config '{compilation_config}' "
        f"> /tmp/vllm_{scenario}.log 2>&1"
    )
    limit_mm = cfg.get("limit_mm_per_prompt")
    if limit_mm:
        cmd = cmd.replace(
            f"--port {cfg['port']} ",
            f"--port {cfg['port']} --limit-mm-per-prompt '{json.dumps(limit_mm, separators=(',', ':'))}' ",
            1,
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
    paths: ModelPaths,
    dataset: str | None = None,
    work_dir_name: str | None = None,
    mode: str = "perf",
    num_prompts: int | None = None,
) -> None:
    spec = SCENARIOS[scenario]
    dataset = dataset or spec["dataset"]
    work_dir = ROOT / "results" / engine / (work_dir_name or scenario)
    work_dir.mkdir(parents=True, exist_ok=True)
    ais_cfg = CONFIG["runtime"]["ais_bench"]
    ais_bench_bin = os.environ.get("AIS_BENCH_BIN") or shutil.which("ais_bench")
    if not ais_bench_bin:
        default_bin = resolved_path(ais_cfg.get("python", ""))
        if default_bin.is_file():
            ais_bench_bin = str(default_bin)
    if not ais_bench_bin:
        raise RuntimeError("ais_bench was not found; set AIS_BENCH_BIN or put it on PATH")
    command = [ais_bench_bin]
    if Path(ais_bench_bin).name.startswith("python"):
        ais_bench_module = os.environ.get("AIS_BENCH_MODULE") or ais_cfg.get(
            "module", "ais_bench.benchmark.cli.main"
        )
        command.extend(["-m", ais_bench_module])
    command.extend(
        [
            "--models",
            str(generated_model_config(engine, scenario, paths)),
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
    if scenario == "text_10c":
        env["QWEN3_1_7B_PATH"] = str(paths.qwen3_text_hf_dir)
    else:
        env["QWEN3_VL_4B_PATH"] = str(paths.qwen3_vl_hf_dir)
    python_paths = [str(ROOT / "configs")]
    ais_bench_stubs = os.environ.get("AISBENCH_STUBS") or ais_cfg.get("stubs")
    if ais_bench_stubs:
        python_paths.insert(0, ais_bench_stubs)
    ais_bench_root = Path(ais_bench_bin).resolve().parents[1]
    python_paths.extend(str(p) for p in sorted(ais_bench_root.glob("lib/python*/site-packages")))
    ais_bench_home = os.environ.get("AIS_BENCH_HOME")
    default_ais_bench_home = resolved_path(ais_cfg.get("site_packages", ""))
    if not ais_bench_home and default_ais_bench_home.is_dir():
        ais_bench_home = str(default_ais_bench_home)
    default_ais_bench_source = resolved_path(ais_cfg.get("source", ""))
    if default_ais_bench_source.is_dir():
        python_paths.insert(0, str(default_ais_bench_source))
    if ais_bench_home:
        python_paths.insert(0, ais_bench_home)
    env["PYTHONPATH"] = os.pathsep.join(python_paths + [env.get("PYTHONPATH", "")])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    cann_env = os.environ.get("CANN_SET_ENV")
    config_cann_env = resolved_path(config_value(CONFIG, "runtime.cann.set_env", ""))
    if not cann_env and config_cann_env.is_file():
        cann_env = str(config_cann_env)
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
    paths: ModelPaths,
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
            command.extend(["--config", CONFIG_PATH])
            if full_dataset or num_prompts <= 0:
                command.append("--full")
            subprocess.run(command, check=True, cwd=ROOT)
    if full_dataset or num_prompts <= 0:
        num_prompts = None
    run_benchmark(
        engine,
        scenario,
        config_dir,
        paths,
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
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(ROOT / "config.json"))
    pre_args, _ = pre_parser.parse_known_args()
    apply_config(load_config(pre_args.config), pre_args.config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH, help="path to config.json")
    parser.add_argument("--engine", choices=["llama", "vllm", "both"], default="llama")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS.keys()], default="all")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument(
        "--bench-mode",
        choices=["all", "perf", "accuracy"],
        default="all",
        help="benchmarks to run: all, perf, or accuracy",
    )
    parser.add_argument(
        "--accuracy-mode",
        choices=["both", "off", "only"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--accuracy-num-prompts", type=int, default=30)
    parser.add_argument("--accuracy-full", action="store_true")
    parser.add_argument(
        "--llama-text-model",
        default=os.environ.get("LLAMA_TEXT_MODEL")
        or config_value(CONFIG, "models.qwen3_1_7b.llama"),
    )
    parser.add_argument(
        "--llama-vl-model",
        default=os.environ.get("LLAMA_VL_MODEL")
        or config_value(CONFIG, "models.qwen3_vl_4b.llama"),
    )
    parser.add_argument(
        "--llama-vl-mmproj",
        default=os.environ.get("LLAMA_VL_MMPROJ")
        or config_value(CONFIG, "models.qwen3_vl_4b.mmproj"),
    )
    parser.add_argument(
        "--qwen3-1-7b-path",
        default=os.environ.get("QWEN3_1_7B_PATH")
        or config_value(CONFIG, "models.qwen3_1_7b.tokenizer"),
    )
    parser.add_argument(
        "--qwen3-vl-4b-path",
        default=os.environ.get("QWEN3_VL_4B_PATH")
        or config_value(CONFIG, "models.qwen3_vl_4b.tokenizer"),
    )
    parser.add_argument(
        "--vllm-text-model",
        default=os.environ.get("VLLM_TEXT_MODEL")
        or config_value(CONFIG, "models.qwen3_1_7b.vllm"),
    )
    parser.add_argument(
        "--vllm-vl-model",
        default=os.environ.get("VLLM_VL_MODEL")
        or config_value(CONFIG, "models.qwen3_vl_4b.vllm"),
    )
    args = parser.parse_args()

    if args.accuracy_mode is not None:
        if args.bench_mode != "all":
            parser.error("--accuracy-mode cannot be combined with --bench-mode")
        args.bench_mode = {"both": "all", "off": "perf", "only": "accuracy"}[args.accuracy_mode]
    run_perf = args.bench_mode in ("all", "perf")
    run_accuracy = args.bench_mode in ("all", "accuracy")

    scenarios = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]
    engines = ['llama', 'vllm'] if args.engine == 'both' else [args.engine]

    paths = ModelPaths(
        llama_text_model=Path(args.llama_text_model).expanduser(),
        llama_vl_model=Path(args.llama_vl_model).expanduser(),
        llama_vl_mmproj=Path(args.llama_vl_mmproj).expanduser() if args.llama_vl_mmproj else Path(),
        qwen3_text_hf_dir=Path(args.qwen3_1_7b_path).expanduser(),
        qwen3_vl_hf_dir=Path(args.qwen3_vl_4b_path).expanduser(),
        vllm_text_model=Path(args.vllm_text_model).expanduser(),
        vllm_vl_model=Path(args.vllm_vl_model).expanduser(),
    )

    missing = []
    if "llama" in engines:
        if "text_10c" in scenarios and not paths.llama_text_model.is_file():
            missing.append(str(paths.llama_text_model))
        if any(s != "text_10c" for s in scenarios):
            if not paths.llama_vl_model.is_file():
                missing.append(str(paths.llama_vl_model))
            if paths.llama_vl_mmproj and not paths.llama_vl_mmproj.is_file():
                missing.append(str(paths.llama_vl_mmproj))
    if "vllm" in engines:
        if "text_10c" in scenarios and not paths.vllm_text_model.is_dir():
            missing.append(str(paths.vllm_text_model))
        if any(s != "text_10c" for s in scenarios) and not paths.vllm_vl_model.is_dir():
            missing.append(str(paths.vllm_vl_model))
    if missing:
        print("ERROR: model paths not found:")
        for path in missing:
            print(f"  {path}")
        return 2

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
                        if not start_vllm_server(scenario, paths):
                            print(f'ERROR: failed to start vllm server for {scenario}')
                            continue
                        vllm_started.add(port)
                    if run_perf:
                        run_benchmark(engine, scenario, ROOT / 'configs', paths)
                    if run_accuracy:
                        run_accuracy_benchmark(
                            engine,
                            scenario,
                            ROOT / 'configs',
                            paths,
                            args.accuracy_num_prompts,
                            args.accuracy_full,
                        )
                stop_vllm_server()
            else:
                for scenario in scenarios:
                    proc = start_server(scenario, paths)
                    if proc is None and os.environ.get("LLAMA_REUSE_SERVER") != "1":
                        print(f'ERROR: failed to start llama server for {scenario}')
                        continue
                    try:
                        if run_perf:
                            run_benchmark(engine, scenario, ROOT / 'configs', paths)
                        if run_accuracy:
                            run_accuracy_benchmark(
                                engine,
                                scenario,
                                ROOT / 'configs',
                                paths,
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
            if run_accuracy:
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
