#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time


FLOAT = r"([0-9]+(?:\.[0-9]+)?)"


def host_load1() -> float | None:
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None


def ascend_device_mapping() -> dict[int, tuple[int, int]]:
    try:
        text = subprocess.check_output(
            ["npu-smi", "info", "-m"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return {}

    mapping: dict[int, tuple[int, int]] = {}
    for line in text.splitlines():
        columns = line.split()
        if len(columns) < 4:
            continue
        try:
            npu_id = int(columns[0])
            chip_id = int(columns[1])
            logic_id = int(columns[2])
        except ValueError:
            continue
        mapping[logic_id] = (npu_id, chip_id)
    return mapping


def device_aicore_usage(device_id: int) -> float | None:
    mapping = ascend_device_mapping()
    if device_id not in mapping:
        return None
    npu_id, chip_id = mapping[device_id]
    try:
        text = subprocess.check_output(
            ["npu-smi", "info", "-i", str(npu_id), "-c", str(chip_id), "-t", "usages"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    match = re.search(r"Aicore Usage Rate\(%\)\s*:\s*" + FLOAT, text)
    return float(match.group(1)) if match else None


def wait_for_idle(args: argparse.Namespace) -> dict:
    deadline = time.monotonic() + args.idle_timeout
    samples: list[dict] = []
    while True:
        load1 = host_load1()
        device_usage = device_aicore_usage(args.device)
        samples.append({"host_load1": load1, "aicore_usage_percent": device_usage})
        load_busy = load1 is not None and args.max_host_load1 >= 0 and load1 > args.max_host_load1
        device_busy = (
            device_usage is not None and args.max_device_usage >= 0 and device_usage > args.max_device_usage
        )
        if not load_busy and not device_busy:
            return {"waited": len(samples) > 1, "samples": samples}
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"device not idle after {args.idle_timeout:g}s: load1={load1}, aicore={device_usage}"
            )
        time.sleep(args.idle_poll_interval)


def find_float(text: str, pattern: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"cannot parse {label}")
    return float(match.group(1))


def make_prompt(path: Path, repeats: int) -> None:
    base = "the quick brown fox jumps over the lazy dog"
    path.write_text(f"{base * repeats} a short deterministic suffix.\n", encoding="ascii")


def paged_command(args: argparse.Namespace, requests: int) -> list[str]:
    return [
        str(args.paged_bin),
        "-m", args.model,
        "-kvp",
        "-kvbls", str(args.block_size),
        "-np", str(requests),
        "-ns", str(requests),
        "-n", str(args.n_predict),
        "-c", str(args.context),
        "-b", str(args.ubatch),
        "-ub", str(args.ubatch),
        "-t", str(args.threads),
        "-tb", str(args.threads),
        "--temp", "0",
        "-ngl", str(args.n_gpu_layers),
        "-sm", "none",
        "-mg", str(args.device),
    ]


def unified_command(args: argparse.Namespace, requests: int) -> list[str]:
    context = args.context if requests == 1 else args.batched_context
    return [
        str(args.unified_bin),
        "-m", args.model,
        "-f", str(args.prompt_file),
        "-np", str(requests),
        "-ns", str(requests),
        "-n", str(args.n_predict),
        "-c", str(context),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-t", str(args.threads),
        "-tb", str(args.threads),
        "--temp", "0",
        "-ngl", str(args.n_gpu_layers),
        "-sm", "none",
        "-mg", str(args.device),
        "--ignore-eos",
    ]


def run_case(
    name: str,
    command: list[str],
    environment: dict[str, str] | None,
    log_path: Path,
    timeout: float,
    dry_run: bool,
    phase: str = "measured",
) -> dict:
    result: dict = {
        "name": name,
        "phase": phase,
        "command": command,
        "command_line": shlex.join(command),
        "environment": environment or {},
        "log": str(log_path),
    }
    if dry_run:
        result["returncode"] = 0
        return result

    env = os.environ.copy()
    if environment:
        env.update(environment)

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout,
            check=False,
        )
    result["wall_time_s"] = time.monotonic() - started
    result["returncode"] = completed.returncode
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with return code {completed.returncode}; see {log_path}")
    return result


def parse_paged_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "=== Paged KV Cache Summary ===" not in text:
        raise ValueError(f"no paged summary in {path}")

    gpu_blocks = int(find_float(text, r"n_gpu_blocks\s+:\s+" + FLOAT, "paged GPU blocks"))
    cpu_blocks = int(find_float(text, r"n_cpu_blocks\s+:\s+" + FLOAT, "paged CPU blocks"))
    return {
        "cache": "paged",
        "total_time_s": find_float(text, r"total elapsed\s+:\s+" + FLOAT + r"\s*s", "paged total time"),
        "decode_time_s": find_float(text, r"decode elapsed\s+:\s+" + FLOAT + r"\s*s", "paged decode time"),
        "decode_throughput_tok_s": find_float(
            text, r"decode aggregate tps\s+:\s+" + FLOAT + r"\s*tokens/s", "paged decode throughput"
        ),
        "ttft_avg_s": find_float(
            text, r"ttft\s+avg\s*/\s*min\s*/\s*max\s*:\s*" + FLOAT, "paged average TTFT"
        ) / 1000.0,
        "ttft_max_s": find_float(
            text, r"ttft\s+avg\s*/\s*min\s*/\s*max\s*:\s*" + FLOAT + r"\s*/\s*" + FLOAT + r"\s*/\s*" + FLOAT,
            "paged maximum TTFT",
        ) / 1000.0,
        "tpot_ms": find_float(text, r"tpot\s+avg\s+:\s+" + FLOAT + r"\s*ms/token", "paged TPOT"),
        "n_gpu_blocks": gpu_blocks,
        "n_cpu_blocks": cpu_blocks,
    }


def parse_unified_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    clients = re.findall(
        r"Client\s+\d+.*?"
        r"prompt\s+" + FLOAT + r"\s+t,\s*"
        r"response\s+" + FLOAT + r"\s+t,\s*"
        r"time\s+" + FLOAT + r"\s+s,\s*"
        r"gen time\s+" + FLOAT + r"\s+s",
        text,
    )
    if not clients:
        raise ValueError(f"no completed clients in {path}")

    total_generated = 0
    total_prompt = 0
    total_decode_intervals = 0.0
    decode_intervals = 0
    ttfts = []
    total_time = 0.0
    decode_time = 0.0
    for prompt_tokens, response_tokens, time_s, gen_time_s in clients:
        prompt_tokens = int(prompt_tokens)
        response_tokens = int(response_tokens)
        time_s = float(time_s)
        gen_time_s = float(gen_time_s)
        total_prompt += prompt_tokens
        total_generated += response_tokens
        total_decode_intervals += gen_time_s
        decode_intervals += max(0, response_tokens - 1)
        ttfts.append(time_s - gen_time_s)
        total_time = max(total_time, time_s)
        decode_time = max(decode_time, gen_time_s)

    return {
        "cache": "unified",
        "total_time_s": total_time,
        "decode_time_s": decode_time,
        "decode_throughput_tok_s": total_generated / decode_time if decode_time else 0.0,
        "ttft_avg_s": statistics.mean(ttfts),
        "ttft_max_s": max(ttfts),
        "tpot_ms": 1000.0 * total_decode_intervals / decode_intervals if decode_intervals else 0.0,
        "n_prompt_tokens": total_prompt,
        "n_generated_tokens": total_generated,
        "n_clients": len(clients),
    }


def summarize(runs: list[dict], stability_metric: str, max_relative_range: float) -> dict:
    keys = (
        "total_time_s",
        "decode_time_s",
        "decode_throughput_tok_s",
        "ttft_avg_s",
        "ttft_max_s",
        "tpot_ms",
    )
    summary: dict = {"runs": len(runs)}
    for key in keys:
        if key not in runs[0]:
            continue
        values = [run[key] for run in runs]
        median = statistics.median(values)
        summary[key] = {
            "mean": statistics.mean(values),
            "median": median,
            "min": min(values),
            "max": max(values),
        }
    if stability_metric in summary:
        metric = summary[stability_metric]
        relative_range = (
            (metric["max"] - metric["min"]) / metric["median"] if metric["median"] else float("inf")
        )
        summary["stability"] = {
            "metric": stability_metric,
            "max_relative_range": max_relative_range,
            "relative_range": relative_range,
            "stable": relative_range <= max_relative_range,
        }
    return summary


def write_markdown(path: Path, output: dict) -> None:
    lines = [
        "# Paged vs Unified performance",
        "",
        f"Generated: {output['generated_at']}",
        "",
        f"- Model: `{output['config']['model']}`",
        f"- Device: `{output['config']['device']}`",
        f"- Prompt tokens: `{output['config']['prompt_tokens']}`",
        f"- Generated tokens: `{output['config']['n_predict']}`",
        f"- Repetitions: `{output['config']['repetitions']}`",
        f"- Discarded warmups: `{output['config']['warmup_runs']}`",
        f"- Scenario order: `{output['config']['scenario_order']}`",
        "",
        "| Scenario | Cache | Runs | Stable | Decode time median | Decode throughput median | TTFT median | TPOT median |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario_name, result in output["scenarios"].items():
        summary = result["summary"]
        stable = summary.get("stability", {}).get("stable", "n/a")
        lines.append(
            f"| {scenario_name} | {result['runs'][0]['cache']} | {summary['runs']} | {stable} "
            f"| {summary['decode_time_s']['median']:.3f} s "
            f"| {summary['decode_throughput_tok_s']['median']:.2f} tok/s "
            f"| {summary['ttft_avg_s']['median']:.3f} s "
            f"| {summary['tpot_ms']['median']:.2f} ms |"
        )

    lines.extend(["", "## Commands", ""])
    for scenario_name, result in output["scenarios"].items():
        lines.append(f"### {scenario_name}")
        lines.append("")
        lines.append("```bash")
        lines.append("")
        env_prefix = ""
        if result["runs"][0]["environment"]:
            env_prefix = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in result["runs"][0]["environment"].items()
            ) + " "
        lines.append(env_prefix + result["runs"][0]["command_line"])
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fixed Paged vs Unified performance case.")
    parser.add_argument("--model", required=True, help="path to the GGUF model")
    parser.add_argument("--bin-dir", required=True, type=Path, help="directory containing llama-paged and llama-parallel")
    parser.add_argument("--device", type=int, default=5, help="Ascend device id passed to -mg")
    parser.add_argument("--output-dir", type=Path, default=None, help="result directory")
    parser.add_argument("--label", default=None, help="optional label used in the output directory")
    parser.add_argument("--repetitions", type=int, default=3, help="runs per scenario")
    parser.add_argument("--warmup-runs", type=int, default=2, help="discarded runs per scenario")
    parser.add_argument("--max-host-load1", type=float, default=16.0, help="wait while 1m load is above this; negative disables")
    parser.add_argument("--max-device-usage", type=float, default=5.0, help="wait while AICore usage is above this; negative disables")
    parser.add_argument("--idle-timeout", type=float, default=300.0, help="maximum time to wait for an idle device")
    parser.add_argument("--idle-poll-interval", type=float, default=5.0, help="idle check interval in seconds")
    parser.add_argument("--run-gap", type=float, default=2.0, help="quiet gap between runs in seconds")
    parser.add_argument(
        "--stability-metric",
        choices=("decode_time_s", "decode_throughput_tok_s", "tpot_ms"),
        default="decode_time_s",
        help="metric used by the stability gate",
    )
    parser.add_argument("--stability-max-range", type=float, default=0.35, help="maximum (max-min)/median per scenario")
    parser.add_argument("--max-extra-runs", type=int, default=2, help="extra measured runs allowed for stability")
    parser.add_argument("--require-stable", action="store_true", help="fail instead of reporting an unstable result")
    parser.add_argument("--timeout", type=float, default=600.0, help="timeout for each run in seconds")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("paged-single", "unified-single", "paged-10", "unified-10"),
        default=("paged-single", "unified-single", "paged-10", "unified-10"),
        help="scenarios to run",
    )
    parser.add_argument("--prompt-file", type=Path, default=None, help="prompt file for Unified")
    parser.add_argument("--prompt-tokens", type=int, default=1024, help="Paged prompt token count")
    parser.add_argument("--unified-prompt-repeats", type=int, default=64, help="repeats used to make a Unified prompt")
    parser.add_argument("--n-predict", type=int, default=128, help="generated tokens per request")
    parser.add_argument("--context", type=int, default=4096, help="single-request context size")
    parser.add_argument("--batched-context", type=int, default=16384, help="10-request Unified context size")
    parser.add_argument("--batch", type=int, default=1024, help="logical batch size")
    parser.add_argument("--ubatch", type=int, default=1024, help="physical ubatch size")
    parser.add_argument("--threads", type=int, default=8, help="host threads for decode and batch compute")
    parser.add_argument("--block-size", type=int, default=16, help="paged KV block size")
    parser.add_argument("--n-gpu-layers", type=int, default=99, help="layers to offload")
    parser.add_argument("--no-warmup", action="store_true", help="pass --no-warmup to both binaries")
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment variable, repeatable",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the case without launching processes")
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    if args.max_extra_runs < 0:
        parser.error("--max-extra-runs cannot be negative")

    args.paged_bin = args.bin_dir / "llama-paged"
    args.unified_bin = args.bin_dir / "llama-parallel"
    for binary in (args.paged_bin, args.unified_bin):
        if not args.dry_run and not binary.is_file():
            parser.error(f"binary not found: {binary}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    label = args.label or timestamp
    args.output_dir = args.output_dir or Path("perf-results") / f"paged-vs-unified-{label}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.prompt_file:
        args.prompt_file = args.prompt_file.resolve()
        if not args.dry_run and not args.prompt_file.is_file():
            parser.error(f"prompt file not found: {args.prompt_file}")
    else:
        args.prompt_file = args.output_dir / "unified-prompt.txt"
        if not args.dry_run:
            make_prompt(args.prompt_file, args.unified_prompt_repeats)

    extra_env: dict[str, str] = {}
    for item in args.extra_env:
        key, separator, value = item.partition("=")
        if not separator or not key:
            parser.error(f"invalid --extra-env value: {item}")
        extra_env[key] = value

    requests_by_scenario = {
        "paged-single": 1,
        "unified-single": 1,
        "paged-10": 10,
        "unified-10": 10,
    }
    output: dict = {
        "generated_at": timestamp,
        "label": label,
        "config": {
            "model": str(args.model),
            "paged_bin": str(args.paged_bin),
            "unified_bin": str(args.unified_bin),
            "device": args.device,
            "repetitions": args.repetitions,
            "warmup_runs": 0 if args.no_warmup else args.warmup_runs,
            "scenario_order": "interleaved-alternating",
            "idle": {
                "max_host_load1": args.max_host_load1,
                "max_device_usage_percent": args.max_device_usage,
                "timeout_s": args.idle_timeout,
                "poll_interval_s": args.idle_poll_interval,
            },
            "run_gap_s": args.run_gap,
            "stability": {
                "metric": args.stability_metric,
                "max_relative_range": args.stability_max_range,
                "max_extra_runs": args.max_extra_runs,
                "require_stable": args.require_stable,
            },
            "timeout_s": args.timeout,
            "prompt_file": str(args.prompt_file),
            "prompt_tokens": args.prompt_tokens,
            "n_predict": args.n_predict,
            "context": args.context,
            "batched_context": args.batched_context,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "threads": args.threads,
            "block_size": args.block_size,
            "n_gpu_layers": args.n_gpu_layers,
            "warmup": not args.no_warmup,
            "extra_env": extra_env,
        },
        "scenarios": {},
    }

    paged_env = {
        "LLAMA_PAGED_PROMPT_TOKENS": str(args.prompt_tokens),
        "LLAMA_PAGED_PREFIX_CACHE": "0",
        "LLAMA_PAGED_PREFIX_CKPT": "0",
        **extra_env,
    }
    warmup_args = ["--no-warmup"] if args.no_warmup else []

    def command_for(scenario: str) -> tuple[list[str], dict[str, str]]:
        requests = requests_by_scenario[scenario]
        if scenario.startswith("paged-"):
            return paged_command(args, requests) + warmup_args, paged_env
        return unified_command(args, requests) + warmup_args, extra_env

    if not args.dry_run and args.warmup_runs:
        for scenario in args.scenarios:
            command, environment = command_for(scenario)
            wait_for_idle(args)
            run_case(
                scenario, command, environment,
                args.output_dir / f"{scenario}_warmup.log",
                args.timeout, False, phase="warmup",
            )
            if args.run_gap > 0:
                time.sleep(args.run_gap)
        print("warmup: done", flush=True)

    scenario_runs: dict[str, list[dict]] = {scenario: [] for scenario in args.scenarios}

    def run_measured(scenario: str) -> None:
        command, environment = command_for(scenario)
        runs = scenario_runs[scenario]
        cache = "paged" if scenario.startswith("paged-") else "unified"
        log_path = args.output_dir / f"{scenario}_r{len(runs) + 1}.log"
        if not args.dry_run:
            wait_for_idle(args)
        run = run_case(scenario, command, environment, log_path, args.timeout, args.dry_run)
        if not args.dry_run:
            if cache == "paged":
                run.update(parse_paged_log(log_path))
            else:
                run.update(parse_unified_log(log_path))
        if not args.dry_run and args.run_gap > 0:
            time.sleep(args.run_gap)
        runs.append(run)
        print(f"{scenario}: r{len(runs)} done", flush=True)

    def stable(scenario: str) -> bool:
        return bool(
            summarize(scenario_runs[scenario], args.stability_metric, args.stability_max_range)
            .get("stability", {})
            .get("stable", False)
        )

    for round_index in range(args.repetitions):
        ordered_scenarios = args.scenarios if round_index % 2 == 0 else tuple(reversed(args.scenarios))
        for scenario in ordered_scenarios:
            run_measured(scenario)

    for extra_run in range(0 if args.dry_run else args.max_extra_runs):
        if all(stable(scenario) for scenario in args.scenarios):
            break
        ordered_scenarios = args.scenarios if (args.repetitions + extra_run) % 2 == 0 else tuple(reversed(args.scenarios))
        for scenario in ordered_scenarios:
            if stable(scenario):
                continue
            run_measured(scenario)

    unstable = [] if args.dry_run else [scenario for scenario in args.scenarios if not stable(scenario)]
    if unstable and args.require_stable:
        names = ", ".join(unstable)
        raise RuntimeError(f"stability gate failed for: {names}; see {args.output_dir}")

    for scenario, runs in scenario_runs.items():
        output["scenarios"][scenario] = {
            "requests": requests_by_scenario[scenario],
            "runs": runs,
            "summary": summarize(runs, args.stability_metric, args.stability_max_range),
        }
        print(f"{scenario}: done", flush=True)

    json_path = args.output_dir / "results.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.dry_run:
        write_markdown(args.output_dir / "results.md", output)

    print(f"results: {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
