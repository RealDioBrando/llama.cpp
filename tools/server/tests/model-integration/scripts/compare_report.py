#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric(data: dict, key: str):
    value = data.get(key)
    return value if isinstance(value, (int, float)) else None


def status(value, target):
    if value is None:
        return "MISSING"
    return "PASS" if value <= target else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "comparison.md")
    args = parser.parse_args()

    current = load(ROOT / "results" / "summary.json")
    baseline = load(ROOT / "results" / "vllm_baseline.json")
    rows = ["| scenario | metric | llama.cpp | vLLM baseline | target | status |", "|---|---|---:|---:|---:|---|"]
    for scenario, spec in {
        "text_10c": ["TTFT", "TPOT"],
        "vl_small": ["E2EL"],
        "vl_large": ["E2EL"],
    }.items():
        llama = current.get("llama", {}).get(scenario, {})
        vllm = baseline.get(scenario, {})
        for metric_name in spec:
            target = llama.get("targets", {}).get(metric_name)
            if target is None:
                continue
            value = metric(llama, metric_name)
            rows.append(
                f"| {scenario} | {metric_name} | {value if value is not None else 'N/A'} "
                f"| {metric(vllm, metric_name) if metric(vllm, metric_name) is not None else 'N/A'} "
                f"| {target} | {status(value, target)} |"
            )
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
