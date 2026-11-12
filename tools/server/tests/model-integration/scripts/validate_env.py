#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs"))
from config_modules.runtime_config import load_config


checks = []
ASSET_DIR = ROOT / "assets"
REPO_ROOT = ROOT.parents[3]


def check(name: str, ok: bool, detail: str = ""):
    checks.append((name, ok, detail))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument(
        "--llama-text-model",
        default=None,
    )
    parser.add_argument(
        "--llama-vl-model",
        default=None,
    )
    parser.add_argument(
        "--llama-vl-mmproj",
        default=None,
    )
    args = parser.parse_args()
    config = load_config(args.config)
    text_model_default = Path(config["models"]["qwen3_1_7b"]["llama"])
    vl_model_default = Path(config["models"]["qwen3_vl_4b"]["llama"])
    vl_mmproj_default = Path(config["models"]["qwen3_vl_4b"]["mmproj"])
    default_ais_bench_bin = Path(config["runtime"]["ais_bench"]["python"])
    if args.llama_text_model is None:
        args.llama_text_model = text_model_default
    if args.llama_vl_model is None:
        args.llama_vl_model = vl_model_default
    if args.llama_vl_mmproj is None:
        args.llama_vl_mmproj = vl_mmproj_default
    text_model = Path(args.llama_text_model).expanduser()
    vl_model = Path(args.llama_vl_model).expanduser()
    vl_mmproj = Path(args.llama_vl_mmproj).expanduser()
    build = Path(os.environ.get("LLAMA_BUILD_DIR") or REPO_ROOT / "build-cann-aclgraph")
    server = build / "bin" / "llama-server"
    cache = build / "CMakeCache.txt"
    check("llama-server", server.is_file(), str(server))
    check("USE_ACL_GRAPH", cache.is_file() and "USE_ACL_GRAPH:BOOL=ON" in cache.read_text(errors="ignore"), str(cache))
    ais_bench = os.environ.get("AIS_BENCH_BIN") or shutil.which("ais_bench")
    if not ais_bench and default_ais_bench_bin.is_file():
        ais_bench = str(default_ais_bench_bin)
    check("ais_bench", ais_bench is not None, str(ais_bench or "not found"))
    check(
        "qwen3-1.7b-gguf",
        text_model.is_file(),
        str(text_model),
    )
    check(
        "qwen3-vl-4b-gguf",
        vl_model.is_file(),
        str(vl_model),
    )
    check("qwen3-vl-4b-mmproj", vl_mmproj.is_file(), str(vl_mmproj))
    for name in ("image_320x320.jpg", "image_1280x720.jpg"):
        check(name, (ASSET_DIR / name).is_file())
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
