#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


checks = []
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_QWEN3_ROOT = Path("/home/common/model/qwen3")
DEFAULT_LLAMA_TEXT_MODEL = DEFAULT_QWEN3_ROOT / "Qwen3-1.7B-F16.gguf"
DEFAULT_LLAMA_VL_MODEL = (
    DEFAULT_QWEN3_ROOT / "Qwen3-VL-4B-Instruct/Qwen3VL-4B-Instruct-F16.gguf"
)
DEFAULT_LLAMA_VL_MMPROJ = (
    DEFAULT_QWEN3_ROOT / "Qwen3-VL-4B-Instruct/Qwen3VL-4B-Instruct-mmproj-f16.gguf"
)
DEFAULT_AIS_BENCH_BIN = Path("/usr/bin/python3")


def check(name: str, ok: bool, detail: str = ""):
    checks.append((name, ok, detail))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llama-text-model",
        default=os.environ.get("LLAMA_TEXT_MODEL") or DEFAULT_LLAMA_TEXT_MODEL,
    )
    parser.add_argument(
        "--llama-vl-model",
        default=os.environ.get("LLAMA_VL_MODEL") or DEFAULT_LLAMA_VL_MODEL,
    )
    parser.add_argument(
        "--llama-vl-mmproj",
        default=os.environ.get("LLAMA_VL_MMPROJ") or DEFAULT_LLAMA_VL_MMPROJ,
    )
    args = parser.parse_args()
    text_model = Path(args.llama_text_model).expanduser()
    vl_model = Path(args.llama_vl_model).expanduser()
    vl_mmproj = Path(args.llama_vl_mmproj).expanduser()
    build = Path(os.environ.get("LLAMA_BUILD_DIR") or REPO_ROOT / "build-cann-aclgraph")
    server = build / "bin" / "llama-server"
    cache = build / "CMakeCache.txt"
    check("llama-server", server.is_file(), str(server))
    check("USE_ACL_GRAPH", cache.is_file() and "USE_ACL_GRAPH:BOOL=ON" in cache.read_text(errors="ignore"), str(cache))
    ais_bench = os.environ.get("AIS_BENCH_BIN") or shutil.which("ais_bench")
    if not ais_bench and DEFAULT_AIS_BENCH_BIN.is_file():
        ais_bench = str(DEFAULT_AIS_BENCH_BIN)
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
