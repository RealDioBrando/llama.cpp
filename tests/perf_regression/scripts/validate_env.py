#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path


checks = []
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


def check(name: str, ok: bool, detail: str = ""):
    checks.append((name, ok, detail))


def main() -> int:
    build = Path(os.environ.get("LLAMA_BUILD_DIR", "/home/z00575061/llama.cpp-develop-qwen3-20261102/build-cann-aclgraph"))
    server = build / "bin" / "llama-server"
    cache = build / "CMakeCache.txt"
    check("llama-server", server.is_file(), str(server))
    check("USE_ACL_GRAPH", cache.is_file() and "USE_ACL_GRAPH:BOOL=ON" in cache.read_text(errors="ignore"), str(cache))
    check("ais_bench", shutil.which("ais_bench") is not None)
    check("qwen3-1.7b-gguf", Path("/home/z00575061/models/Qwen3-1.7B-F16.gguf").is_file())
    check("qwen3-1.7b-hf", Path("/home/c30081098/models/Qwen3-1.7B").is_dir())
    check("qwen3-vl-4b-gguf", Path("/home/common/model/qwen3/Qwen3-VL-4B-Instruct/Qwen3VL-4B-Instruct-F16.gguf").is_file())
    for name in ("image_320x320.jpg", "image_1280x720.jpg"):
        check(name, (ASSET_DIR / name).is_file())
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
