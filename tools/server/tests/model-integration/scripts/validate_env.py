#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path


checks = []
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
REPO_ROOT = Path(__file__).resolve().parents[2]


def check(name: str, ok: bool, detail: str = ""):
    checks.append((name, ok, detail))


def main() -> int:
    build = Path(os.environ.get("LLAMA_BUILD_DIR") or REPO_ROOT / "build-cann-aclgraph")
    server = build / "bin" / "llama-server"
    cache = build / "CMakeCache.txt"
    check("llama-server", server.is_file(), str(server))
    check("USE_ACL_GRAPH", cache.is_file() and "USE_ACL_GRAPH:BOOL=ON" in cache.read_text(errors="ignore"), str(cache))
    ais_bench = os.environ.get("AIS_BENCH_BIN") or shutil.which("ais_bench")
    check("ais_bench", ais_bench is not None, str(ais_bench or "not found"))
    text_model = os.environ.get("LLAMA_TEXT_MODEL") or os.environ.get("LLAMA_MODEL")
    vl_model = os.environ.get("LLAMA_VL_MODEL") or os.environ.get("LLAMA_MODEL")
    vl_mmproj = os.environ.get("LLAMA_VL_MMPROJ") or os.environ.get("LLAMA_MMPROJ")
    check(
        "qwen3-1.7b-gguf",
        text_model is not None and Path(text_model).is_file(),
        text_model or "LLAMA_TEXT_MODEL is not set",
    )
    check(
        "qwen3-vl-4b-gguf",
        vl_model is not None and Path(vl_model).is_file(),
        vl_model or "LLAMA_VL_MODEL is not set",
    )
    if vl_mmproj:
        check("qwen3-vl-4b-mmproj", Path(vl_mmproj).is_file(), vl_mmproj)
    for name in ("image_320x320.jpg", "image_1280x720.jpg"):
        check(name, (ASSET_DIR / name).is_file())
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
