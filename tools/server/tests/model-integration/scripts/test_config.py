#!/usr/bin/env python3
import argparse
import json
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def config_path(path: str | None = None) -> Path:
    return Path(path or DEFAULT_CONFIG).expanduser().resolve()


def load_config(path: str | None = None) -> dict:
    selected = config_path(path)
    if not selected.is_file():
        raise FileNotFoundError(f"config not found: {selected}")
    with selected.open(encoding="utf-8") as f:
        data = json.load(f)
    version = int(data.get("version", 0))
    if version != 1:
        raise ValueError(f"unsupported config version: {version}")
    return data


def config_value(config: dict, dotted: str, default=None):
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolved_path(value, base: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def dataset_config(config: dict, name: str) -> dict:
    datasets = config.get("datasets", {})
    if name not in datasets:
        raise KeyError(f"dataset not found in config: {name}")
    cfg = dict(datasets[name])
    path = cfg.get("path")
    if path and not str(path).startswith("ais_bench/"):
        cfg["path"] = str(resolved_path(path))
    return cfg


def shell_escape(value) -> str:
    if isinstance(value, bool):
        value = "1" if value else "0"
    return shlex.quote(str(value))


def merged(config: dict, engine: str, scenario: str | None) -> dict:
    server = config.get("servers", {}).get(engine, {})
    defaults = dict(server.get("defaults", {}))
    settings = dict(defaults)
    if scenario:
        settings.update(server.get("scenarios", {}).get(scenario, {}))
    return settings


def shell_variables(config: dict, engine: str, scenario: str) -> dict:
    runtime = config["runtime"]
    build = runtime.get("build", {})
    cann = runtime.get("cann", {})
    values = {
        "CFG_NPU_ID": runtime.get("npu_id", 5),
        "CFG_VLLM_NPU_ID": runtime.get("vllm_npu_id", runtime.get("npu_id", 5)),
        "CFG_VLLM_CONTAINER": runtime.get("vllm_container", ""),
        "CFG_LLAMA_SOURCE": build.get("source", ""),
        "CFG_LLAMA_BUILD_DIR": build.get("build_dir", ""),
        "CFG_BUILD_JOBS": build.get("jobs", 8),
        "CFG_ASCEND_HOME": cann.get("ascend_home", "/usr/local/Ascend"),
        "CFG_CANN_SET_ENV": cann.get("set_env", ""),
        "CFG_ATB_SET_ENV": cann.get("atb_set_env", ""),
    }

    if engine == "llama":
        settings = merged(config, engine, scenario)
        features = settings.get("backend_features", {})
        text_cfg = config["models"]["qwen3_1_7b"]
        vl_cfg = config["models"]["qwen3_vl_4b"]
        values.update({
            "CFG_LLAMA_TEXT_MODEL": text_cfg.get("llama", ""),
            "CFG_LLAMA_VL_MODEL": vl_cfg.get("llama", ""),
            "CFG_LLAMA_VL_MMPROJ": vl_cfg.get("mmproj", ""),
            "CFG_PORT": settings.get("port", ""),
            "CFG_CTX": settings.get("context", ""),
            "CFG_PARALLEL": settings.get("parallel", ""),
            "CFG_THREADS": settings.get("threads", ""),
            "CFG_THREADS_BATCH": settings.get("threads_batch", ""),
            "CFG_ACL_GRAPH": settings.get("acl_graph", False),
            "CFG_PREFILL_GRAPH": settings.get("prefill_graph", False),
            "CFG_ACT_TYPE": settings.get("act_type", "none"),
            "CFG_MMPROJ_OFFLOAD": settings.get("mmproj_offload", True),
            "CFG_KV_BLOCK_SIZE": settings.get("kv_block_size", ""),
            "CFG_BATCH_SIZE": settings.get("batch_size", ""),
            "CFG_UBATCH_SIZE": settings.get("ubatch_size", ""),
            "CFG_KV_PAGED": settings.get("kv_paged", False),
            "CFG_CACHE_RAM": settings.get("cache_ram", 0),
            "CFG_NGPU_LAYERS": settings.get("n_gpu_layers", 99),
            "CFG_MAIN_GPU": settings.get("main_gpu", 0),
            "CFG_SAMPLING_MODE": settings.get("sampling_mode", "none"),
            "CFG_FA_V5": features.get("fa_v5", False),
            "CFG_WEIGHT_NZ": features.get("weight_nz", False),
            "CFG_GDN_PROJ": features.get("gdn_proj", False),
            "CFG_GDN_SMALL": features.get("gdn_small", False),
            "CFG_GDN_PREFILL_CHUNK": features.get("gdn_prefill_chunk", False),
        })
    elif engine == "vllm":
        settings = merged(config, engine, scenario)
        text_cfg = config["models"]["qwen3_1_7b"]
        vl_cfg = config["models"]["qwen3_vl_4b"]
        values.update({
            "CFG_VLLM_TEXT_MODEL": text_cfg.get("vllm", ""),
            "CFG_VLLM_VL_MODEL": vl_cfg.get("vllm", ""),
            "CFG_PORT": settings.get("port", ""),
            "CFG_MAX_LEN": settings.get("max_model_len", ""),
            "CFG_MAX_SEQS": settings.get("max_num_seqs", ""),
            "CFG_GPU_MEM_UTIL": settings.get("gpu_memory_utilization", ""),
            "CFG_DTYPE": settings.get("dtype", "float16"),
            "CFG_PREFIX_CACHING": settings.get("enable_prefix_caching", False),
            "CFG_LIMIT_MM": json.dumps(settings.get("limit_mm_per_prompt", {}), separators=(",", ":")),
            "CFG_COMPILATION_CONFIG": json.dumps(settings.get("compilation_config", {}), separators=(",", ":")),
        })
    elif engine == "build":
        build_options = build.get("options", {})
        values.update({
            "CFG_BUILD_GGML_CANN": build_options.get("GGML_CANN", "ON"),
            "CFG_BUILD_SOC_VERSION": build_options.get("CANN_SOC_VERSION", "310P3"),
            "CFG_BUILD_TYPE": build_options.get("CMAKE_BUILD_TYPE", "Release"),
            "CFG_BUILD_USE_ACL_GRAPH": build_options.get("USE_ACL_GRAPH", "ON"),
            "CFG_BUILD_LLAMA_CURL": build_options.get("LLAMA_CURL", "OFF"),
        })
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Read model-integration config.json")
    parser.add_argument("--config", help="path to config.json")
    parser.add_argument("--engine", choices=["llama", "vllm", "build"], default="llama")
    parser.add_argument("--scenario", help="scenario name")
    parser.add_argument("--get", action="append", default=[], help="print one dotted config key")
    parser.add_argument("--shell", action="store_true", help="print shell-compatible CFG_* values")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.get:
        for key in args.get:
            value = config_value(config, key)
            if value is None:
                print(f"ERROR: config key not found: {key}", file=sys.stderr)
                return 1
            print(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        return 0

    if args.shell:
        scenario = args.scenario or "text_10c"
        values = shell_variables(config, args.engine, scenario)
        for key, value in values.items():
            print(f"{key}={shell_escape(value)}")
        return 0

    print(json.dumps(config, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
