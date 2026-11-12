import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config.json"
RUN_OVERRIDE_PATH = Path(__file__).resolve().parents[1] / ".dataset_override.json"


def load_config(path=None):
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def model_path(model_name: str, config_path=None) -> str:
    config = load_config(config_path)
    model = config["models"][model_name]
    return model.get("tokenizer") or model.get("llama") or model.get("vllm", "")


def dataset_config(name: str, config_path=None) -> dict:
    config = load_config(config_path)
    cfg = dict(config["datasets"][name])
    if RUN_OVERRIDE_PATH.is_file():
        overrides = json.loads(RUN_OVERRIDE_PATH.read_text(encoding="utf-8"))
        cfg.update(overrides.get("datasets", {}).get(name, {}))
    path = str(cfg.get("path", ""))
    if path.startswith("aisbench://"):
        source = Path(config["runtime"]["ais_bench"]["source"]).expanduser()
        cfg["path"] = str((source / "ais_bench" / path[len("aisbench://"):]).resolve())
    elif path:
        cfg["path"] = str((ROOT / path).resolve())
    return cfg
