import json
from pathlib import Path

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

from config_modules.local_mm_perf import LocalImagePerfDataset, NullEvaluator

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
dataset_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["datasets"]["vl_image_large"]
dataset_cfg["path"] = str((CONFIG_PATH.parent / dataset_cfg["path"]).resolve())


reader_cfg = dict(input_columns=["question", "image"], output_column="answer")

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            type="image_text",
            data=["image_url_base64", "text"],
            prompt=" Describe this image briefly.",
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

eval_cfg = dict(evaluator=dict(type=NullEvaluator))
vl_large_datasets = [
    dict(
        abbr="vl-large-1280",
        type=LocalImagePerfDataset,
        path=dataset_cfg["path"],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
