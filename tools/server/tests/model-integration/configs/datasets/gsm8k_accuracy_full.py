import json
from pathlib import Path

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

from config_modules.local_gsm8k import GSM8KSubsetDataset

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
gsm8k_cfg = _config["datasets"]["gsm8k_accuracy_full"]
_aisbench_source = Path(_config["runtime"]["ais_bench"]["source"]).expanduser()
gsm8k_cfg["path"] = str(
    (_aisbench_source / "ais_bench" / gsm8k_cfg["path"][len("aisbench://"):]).resolve()
)


reader_cfg = dict(input_columns=["question"], output_column="answer")

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=(
                        "Answer the following question. The last line should follow "
                        "this format: \"answer:$ANSWER\" (without quotes), where "
                        "ANSWER is a number. Let's think step by step.\n\nQuestion: {question}"
                    ),
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, stopping_criteria=["Question"]),
)

eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)
gsm8k_datasets = [
    dict(
        abbr="gsm8k-accuracy-full",
        type=GSM8KSubsetDataset,
        path=gsm8k_cfg["path"],
        subset_size=gsm8k_cfg["subset_size"],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
