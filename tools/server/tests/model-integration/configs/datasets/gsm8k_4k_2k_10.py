import json
from pathlib import Path

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

from config_modules.fixed_datasets import NullEvaluator
from config_modules.local_gsm8k import GSM8KFixedTokenDataset

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"
_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
gsm8k_cfg = _config["datasets"]["gsm8k_4k_2k_10"]
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
    inferencer=dict(type=GenInferencer),
)

eval_cfg = dict(evaluator=dict(type=NullEvaluator))
perf_datasets = [
    dict(
        abbr="gsm8k-4k-2k-10",
        type=GSM8KFixedTokenDataset,
        path=gsm8k_cfg["path"],
        subset_size=gsm8k_cfg["subset_size"],
        input_tokens=gsm8k_cfg["input_tokens"],
        dataset_name="gsm8k_4k_2k_10",
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
