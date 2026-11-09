from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

from ais_bench.benchmark.datasets import MMCustomDataset, MMCustomEvaluator


reader_cfg = dict(input_columns=["content", "question"], output_column="answer")
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{content}",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
eval_cfg = dict(evaluator=dict(type=MMCustomEvaluator))

vl_small_datasets = [
    dict(
        abbr="vl-small-320",
        type=MMCustomDataset,
        path="/home/z00575061/llama.cpp-paged-integration-latest/tests/perf_regression/assets/image_320x320_questions.jsonl",
        mm_type="base64",
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
