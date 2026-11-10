from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

from config_modules.fixed_datasets import FixedTokenDataset, NullEvaluator


reader_cfg = dict(input_columns=["question"], output_column="answer")
infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template="{question}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, is_synthetic=True),
)
eval_cfg = dict(evaluator=dict(type=NullEvaluator))

perf_datasets = [
    dict(
        abbr="text-4k-2k-10",
        type=FixedTokenDataset,
        path="fixed",
        input_tokens=4096,
        request_count=10,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
