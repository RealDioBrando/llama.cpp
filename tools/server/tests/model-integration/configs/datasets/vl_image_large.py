from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

from config_modules.local_mm_perf import LocalImagePerfDataset, NullEvaluator
from config_modules.runtime_config import dataset_config

dataset_cfg = dataset_config("vl_image_large")


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
