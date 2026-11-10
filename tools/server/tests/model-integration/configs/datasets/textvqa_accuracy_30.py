from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from config_modules.local_textvqa import LocalTextVQADataset
from config_modules.local_textvqa import LocalTextVQAEvaluator


reader_cfg = dict(
    input_columns=["question", "image"],
    output_column="answer",
    max_tokens_column="max_tokens",
)

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            type="image_text",
            data=["image_url", "text"],
            prompt=" Answer the question using a single word or phrase.",
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

eval_cfg = dict(evaluator=dict(type=LocalTextVQAEvaluator))

textvqa_datasets = [
    dict(
        abbr="textvqa-accuracy-30",
        type=LocalTextVQADataset,
        path="assets/textvqa/textvqa_json/textvqa_val_30.jsonl",
        image_type="image_base64",
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
