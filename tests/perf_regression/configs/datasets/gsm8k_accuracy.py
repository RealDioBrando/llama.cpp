from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator


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
        abbr="gsm8k-accuracy",
        type=GSM8KDataset,
        path="ais_bench/datasets/gsm8k",
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
