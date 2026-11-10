import json
from pathlib import Path

from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, EVALUATORS


@EVALUATORS.register_module()
class NullEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        return {"perf_only": 1.0}


@LOAD_DATASET.register_module()
class LocalImagePerfDataset(BaseDataset):
    @staticmethod
    def load(path):
        with Path(path).open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            row["image"] = row["path"][0]
        return Dataset.from_list(rows)
