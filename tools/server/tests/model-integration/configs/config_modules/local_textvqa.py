import base64
import json
import re
from pathlib import Path

from datasets import Dataset

from ais_bench.benchmark.datasets import TEXTEvaluator
from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.registry import LOAD_DATASET
REPO_ROOT = Path(__file__).resolve().parents[2]


@LOAD_DATASET.register_module()
class LocalTextVQADataset(BaseDataset):
    @staticmethod
    def load(path, image_type="image_base64"):
        jsonl_path = Path(path)
        if not jsonl_path.is_absolute():
            jsonl_path = REPO_ROOT / jsonl_path
        annotations_path = jsonl_path.with_name(f"{jsonl_path.stem}_annotations.json")

        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
        answers = {
            int(item["question_id"]): item["answers"]
            for item in annotations.get("annotations", [])
        }

        rows = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["answer"] = answers[int(row["question_id"])]
                row["max_tokens"] = 16
                if image_type == "image_base64":
                    image_path = Path(row["image"])
                    if not image_path.is_absolute():
                        image_path = REPO_ROOT / image_path
                    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
                    row["image"] = {"url": f"data:image/jpeg;base64,{image_data}"}
                row["image_url"] = row["image"]
                rows.append(row)
        return Dataset.from_list(rows)


class LocalTextVQAEvaluator(TEXTEvaluator):
    """VQA scoring that accepts a short reference answer as a response prefix."""

    @staticmethod
    def _normalize(text):
        return " ".join(re.findall(r"[a-z0-9]+", text.lower())).strip()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {"error": "predictions and references have different length"}

        correct = 0
        details = []
        for prediction, reference in zip(predictions, references):
            pred = self._normalize(prediction)
            refs = [self._normalize(item["answer"]) for item in reference]
            matches = sum(bool(pred) and (pred == ref or pred.startswith(ref + " ")) for ref in refs)
            accuracy = min(1.0, matches / 3)
            correct += accuracy
            details.append({
                "pred": prediction,
                "answer": reference,
                "correct": accuracy > 0.5,
                "accuracy": accuracy,
            })

        return {"accuracy": 100 * correct / len(predictions), "details": details}
