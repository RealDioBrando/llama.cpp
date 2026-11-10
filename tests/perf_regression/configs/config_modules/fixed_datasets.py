import base64
import os
import random

from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from transformers import AutoTokenizer


class NullEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        return {"perf_only": 1.0}


class FixedTokenDataset(BaseDataset):
    @staticmethod
    def load(path, input_tokens=4096, request_count=10):
        model_path = os.environ.get("QWEN3_1_7B_PATH") or os.environ.get("TOKENIZER_PATH")
        if not model_path:
            raise ValueError("QWEN3_1_7B_PATH or TOKENIZER_PATH must be set")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        special_ids = set(tokenizer.all_special_ids)
        valid_ids = [i for i in range(vocab_size) if i not in special_ids]
        items = []
        for _ in range(request_count):
            # generate random tokens and decode to text
            # re-encode to verify token count, trim or pad to hit exact target
            ids = random.sample(valid_ids, input_tokens)
            text = tokenizer.decode(ids)
            re_ids = tokenizer.encode(text, add_special_tokens=False)
            # trim if decode produced more tokens
            while len(re_ids) > input_tokens:
                re_ids.pop()
            # pad if decode produced fewer tokens
            while len(re_ids) < input_tokens:
                re_ids.append(random.choice(valid_ids))
            # verify: final text must re-tokenize to exactly input_tokens
            final_text = tokenizer.decode(re_ids)
            verify_ids = tokenizer.encode(final_text, add_special_tokens=False)
            if len(verify_ids) != input_tokens:
                # fallback: trim again
                verify_ids = verify_ids[:input_tokens]
                while len(verify_ids) < input_tokens:
                    verify_ids.append(random.choice(valid_ids))
                final_text = tokenizer.decode(verify_ids)
            items.append({"question": final_text, "answer": "perf_only"})
        return Dataset.from_list(items)


class FixedImageDataset(BaseDataset):
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            image = base64.b64encode(f.read()).decode("ascii")
        return Dataset.from_list(
            [{"image": image, "question": "Describe this image briefly.", "answer": "perf_only"}]
        )
