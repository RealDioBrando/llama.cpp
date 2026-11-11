import os
import random

from datasets import Dataset
from transformers import AutoTokenizer

from ais_bench.benchmark.datasets import GSM8KDataset
from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.utils import get_data_path
from config_modules.runtime_config import dataset_config, model_path


class GSM8KSubsetDataset(BaseDataset):
    @staticmethod
    def load(path, subset_size=0):
        data = GSM8KDataset.load(get_data_path(path, local_mode=True))
        if subset_size > 0:
            data["test"] = data["test"].select(range(min(subset_size, len(data["test"]))))
        return data


class GSM8KFixedTokenDataset(BaseDataset):
    @staticmethod
    def load(path, subset_size=10, input_tokens=4096, dataset_name="gsm8k_4k_2k_10"):
        data = GSM8KSubsetDataset.load(path, subset_size)
        tokenizer_model = os.environ.get("QWEN3_1_7B_PATH") or os.environ.get("TOKENIZER_PATH")
        if not tokenizer_model:
            tokenizer_model = model_path(dataset_config(dataset_name).get("tokenizer_model", ""))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=False)

        vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        special_ids = set(tokenizer.all_special_ids)
        valid_ids = [i for i in range(vocab_size) if i not in special_ids]
        separator_ids = tokenizer.encode("\n\n", add_special_tokens=False)
        items = []
        for row in data["test"]:
            ids = tokenizer.encode(row["question"], add_special_tokens=False)
            if len(ids) > input_tokens:
                ids = ids[:input_tokens]
            else:
                while len(ids) < input_tokens:
                    ids.extend(ids + separator_ids)
                ids = ids[:input_tokens]

            text = tokenizer.decode(ids)
            re_ids = tokenizer.encode(text, add_special_tokens=False)
            while len(re_ids) > input_tokens:
                re_ids.pop()
            while len(re_ids) < input_tokens:
                re_ids.append(random.choice(valid_ids))
            final_text = tokenizer.decode(re_ids)
            verify_ids = tokenizer.encode(final_text, add_special_tokens=False)
            if len(verify_ids) != input_tokens:
                verify_ids = verify_ids[:input_tokens]
                while len(verify_ids) < input_tokens:
                    verify_ids.append(random.choice(valid_ids))
                final_text = tokenizer.decode(verify_ids)
            items.append({"question": final_text, "answer": row["answer"]})
        return Dataset.from_list(items)
