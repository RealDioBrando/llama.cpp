from ais_bench.benchmark.datasets import GSM8KDataset
from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.utils import get_data_path


class GSM8KSubsetDataset(BaseDataset):
    @staticmethod
    def load(path, subset_size=0):
        data = GSM8KDataset.load(get_data_path(path, local_mode=True))
        if subset_size > 0:
            data["test"] = data["test"].select(range(min(subset_size, len(data["test"]))))
        return data
