import abc
from typing import Dict

import polars as pl


class DataSplitter(abc.ABC):
    def __init__(self, train_frac=0.8, val_frac=0.1, test_frac=0.1):
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

    @abc.abstractmethod
    def split(self, df: pl.DataFrame, **kwargs) -> Dict[str, pl.DataFrame]:
        pass
