from typing import Optional

import polars as pl
from sklearn.model_selection import train_test_split

from embedder_train.processing.mask import DataSplitter


class StratifiedDataSplitter(DataSplitter):
    def __init__(self, train_frac=0.8, val_frac=0.1, test_frac=0.1, label_col="label", seed=0):
        super().__init__(train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)
        self.label_col = label_col
        self.seed = seed

    def split(self, df: pl.DataFrame, seed: Optional[int] = None, **kwargs):
        seed = seed or self.seed

        big_class_cond = pl.col(self.label_col).count().over(self.label_col) * (self.val_frac + self.test_frac) >= 2
        equal_dist_cond = pl.col(self.label_col).count().over(self.label_col) >= 3

        big_classes_df = df.filter(big_class_cond)  # делится по заданным долям
        small_classes_df = df.filter(~big_class_cond)

        equal_classes_df = small_classes_df.filter(equal_dist_cond)  # делится поровну на все фолды

        not_equal_classes_df = small_classes_df.filter(~equal_dist_cond)  # дублируются во все фолды

        train_size = int(len(big_classes_df) * self.train_frac)
        val_size = int(len(big_classes_df) * self.val_frac)
        test_size = len(big_classes_df) - train_size - val_size

        train_df, val_test_df = train_test_split(
            big_classes_df,
            test_size=val_size + test_size,
            random_state=seed,
            stratify=big_classes_df[self.label_col].to_list(),
        )
        val_df, test_df = train_test_split(
            val_test_df,
            test_size=test_size,
            random_state=seed,
            stratify=val_test_df[self.label_col].to_list(),
        )

        if len(equal_classes_df) > 0:
            add_train_df, add_val_test_df = train_test_split(
                equal_classes_df,
                test_size=0.66,
                random_state=seed,
                stratify=equal_classes_df[self.label_col].to_list(),
            )
            add_val_df, add_test_df = train_test_split(
                add_val_test_df,
                test_size=0.5,
                random_state=seed,
                stratify=add_val_test_df[self.label_col].to_list(),
            )

            train_df = pl.concat([train_df, add_train_df, not_equal_classes_df])
            val_df = pl.concat([val_df, add_val_df, not_equal_classes_df])
            test_df = pl.concat([test_df, add_test_df, not_equal_classes_df])

        return {"train": train_df, "val": val_df, "test": test_df}
