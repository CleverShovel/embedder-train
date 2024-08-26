import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import (
    BatchAllTripletLoss,
    BatchHardTripletLoss,
    BatchSemiHardTripletLoss,
    BatchHardSoftMarginTripletLoss,
)
from sentence_transformers.losses.BatchHardTripletLoss import (
    BatchHardTripletLossDistanceFunction,
)
from sentence_transformers.training_args import BatchSamplers
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from embedder_train.utils.distance import dot_distance


def parse_args():
    parser = argparse.ArgumentParser(description="Triplet Loss Training Script")

    # Добавляем аргументы
    parser.add_argument(
        "--text_col", default="Суть ситуации", type=str, help="Name of the text column"
    )
    parser.add_argument(
        "--label_col", default="Id", type=str, help="Name of the label column"
    )
    parser.add_argument(
        "--data_volume",
        default="train-embedder-cpu-datavol-1",
        type=str,
        help="Data volume path",
    )
    parser.add_argument(
        "--dataset_path", default=None, type=str, help="Override default dataset path"
    )
    parser.add_argument(
        "--data_folds_path",
        default=None,
        type=str,
        help="Override default data folds path",
    )
    parser.add_argument(
        "--eval_data_path",
        default=None,
        type=str,
        help="Override default evaluators data path",
    )
    parser.add_argument(
        "--artifacts_dir",
        default=None,
        type=str,
        help="Override default artifacts directory",
    )
    parser.add_argument(
        "--checkpoints_dir",
        default=None,
        type=str,
        help="Override default checkpoints directory",
    )
    parser.add_argument(
        "--experiment_name",
        default=None,
        type=str,
        help="Override default experiment name",
    )
    parser.add_argument(
        "--base_model",
        default="sergeyzh/rubert-tiny-turbo",
        type=str,
        help="Base model path or name",
    )
    parser.add_argument(
        "--max_seq_length", default=512, type=int, help="Maximum sequence length"
    )
    parser.add_argument(
        "--distance_metric", default="cosine", type=str, help="Distance metric to use"
    )
    parser.add_argument(
        "--loss_func",
        default="BatchSemiHardTripletLoss",
        type=str,
        help="Loss function to use. One of BatchAllTripletLoss, BatchHardTripletLoss, BatchSemiHardTripletLoss, BatchHardSoftMarginTripletLoss",
    )
    parser.add_argument(
        "--margin",
        default=5.0,
        type=float,
        help="Margin for loss function if applicable",
    )
    parser.add_argument(
        "--seed", default=545454663, type=int, help="Random seed for initialization"
    )
    parser.add_argument(
        "--max_steps",
        default=1000,
        type=int,
        help="Total number of training steps to perform",
    )
    parser.add_argument(
        "--log_steps", default=100, type=int, help="Number of steps to measure metrics"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=256,
        type=int,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=32,
        type=int,
        help="Batch size per device for evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help="Warmup ratio for learning rate schedule",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--metric_for_best_model",
        default="eval_ffl_triplet_val_cosine_accuracy",
        type=str,
        help="Metric name to use for best model selection",
    )
    parser.add_argument(
        "--torch_compile",
        default=True,
        action="store_true",
        help="Enable TorchScript compilation",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        default=True,
        action="store_true",
        help="Pin memory for DataLoader",
    )
    parser.add_argument(
        "--batch_sampler",
        default="default",
        type=str,
        help="Batch sampler type, one of 'default', 'no_duplicates' or 'group_by_label'",
    )
    parser.add_argument("--use_cpu", default=False, action="store_true", help="Use cpu")

    # Парсим аргументы
    args = parser.parse_args()

    return args


@dataclass
class Config:
    args: any

    # Columns
    text_col: str = field(init=False)
    label_col: str = field(init=False)
    preprocessed_text_col: str = "preprocessed_text"
    preprocessed_label_col: str = "preprocessed_label"

    # Paths
    data_volume: Path = field(init=False)
    dataset_path: Path = field(init=False)
    data_folds_path: Path = field(init=False)
    eval_data_path: Path = field(init=False)
    train_path: Path = field(init=False)
    val_path: Path = field(init=False)
    test_path: Path = field(init=False)
    val_triplets_path: Path = field(init=False)
    test_triplets_path: Path = field(init=False)
    base_model: str = field(init=False)
    artifacts_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    experiment_name: str = field(init=False)
    result_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    # Model parameters
    max_seq_length: int = field(init=False)

    # Training parameters
    distance_metric: str = field(init=False)
    loss_func: str = field(init=False)
    margin: float = field(init=False)
    seed: int = field(init=False)
    max_steps: int = field(init=False)
    log_steps: int = field(init=False)
    per_device_train_batch_size: int = field(init=False)
    per_device_eval_batch_size: int = field(init=False)
    learning_rate: float = field(init=False)
    warmup_ratio: float = field(init=False)
    gradient_checkpointing: bool = field(init=False)
    metric_for_best_model: str = field(init=False)
    lr_scheduler_kwargs: dict = field(init=False)
    torch_compile: bool = field(init=False)
    dataloader_pin_memory: bool = field(init=False)
    batch_sampler: str = field(init=False)
    use_cpu: bool = field(init=False)

    def __post_init__(self):
        # Columns
        self.text_col = self.args.text_col
        self.label_col = self.args.label_col

        # Paths
        self.data_volume = Path(self.args.data_volume)
        self.dataset_path = (
            Path(self.args.dataset_path)
            if self.args.dataset_path
            else self.data_volume / "data" / "dataset-2"
        )
        self.data_folds_path = (
            Path(self.args.data_folds_path)
            if self.args.data_folds_path
            else self.dataset_path / "folds"
        )
        self.eval_data_path = (
            Path(self.args.eval_data_path)
            if self.args.eval_data_path
            else self.dataset_path / "eval"
        )
        self.train_path = self.data_folds_path / "train_data.parquet"
        self.val_path = self.data_folds_path / "val_data.parquet"
        self.test_path = self.data_folds_path / "test_data.parquet"
        self.val_triplets_path = self.eval_data_path / "val_triplets.parquet"
        self.test_triplets_path = self.eval_data_path / "test_triplets.parquet"

        self.base_model = self.args.base_model
        self.artifacts_dir = (
            Path(self.args.artifacts_dir)
            if self.args.artifacts_dir
            else self.data_volume / "artifacts"
        )
        self.current_time_str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        self.experiment_name = (
            self.args.experiment_name
            if self.args.experiment_name
            else f"{self.current_time_str}_{Path(self.base_model).stem}"
        )
        self.outputs_dir = self.artifacts_dir / self.experiment_name
        self.checkpoints_dir = (
            Path(self.args.checkpoints_dir)
            if self.args.checkpoints_dir
            else self.artifacts_dir / Path(self.base_model).stem
        )
        self.result_dir = self.outputs_dir / "weights"
        self.metrics_dir = self.outputs_dir / "metrics"
        self.logs_dir = self.outputs_dir / "logs"

        # Model parameters
        self.max_seq_length = self.args.max_seq_length

        # Training parameters
        self.distance_metric = self.args.distance_metric
        self.loss_func = self.args.loss_func
        self.margin = self.args.margin
        self.seed = self.args.seed
        self.max_steps = self.args.max_steps
        self.log_steps = self.args.log_steps
        self.per_device_train_batch_size = self.args.per_device_train_batch_size
        self.per_device_eval_batch_size = self.args.per_device_eval_batch_size
        self.learning_rate = self.args.learning_rate
        self.warmup_ratio = self.args.warmup_ratio
        self.gradient_checkpointing = self.args.gradient_checkpointing
        self.metric_for_best_model = self.args.metric_for_best_model
        self.lr_scheduler_kwargs = {
            "num_warmup_steps": int(self.warmup_ratio * self.max_steps),
            "num_training_steps": self.max_steps,
            "num_cycles": 10,
            "last_epoch": -1,
        }
        self.torch_compile = self.args.torch_compile
        self.dataloader_pin_memory = self.args.dataloader_pin_memory
        self.batch_sampler = self.args.batch_sampler
        self.use_cpu = self.args.use_cpu

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def prepare_datasets(
    train_path,
    val_path,
    test_path,
    text_col="text",
    label_col="label",
    preprocessed_label_col="label_",
    min_num_examples_threshold=2,
):
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)

    classes = set(
        [
            *train_df[label_col].unique().to_list(),
            *val_df[label_col].unique().to_list(),
            *test_df[label_col].unique().to_list(),
        ]
    )

    inverse_mapping = {c: idx for idx, c in enumerate(classes)}

    train_df = train_df.with_columns(
        pl.col(label_col)
        .replace(inverse_mapping, return_dtype=pl.Int32)
        .alias(preprocessed_label_col)
    )
    val_df = val_df.with_columns(
        pl.col(label_col)
        .replace(inverse_mapping, return_dtype=pl.Int32)
        .alias(preprocessed_label_col)
    )
    test_df = test_df.with_columns(
        pl.col(label_col)
        .replace(inverse_mapping, return_dtype=pl.Int32)
        .alias(preprocessed_label_col)
    )

    min_num_examples_threshold = 2

    train_df = train_df.filter(
        pl.col(label_col).count().over(label_col) >= min_num_examples_threshold
    )

    def process_hf_dataset(dataset):
        dataset = dataset.select_columns([text_col, preprocessed_label_col])
        dataset = dataset.rename_columns(
            {text_col: "sentence", preprocessed_label_col: "label"}
        )
        return dataset

    train_dataset = process_hf_dataset(Dataset.from_polars(train_df))
    val_dataset = process_hf_dataset(Dataset.from_polars(val_df))
    test_dataset = process_hf_dataset(Dataset.from_polars(test_df))

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}


def prepare_evaluator(triplets_path, fold="val", batch_size=32):
    triplets_df = pl.read_parquet(triplets_path)
    anchors = triplets_df["anchors"].to_list()
    positives = triplets_df["positives"].to_list()
    negatives = triplets_df["negatives"].to_list()

    evaluator = TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name=f"ffl_triplet_{fold}",
        batch_size=batch_size,
        main_distance_function="cosine",
    )

    return evaluator


def init_model(base_model, max_seq_length=512):
    model = SentenceTransformer(base_model)
    model.max_seq_length = max_seq_length
    return model


def get_distance_func(distance_metric):
    distance_func = BatchHardTripletLossDistanceFunction.cosine_distance
    if distance_metric == "dot":
        distance_func = dot_distance
    return distance_func


def prepare_loss(model, loss_name, **args):
    if loss_name == "BatchSemiHardTripletLoss":
        distance_metric = args["distance_metric"]
        margin = args["margin"]
        loss = BatchSemiHardTripletLoss(
            model, distance_metric=distance_metric, margin=margin
        )
    elif loss_name == "BatchHardTripletLoss":
        distance_metric = args["distance_metric"]
        margin = args["margin"]
        loss = BatchHardTripletLoss(
            model, distance_metric=distance_metric, margin=margin
        )
    elif loss_name == "BatchHardSoftMarginTripletLoss":
        distance_metric = args["distance_metric"]
        loss = BatchHardSoftMarginTripletLoss(model, distance_metric=distance_metric)
    elif loss_name == "BatchAllTripletLoss":
        distance_metric = args["distance_metric"]
        margin = args["margin"]
        loss = BatchAllTripletLoss(
            model, distance_metric=distance_metric, margin=margin
        )
    return loss


def get_batch_sampler(batch_sampler):
    if batch_sampler == "default":
        return BatchSamplers.BATCH_SAMPLER
    elif batch_sampler == "no_duplicates":
        return BatchSamplers.NO_DUPLICATES
    elif batch_sampler == "group_by_label":
        return BatchSamplers.GROUP_BY_LABEL
    raise ValueError(f"unknown batch_sampler option '{batch_sampler}'")


def train(
    checkpoints_dir="checkpoints/",
    max_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    log_steps=50,
    seed=42,
    metric_for_best_model="accuracy",
    gradient_checkpointing=False,
    torch_compile=False,
    dataloader_pin_memory=True,
    model=None,
    train_dataset=None,
    val_dataset=None,
    loss=None,
    val_evaluator=None,
    optimizer=None,
    lr_scheduler=None,
    batch_sampler=BatchSamplers.BATCH_SAMPLER,
    use_cpu=False,
):
    train_args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=checkpoints_dir,
        # Optional training parameters:
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=not use_cpu,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=batch_sampler,  # losses that use "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=log_steps,
        save_strategy="steps",
        save_steps=log_steps,
        save_total_limit=2,
        logging_steps=log_steps,
        run_name="train-experiment",
        report_to="none",
        use_cpu=use_cpu,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        dataloader_num_workers=4,
        gradient_checkpointing=gradient_checkpointing,
        torch_compile=torch_compile,
        dataloader_pin_memory=dataloader_pin_memory,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=val_evaluator,
        optimizers=(optimizer, lr_scheduler),
    )
    trainer.train()


def save_experiment_params(config, result_dir):
    dataset_params = dict(
        text_col=config.text_col,
        label_col=config.label_col,
        preprocessed_text_col=config.preprocessed_text_col,
        preprocessed_label_col=config.preprocessed_label_col,
        data_volume=str(config.data_volume),
        dataset_path=str(config.dataset_path),
        data_folds_path=str(config.data_folds_path),
        eval_data_path=str(config.eval_data_path),
    )

    model_params = dict(
        base_model=config.base_model, max_seq_length=config.max_seq_length
    )

    train_params = dict(
        distance_metric=config.distance_metric,
        loss=config.loss_func,
        margin=config.margin,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        lr_scheduler_kwargs=config.lr_scheduler_kwargs,
        torch_compile=config.torch_compile,
        dataloader_pin_memory=config.dataloader_pin_memory,
        batch_sampler=config.batch_sampler,
    )

    params = {
        "dataset_params": dataset_params,
        "train_params": train_params,
        "model_params": model_params,
    }

    result_dir = Path(result_dir)
    with open(result_dir / "experiment_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f)


def evaluate_and_save_results(model, evaluator, fold="test", metrics_dir="metrics/"):
    metrics_dir = Path(metrics_dir)
    results = evaluator(model)
    test_metrics = dict(results)
    with open(metrics_dir / f"{fold}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f)


def main():
    args = parse_args()

    config = Config(args=args)

    datasets = prepare_datasets(
        config.train_path,
        config.val_path,
        config.test_path,
        text_col=config.text_col,
        label_col=config.label_col,
        preprocessed_label_col=config.preprocessed_label_col,
        min_num_examples_threshold=2,
    )

    if config.eval_data_path.exists() and any(config.eval_data_path.iterdir()):
        val_evaluator = prepare_evaluator(
            config.val_triplets_path,
            fold="val",
            batch_size=config.per_device_eval_batch_size,
        )

        test_evaluator = prepare_evaluator(
            config.test_triplets_path,
            fold="test",
            batch_size=config.per_device_eval_batch_size,
        )
    else:
        val_evaluator = None
        test_evaluator = None

    logging.basicConfig(
        filename=str(config.logs_dir / "train_script.log"),
        level=logging.INFO,
    )

    model = init_model(config.base_model, max_seq_length=config.max_seq_length)
    distance_metric = get_distance_func(config.distance_metric)
    loss = prepare_loss(
        model, config.loss_func, distance_metric=distance_metric, margin=config.margin
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, fused=True)
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, **config.lr_scheduler_kwargs
    )
    batch_sampler = get_batch_sampler(config.batch_sampler)

    train(
        checkpoints_dir=config.checkpoints_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        log_steps=config.log_steps,
        metric_for_best_model=config.metric_for_best_model,
        torch_compile=config.torch_compile,
        dataloader_pin_memory=config.dataloader_pin_memory,
        model=model,
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        loss=loss,
        val_evaluator=val_evaluator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        batch_sampler=batch_sampler,
        use_cpu=config.use_cpu,
    )

    model.save_pretrained(str(config.result_dir))

    save_experiment_params(config, config.result_dir)

    if test_evaluator is not None:
        evaluate_and_save_results(model, test_evaluator, metrics_dir=config.metrics_dir)


if __name__ == "__main__":
    main()
