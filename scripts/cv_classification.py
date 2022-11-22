# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning any 🤗 Transformers model for image classification leveraging 🤗 Accelerate."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import time

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset)."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="google/vit-base-patch16-224-in21k",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--dynamo_backend", type=str, default="no", help="Dynamo backend")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(dynamo_backend=args.dynamo_backend)

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    dataset = load_dataset(args.dataset_name, task="image-classification")

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = dataset["train"].features["labels"].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Load pretrained model and feature extractor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
        i2label=id2label,
        label2id=label2id,
        finetuning_task="image-classification",
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=True,
    )

    # Preprocessing the datasets

    # Define torchvision transforms to be applied to each image.
    if "shortest_edge" in feature_extractor.size:
        size = feature_extractor.size["shortest_edge"]
    else:
        size = (feature_extractor.size["height"], feature_extractor.size["width"])
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    with accelerator.main_process_first():
        dataset["train"] = dataset["train"].shuffle(seed=args.seed)
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)
        dataset["validation"] = dataset["validation"].shuffle(seed=args.seed)
        # Set the validation transforms
        eval_dataset = dataset["validation"].with_transform(preprocess_val)

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Scheduler.
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Get the metric function
    metric = evaluate.load("accuracy")
    # Train!
    # Only show the progress bar once on each machine.
    train_steps = len(train_dataloader) * args.num_epochs
    progress_bar = tqdm(range(train_steps), disable=not accelerator.is_local_main_process)

    start_time = time.time()
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            predictions, references = accelerator.gather_for_metrics((outputs.logits.argmax(dim=-1), batch["labels"]))
            metric.add_batch(predictions=predictions, references=references)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            if step == 0 and epoch == 0:
                first_step_time = time.time() - start_time

        eval_train_metric = metric.compute()
        print(f"Training Accuracy for backend {args.dynamo_backend} at epoch {epoch}: {eval_train_metric}")

    total_training_time = time.time() - start_time
    avg_train_iteration_time = (total_training_time - first_step_time) / (train_steps - 1)
    print("Training finished.")
    print(f"First iteration took: {first_step_time:.2f}s")
    print(f"Average time after the first iteration: {avg_train_iteration_time * 1000:.2f}ms")

    model.eval()
    start_time = time.time()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        metric.add_batch(predictions=predictions, references=references)

        if step == 0:
            first_step_time = time.time() - start_time
    total_eval_time = time.time() - start_time
    avg_test_iteration_time = (total_eval_time - first_step_time) / (len(eval_dataloader) - 1)
    print("Evaluation finished.")
    print(f"First iteration took: {first_step_time:.2f}s")
    print(f"Average time after the first iteration: {avg_test_iteration_time * 1000:.2f}ms")

    eval_test_metric = metric.compute()
    print(f"Test Accuracy for backend {args.dynamo_backend}: {eval_test_metric}")

    out_dict = {
        "backend": args.dynamo_backend,
        "num_epochs": str(args.num_epochs),
        "seed": str(args.seed),
        "train_acc": str(eval_train_metric["accuracy"]),
        "avg_train_time": str(avg_train_iteration_time * 1000),
        "test_acc": str(eval_test_metric["accuracy"]),
        "avg_test_time": str(avg_test_iteration_time * 1000),
    }

    with open("cv_classification_results.csv", "a") as fd:
        fd.write("\n")
        fd.write(",".join(out_dict.values()))


if __name__ == "__main__":
    main()
