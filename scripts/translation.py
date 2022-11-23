#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on text translation.
"""
# You can also adapt this script on your own text translation task. Pointers for this are left as comments.

import argparse
import logging
import random
import time

import datasets
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    default_data_collator,
    get_scheduler,
)

torch.backends.cuda.matmul.allow_tf32 = True
logger = get_logger(__name__)


# Parsing input arguments
def parse_args():

    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="t5-small",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded unless `--dynamic_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--dynamic_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the dataloaders.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--dynamo_backend", type=str, default="no", help="Dynamo backend")
    parser.add_argument("--mixed_precision", type=str, default="no", help="`no` or `fp16`")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    accelerator = Accelerator(dynamo_backend=args.dynamo_backend, mixed_precision=args.mixed_precision)

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

    # Load data
    raw_datasets = load_dataset("wmt16", "ro-en")

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # MBART requires some language codes
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "ro_RO"
        if model.config.decoder_start_token_id is None:
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id["ro_RO"]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids("ro_RO")

    # T5 requires a prefix
    if args.model_name_or_path in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
        prefix = "translate English to Romanian: "
    else:
        prefix = ""

    # Preprocessing the datasets.
    padding = False if args.dynamic_length else "max_length"

    def preprocess_function(examples):
        inputs = [ex["en"] for ex in examples["translation"]]
        targets = [ex["ro"] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=args.max_length, padding=padding, truncation=True
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
            ]

        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if not args.dynamic_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.batch_size, drop_last=not args.dynamic_length
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

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

    # Metric
    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    # Train!
    # Only show the progress bar once on each machine.
    train_steps = min(len(train_dataloader) * args.num_epochs, 1000)
    progress_bar = tqdm(range(train_steps), disable=not accelerator.is_local_main_process)
    start_time = time.time()

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            if step == 0 and epoch == 0:
                first_step_time = time.time() - start_time
            elif step >= 1000:
                break

    total_training_time = time.time() - start_time
    avg_iteration_time = (total_training_time - first_step_time) / (train_steps - 1)
    print("Training finished.")
    print(f"First iteration took: {first_step_time:.2f}s")
    print(f"Average time after the first iteration: {avg_iteration_time * 1000:.2f}ms")

    model.eval()
    start_time = time.time()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"], attention_mask=batch["attention_mask"], max_length=args.max_length
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            if args.dynamic_length:
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            if step == 0:
                first_step_time = time.time() - start_time

    total_eval_time = time.time() - start_time
    avg_iteration_time = (total_eval_time - first_step_time) / (len(eval_dataloader) - 1)

    print("Evaluation finished.")
    print(f"First iteration took: {first_step_time:.2f}s")
    print(f"Average time after the first iteration: {avg_iteration_time * 1000:.2f}ms")

    eval_metric = metric.compute()
    print(f"Test BLEU score for backend {args.dynamo_backend}: {eval_metric['score']}")


if __name__ == "__main__":
    main()
