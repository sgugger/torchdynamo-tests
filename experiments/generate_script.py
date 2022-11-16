import argparse
import random
import time
import torch

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Make a couple of generations")
    parser.add_argument("--model_name", type=str, default="gpt2")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    inputs = tokenizer(["Once upon a time,"] * 8, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    model = model.eval()
    model = accelerator.prepare(model)

    start_time = time.time()
    for step in range(50):
        batch = {k: v.to(accelerator.device) for k, v in inputs.items()}
        output = model.generate(**batch)
        if step == 0:
            first_step_time = time.time() - start_time

    total_training_time = time.time() - start_time
    avg_iteration_time = (total_training_time - first_step_time) / (50 - 1)
    print("Generations finished.")
    print(f"First iteration took: {first_step_time:.2f}s")
    print(f"Average time after the first iteration: {avg_iteration_time * 1000:.2f}ms")


if __name__ == "__main__":
    main()
