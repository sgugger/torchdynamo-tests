Repo to test torchdynamo

`pip install -r requirements.txt`

## Experiments

This folder contains quick reproducers to observe different behaviors. The base scripts give a benchmark without any optimization. Then we can observe what happens when optimizing different things. This table regroups the average iteration time (alle executed on an A100):

Batch size 16:

| Script | FP32 | FP16 |
|:--|:-:|:-:|
| base | 54.44ms | 62.24ms |
| optimize_model | 38.20ms | 29.85ms |
| optimize_forward | 38.36ms | 29.49ms |
| train_step | x | x |

Batch size 8:

| Script | FP32 | FP16 |
|:--|:-:|:-:|
| base | 53.47ms | 59.68ms |
| optimize_model | 28.34ms | 23.80ms |
| optimize_forward | 28.16ms | 29.34ms |
| train_step | 1754.47ms | 1740.21ms |

Using torchdynamo to optimize the train step does not really work lots of warning like this and the times are not right:

```
[2022-11-04 15:32:56,201] torch._dynamo.optimizations.training: [WARNING] Unable to use Aot Autograd because of presence of mutation
[2022-11-04 15:32:56,201] torch._inductor.compile_fx: [WARNING] Aot Autograd is not safe to run, so falling back to eager
```

Reproducer: `python experiments/optimize_train_step_fp16.py`

Dynamic

| Script | FP32 | FP16 |
|:--|:-:|:-:|
| dynamic | 59.23ms | 63.53ms |
| dyanmic_optimized | OOM? | OOM? |

## Scripts

### Text classification

Iteration avg time for training/evaluation (excluding the first) when fine-tuning BERT on MRPC. Final results of the models are within the variance of fine-tuning, no particular performance drop observed except for FP16 + torchdynamo which seems to underperform a bit (more like 82%-84% accuracy compared to 86%-87% for other tests and 0.86/0.87 F1 score instead of 0.89/0.90).

| Dynamo | FP32 | FP16 |
|:--|:-:|:-:|
| no | 57.9ms/15.65ms | 65.87ms/18.52ms |
| inductor | 36.24ms/10.55ms | 39.43ms/9.09ms |

To reproduce:

```bash
accelerate launch --config_file configs/base_fp32.yaml scripts/text_classification.py --task_name mrpc
```

and change the config file to one of the four options in `configs` to get the four squares.


### Language Modeling

```bash
accelerate launch scripts/language_modeling.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --dynamo_backend inductor
```

### Vision Classification 

```bash
accelerate launch scripts/cv_classification.py \
    --model_name_or_path microsoft/resnet-18 \
    --dynamo_backend inductor
```