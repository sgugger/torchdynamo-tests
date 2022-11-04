Repo to test torchdynamo

## Experiments

This folder contains quick reproducers to observe different behaviors. The base scripts give a benchmark without any optimization. Then we can observe what happens when optimizing different things. This table regroups the average iteration time (alle executed on an A100):

Batch size 16:

| Script | FP32 | FP16 |
|:--|:-:|:-:|
| base | 105.74ms | 62.24ms |
| optimize_model | 104.70ms | 29.85ms |
| optimize_forward | 104.73ms | 29.49ms |
| train_step | x | x |

Batch size 8:

| Script | FP32 | FP16 |
|:--|:-:|:-:|
| base | 62.67ms | 59.68ms |
| optimize_model | 61.54ms | 23.80ms |
| optimize_forward | 61.84ms | 29.34ms |
| train_step | 1754.47ms | 1740.21ms |

Using torchdynamo to optimize the train step does not really work lots of warning like this and the times are not right:

```
[2022-11-04 15:32:56,201] torch._dynamo.optimizations.training: [WARNING] Unable to use Aot Autograd because of presence of mutation
[2022-11-04 15:32:56,201] torch._inductor.compile_fx: [WARNING] Aot Autograd is not safe to run, so falling back to eager
```

Reproducer: `python experiments/optimize_train_step_fp16.py`

