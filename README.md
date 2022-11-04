Repo to test torchdynamo

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

