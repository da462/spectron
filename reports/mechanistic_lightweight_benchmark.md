# Lightweight diagnostics benchmark

Date: 2026-09-01. Hardware: 4 x H100 on Jean Zay (`gpu_p6`, H100 dev QoS).
Representative configuration: 134M dense reference shape, FFN-only rank-0.25
factorization, embedding std 0.02, base LR 0.05, Spectron FFN multiplier 14,
global batch 512 x 2048 tokens. Each smoke ran 20 optimizer steps; medians below
exclude the first five steps.

| Mode | Job | Median TPS | Aggregate TPS | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Disabled | 1646912 | 951,827 | 951,246 | 29.779 GiB | 32.920 GiB |
| Cheap every-step scalars | 1646913 | 904,701 | 905,954 | 29.781 GiB | 32.920 GiB |
| Cheap + product Grams every step | 1646914 | 770,524 | 772,184 | 29.781 GiB | 32.920 GiB |
| Cheap + product Grams every 5 steps | 1646915 | 903,239 ordinary | 875,713 | 29.780 GiB | 32.920 GiB |
| Sparse heavy checkpoints | 1646916 | 944,732 ordinary | n/a | 29.777 GiB | 32.922 GiB |

Cheap scalar diagnostics reduce aggregate throughput by about 4.8%. Product
Grams every step reduce it by about 18.8%, so they fail the requested 2-3%
incremental-overhead threshold. Recording product Grams every five steps gives
about 8.0% total slowdown versus disabled and 3.3% incremental slowdown versus
cheap-only diagnostics. The mechanistic matrix therefore defaults to a product
interval of five.

Activation scalars are measured on the first real rank-0 training microbatch
each optimizer step (16 x 2048 = 32,768 tokens). Hooks deactivate after the
logits reduction. They never capture fixed diagnostic/evaluation forwards and
retain only scalar sufficient statistics.
