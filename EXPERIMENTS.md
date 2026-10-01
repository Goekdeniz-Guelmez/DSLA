# Experiment Summary

This page summarizes selected training, validation, generation, and ablation runs reported in the paper. It is a concise reading guide, not a dump of raw logs or a claim of benchmark-level confirmation. The paper contains the full result tables, plots, and protocol details.

## Study design

The experiments compare matched DPO and ORPO baselines with their DSLA-regularized variants. They cover Qwen2.5, Qwen3, and Llama 3 instruction models, including quantized variants, and two preference settings: Human-Like-DPO and a hand-curated JOSIE preference set. The primary comparison matrix uses a fixed seed (`42`), 128 held-out validation examples, and 16 generated validation examples per run. Separate ablations study extraction layer, latent-loss weight, and pooling.

The shared setup reported in the paper uses a learning rate of `1e-5`, DPO `beta=0.1`, ORPO `alpha=0.1`, a maximum sequence length of 2048, LoRA rank 8 and scale 20 on 16 layers, and (for DSLA) latent margin `0.05`, scale `10`, both latent terms, answer-mean pooling, and final-layer states. The paper reports individual deviations for its ablation runs.

## Main observations

### Hidden-state separation

The most consistent result is geometric: every matched comparison in the paper's generation tables reports a larger chosen–rejected latent distance for the DSLA variant than for its probability-only baseline. Selected examples:

| Matched setting | Baseline latent distance | DSLA latent distance |
|---|---:|---:|
| DPO, Human-Like-DPO | 0.1042 | 0.6931 |
| DPO, JOSIE preference set | 0.0850 | 0.4655 |
| ORPO, Qwen2.5-0.5B, JOSIE preference set | 0.0496 | 0.8789 |
| ORPO, Qwen2.5-1.5B, JOSIE preference set | 0.2714 | 0.8912 |

Latent distance measures hidden-state separation under the paper's extraction and pooling protocol. It does not establish that the separated direction represents a particular semantic quality such as truthfulness, helpfulness, or safety.

### Generated responses

Generation preferences do not move uniformly with latent distance. In the DPO family, the reported win rate rises from `0.6250` to `0.9375` on Human-Like-DPO, while it falls from `0.0625` to `0.0000` on the JOSIE validation generations. For the ORPO family, DSLA-ORPO improves three matched comparisons, ties six, and is lower in two. Each result is based on only 16 generated validation examples, so these values are descriptive diagnostics.

### Layer and pooling ablations

- In a Qwen2.5-0.5B, 600-example layer ablation, late and final extraction anchors outperform the middle anchor across the reported diagnostics. The paper recommends a late or final anchor for this tested setting; the result does not establish a universal best layer across architectures.
- In the corresponding pooling ablation, `last_token` pooling gives DSLA-ORPO validation preference accuracy `0.9375` and latent similarity margin `0.886`, compared with `0.875` and `0.513` for `answer_mean`. This is a small, single-setting ablation and should be treated as a candidate for further study.
- The latent-weight sweep shows that the hidden-state geometry changes as the regularizer weight changes, even when probability-level validation losses are close or saturated. A larger latent margin alone is not evidence of better generated responses.

### Shortcut diagnostics

The paper reports lower correlation between the learned preference direction and response length for DSLA-ORPO than for ORPO in two observational Human-Like-DPO settings: `0.563` to `0.312` for Qwen2.5-1.5B and `0.424` to `0.235` for Qwen2.5-0.5B. It also includes controlled tests with engineered surface features. These are focused diagnostics of the tested shortcuts; they do not establish that DSLA prevents reward hacking in general.

## Scope and limitations

- The primary comparison uses one training seed (`42`); seed-to-seed variance is not measured.
- Generation evaluations use 16 examples per run and are too small for benchmark-level conclusions.
- The training datasets and step budgets are compact; the study is compute-bounded and heterogeneous.
- Batch size one makes the batch-direction estimate degenerate. Results from those runs do not test the benefit of a multi-example directional estimate.
- Hidden-state separation is a method diagnostic, not a direct measure of response quality.

For complete tables, definitions, and plots, see [the paper](DSLA-paper.pdf).
