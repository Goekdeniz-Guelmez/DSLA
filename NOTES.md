# Project Notes

## Naming

The method is named **DSLA**: **D**irectional and **S**imilarity-aware **L**atent **A**lignment for Preference Optimization. Earlier local run folders and notes may use the historical name `DLPO` and labels such as `dlpo-orpo` or `dlpo-dpo`; the paper uses `DSLA-ORPO` and `DSLA-DPO` for the corresponding method variants.

## How to read the results

The paper's strongest evidence is that its latent regularizer changes chosen–rejected hidden-state geometry. Improvements in generated-response preference are heterogeneous across datasets and objective families. Treat the generation results as small-sample diagnostics and the shortcut analyses as scoped tests of the specific features examined.

The earlier local `NOTES.md` contains exploratory interpretations from different run subsets. Those notes are not copied into this release as headline evidence. The paper and [EXPERIMENTS.md](EXPERIMENTS.md) summarize the results intended for readers.

## Follow-up work

- Repeat the matched comparisons across multiple training seeds.
- Increase the held-out generation sample size and evaluate on external preference benchmarks.
- Test the batch-direction term with controlled effective batch sizes greater than one.
- Check whether layer and pooling findings transfer across model families and datasets.
- Publish the exact MLX-LM-LoRA release, version, and reproduction interface alongside the package update.

## Release contents

This folder is the public-facing snapshot. Keep it limited to the paper PDF and reader documentation; executable training code and model artifacts belong to their respective package or model releases.
