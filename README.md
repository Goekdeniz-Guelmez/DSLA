# DSLA: Directional and Similarity-aware Latent Alignment

DSLA is a representation-aware regularizer for preference optimization. It augments DPO- and ORPO-style objectives with two hidden-state terms: a prompt–response similarity margin and a batch-level directional alignment margin.

The paper and selected experimental summaries are collected here for public release.

## Contents

- [Research paper](DSLA-paper.pdf)
- [Selected experiment results](EXPERIMENTS.md)
- [Reproduction notes](REPRODUCIBILITY.md)
- [Research notes and open questions](NOTES.md)

## Current evidence

Across the reported matched runs, DSLA consistently increases chosen–rejected hidden-state separation. Generation results are mixed across datasets and objective variants. The experiments are small and use a single training seed, so the evidence supports a representation-level effect, not a claim that DSLA is generally better than DPO or ORPO.

## Using the method

The training implementation is intended to arrive in a forthcoming release of [MLX-LM-LoRA](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora). Check that package's release notes for the version and supported options when the update is published. This repository is a paper and documentation snapshot; it contains no Python source, training scripts, model adapters, or executable test suite.

## Citation

Gökdeniz Gülmez. *DSLA: Directional and Similarity-aware Latent Alignment for Preference Optimization: Augmenting DPO and ORPO with a Hidden-State Preference Regularizer.* Research manuscript, 2026. See [DSLA-paper.pdf](DSLA-paper.pdf).
