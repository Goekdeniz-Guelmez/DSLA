# Reproduction Notes

## Implementation

The experiment implementation is intended to be distributed through a forthcoming update to [MLX-LM-LoRA](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora). The package release, version number, and command-line interface should be taken from its release notes once that update is published. This repository contains no training implementation or executable commands.

## Experimental configuration

The paper's primary matched comparisons use the following reported defaults:

| Setting | Value |
|---|---|
| Preference objectives | DPO, DSLA-DPO, ORPO, DSLA-ORPO |
| Learning rate | `1e-5` |
| DPO beta | `0.1` |
| ORPO alpha | `0.1` |
| Maximum sequence length | `2048` tokens |
| LoRA | Rank `8`, scale `20`, dropout `0`, applied to `16` layers |
| DSLA latent margin | `0.05` |
| DSLA latent scale | `10` |
| DSLA terms | Similarity and directional terms (`both`) |
| Default pooling and extraction | Answer mean, final layer |
| Random seed | `42` |
| Validation split | `128` examples |
| Generation evaluation | `16` examples per run |

The paper's layer, latent-weight, and pooling ablations intentionally override these defaults. The primary dataset budgets are 500 examples; a separate layer and pooling study uses 600 examples. Exact per-run conditions and result definitions are in the paper.

## Models and data

The reported runs use instruction-tuned checkpoints from these families, with full-precision and quantized variants depending on the run:

- Qwen2.5: 0.5B, 1.5B, 3B, and 7B.
- Qwen3: 4B Instruct 2507.
- Llama 3: 3.1 8B and 3.2 1B.

The paper's preference datasets are Human-Like-DPO and a hand-curated JOSIE-2-Instruct-DPO set. The run manifest identifies them as [`mlx-community/Human-Like-DPO`](https://huggingface.co/datasets/mlx-community/Human-Like-DPO) and `Goekdeniz-Guelmez/JOSIE-2-Instruct-DPO-1K-Qwen3-4B-Gabliterated`, respectively. Confirm that the model and dataset revisions match the intended experiment; repository names alone may not identify an immutable revision.

## Comparison requirements

For a matched comparison, keep the model checkpoint and tokenizer, dataset revision, train/validation split, example budget, random seed, effective batch size, quantization, and LoRA settings the same between the baseline and DSLA runs. Record the package release and all per-run options alongside the results. The paper's single-seed, small-generation-evaluation protocol is not sufficient to estimate run-to-run uncertainty.

## What is included here

This public snapshot contains the paper PDF and Markdown summaries. It does not contain model weights, adapters, datasets, raw training logs, or the package implementation. The forthcoming package and the paper are the sources for executable reproduction details.
