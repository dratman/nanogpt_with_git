# nanogpt Usage Guide

This document covers:
- Organization philosophy (code vs. artifacts separation)
- Directory structure and branches
- Training: scripts, presets, and parameters
- Sampling from trained models

## Organization Philosophy

This project uses a "compiler-style" separation:

| What | Role | Location |
|------|------|----------|
| **Code** | The "compiler" | `nanogpt/` (this git repo) |
| **Training data** | "Source files" to compile | `nanogpt_artifacts/txt_local/` |
| **Checkpoints** | "Compiled output" | `nanogpt_artifacts/pt/` |
| **Utility scripts** | Ad-hoc tools | `nanogpt_artifacts/scripts/` |

The git repository contains only the code itself. All inputs and outputs live
outside the repo in `nanogpt_artifacts/`. This keeps the repo clean and avoids
tracking large binary files.

## Directory Structure

```
~/0-Home-Working-on-M3-Pro/
    nanogpt/                        # Git repo (code only)
        py/                         # Python source
            train.py                # Training script
            sample.py               # Sampling script
            model.py                # Model definition
            tokenizer.py            # Tokenizer classes
        sh/                         # Shell scripts
            train.sh                # Wrapper with logging
            train_sentence.sh       # Preset for sentence training
        doc/                        # Documentation

    nanogpt_artifacts/              # All non-code files
        txt_local/                  # Training corpora
        pt/                         # Model checkpoints
        scripts/                    # One-off utility scripts
```

## Branches

Different model variants and features live on separate branches:

- `combined` - Main branch with sentence/word training modes
- `autocorrelation` - Experimental autocorrelation attention
- `bpe-option` - BPE tokenizer support
- `shell-script-launch` - Shell script wrapper features

Switch between them with `git checkout <branch>`.

## Training

### Quick Start

From the `nanogpt/` directory:

```bash
# Sentence-mode training with defaults
sh/train_sentence.sh

# Override specific parameters
sh/train_sentence.sh --max_iters 50000
sh/train_sentence.sh --n_layer 6 --n_head 4

# Full manual control
python py/train.py \
    --input ~/0-Home-Working-on-M3-Pro/nanogpt_artifacts/txt_local/your_corpus.txt \
    --checkpoints_to ~/0-Home-Working-on-M3-Pro/nanogpt_artifacts/pt \
    --mode sentence \
    ...
```

### Using train.sh Wrapper

The `train.sh` wrapper provides logging and background execution:

```bash
sh/train.sh --input /path/to/corpus.txt --n_layer 10 ...
```

This creates timestamped logs in `terminal_logs/` and runs training in the
background. Ctrl+C stops both the log viewer and training.

### Preset Scripts

The `sh/` directory contains preset scripts for common configurations.
These scripts encode sensible defaults and use absolute paths to
`nanogpt_artifacts/`.

**On script naming:** Script names will evolve based on what variations
prove useful in practice. Today's `train_sentence.sh` might become
`train_gibbon.sh` or `train_10layer.sh` tomorrow. The scripts are cheap
to add, rename, or delete as your workflow develops. Let the organization
emerge from actual usage patterns rather than trying to predict them
upfront.

Current presets:

- `train_sentence.sh` - Large corpus, sentence mode, 10-layer model

## Sampling

Generate text from a trained model:

```bash
# Single sentences (default: stops at newline)
python py/sample.py --model ~/0-Home-Working-on-M3-Pro/nanogpt_artifacts/pt/model.pt

# Multiple sentences (continue past newlines)
python py/sample.py --model /path/to/model.pt --no_stop_on_newline

# With prompt
python py/sample.py --model /path/to/model.pt --prompt "The Roman Empire"

# Adjust generation parameters
python py/sample.py --model /path/to/model.pt \
    --temperature 0.9 \
    --top_k 50 \
    --rep_penalty 1.15 \
    --num_samples 10
```

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 0.8 | Higher = more random |
| `--top_k` | 40 | Limit to top-k tokens |
| `--rep_penalty` | 0.0 | Penalize repetition (try 1.15) |
| `--max_tokens` | 130 | Maximum tokens per sample |
| `--num_samples` | 5 | Number of samples to generate |
| `--no_stop_on_newline` | off | Continue past sentence boundaries |

## Training Parameters Reference

Key parameters for `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | required | Path to training corpus |
| `--checkpoints_to` | `pt` | Checkpoint output directory |
| `--mode` | `sentence` | `sentence` or `word` |
| `--tokenizer` | `bpe` | `char` or `bpe` |
| `--n_layer` | 1 | Number of transformer layers |
| `--n_head` | 4 | Number of attention heads |
| `--n_embd` | 256 | Embedding dimension |
| `--block_size` | 128 | Context window size |
| `--batch_size` | 64 | Batch size |
| `--max_iters` | 30000 | Training iterations |
| `--sample_max_tokens` | 128 | Max tokens in training samples |
| `--min_sentence_tokens` | 5 | Skip sentences shorter than this |
| `--untie_weights` | off | Don't tie embeddings to output |
| `--linear_attention` | off | Use linear attention |
| `--autocorrelation_attention` | off | Use autocorrelation attention |
