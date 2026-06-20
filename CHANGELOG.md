# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow semantic versioning.

## [0.1.1] - 2026-06-20

### Added

- **Causal-language-model pretraining and generation.** A `causal` objective
  (`task.kind: causal`) trains a decoder (`gpt2`, `gpt_neox`, `llama`, …) on
  next-token prediction. `sboltorch generate` and `st.generate`/
  `st.generate_sequence` do autoregressive sampling (temperature / top-k / top-p)
  and design completion from a prefix. Tokenizers gained `decode`.
- **Streaming, sharded data.** Corpora materialize to sharded Parquet and can be
  streamed (`streaming: true`) so training no longer needs the corpus in RAM, with
  a stable `hash` split, multi-worker shard assignment, and optional token
  `packing` into fixed-length blocks for LM pretraining.
- **Long context & modern attention.** `model.arch` gained `attn_implementation`
  (SDPA by default — FlashAttention on CUDA) and `rope_theta`; RoPE architectures
  run past an absolute model's context limit.
- **Distributed training (DDP).** `train.distributed.strategy: ddp` replicates the
  model and all-reduces gradients across ranks (launch with `torchrun`), with
  rank-aware data sharding, rank-0-only checkpoints/logs, and cross-rank metric
  reduction. Data-parallel only; no parameter sharding.
- **Hardened training loop.** Resumable checkpoints (`sboltorch train --resume`)
  carrying optimizer/scheduler/scaler/RNG state; step-budgeted training
  (`max_steps`, `eval_every_n_steps`, `checkpoint_every_n_steps`); `bf16`/`fp16`
  precision; gradient checkpointing; `torch.compile`.
- **sbol CLI normalization.** `scripts/normalize_sbol.sh` converts raw GenBank/
  SBOL2 inputs to SBOL3 via the `sbol` CLI for the `local` corpus source, with
  Component-centric SBOL3 parsing that unlocks the structure-aware and graph
  modalities on real data.
- Continuous integration via GitHub Actions.
- New example config `pretrain_causal_long.yaml` (RoPE decoder, SDPA, streamed +
  packed corpus).

### Changed

- `model.arch.max_position_embeddings` default raised from 1024 to 2048.
- Corpus materialization writes sharded Parquet (`corpus.shard_size`) instead of a
  single file; existing single-file caches are still read.
- Package now uses absolute imports throughout.

## [0.1.0]

- Initial release: SBOL/FASTA and sbol-db data sources, `sequence` /
  `structure_aware` / `graph` modalities, `supervised` / `frozen` / `mlm`
  objectives, `hf` / `kmer` / `char` tokenizers, a raw-PyTorch training engine with
  early stopping and AMP, reproducible Parquet caching, and Weights & Biases
  tracking.
