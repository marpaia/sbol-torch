"""Command-line interface: ``sboltorch <command> <config.yaml>``."""

from __future__ import annotations

import argparse
import sys

from sboltorch.config import RunConfig
from sboltorch.data.corpus import build_corpus
from sboltorch.data.materialize import materialize
from sboltorch.pipeline import run_training


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = RunConfig.from_yaml(args.config)
    corpus = build_corpus(config.corpus)
    result = materialize(corpus, config.corpus.cache_dir, force=args.force)
    print(f"materialized {result.count} objects -> {result.path}")
    print(f"fingerprint: {result.fingerprint}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    config = RunConfig.from_yaml(args.config)
    metrics = run_training(config, resume_from=args.resume)
    print("final metrics:", metrics)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sboltorch", description="Train transformer models on SBOL data")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Materialize a corpus to the local Parquet cache")
    p_ingest.add_argument("config", help="Path to a run config YAML")
    p_ingest.add_argument("--force", action="store_true", help="Re-materialize even if cached")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_train = sub.add_parser("train", help="Run training from a config")
    p_train.add_argument("config", help="Path to a run config YAML")
    p_train.add_argument(
        "--resume", metavar="CKPT", default=None, help="Resume from a checkpoint (e.g. an output dir's last.pt)"
    )
    p_train.set_defaults(func=_cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
