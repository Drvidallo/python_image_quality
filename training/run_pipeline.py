"""End-to-end orchestration for the model-focused preprocessing pipeline.

Usage:
    python run_pipeline.py --config config.yaml --stage all
    python run_pipeline.py --stage metrics --every-n 20 --limit-folders 3
    python run_pipeline.py --stage build --max-variants 4
    python run_pipeline.py --stage plots
"""

import argparse
import os

import build_dataset
import eda_plots
import make_plot_docs
import preprocess_midv500
import train_models
from data_common import load_config


def main():
    parser = argparse.ArgumentParser(description="Run the EDA preprocessing pipeline stages.")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--stage", choices=["all", "metrics", "build", "plots", "train", "docs"], default="all")
    parser.add_argument("--every-n", type=int, default=None)
    parser.add_argument("--limit-folders", type=int, default=None)
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = config["paths"]["processed_dir"]
    plots_dir = config["paths"]["plots_dir"]

    if args.stage in ("all", "metrics"):
        print("=== Stage 1: ingest raw metrics ===")
        preprocess_midv500.run(
            config, args.every_n, args.limit_folders, args.workers, args.force
        )

    if args.stage in ("all", "build"):
        print("=== Stage 2: build dataset (clean/transform/reduce/engineer/split) ===")
        build_dataset.run(config, args.max_variants, args.workers, args.force)

    if args.stage in ("all", "plots"):
        print("=== Stage 3: EDA plots ===")
        eda_plots.run(processed_dir, plots_dir, config)

    if args.stage in ("all", "train"):
        print("=== Stage 4: train learned models ===")
        train_models.run(config, args.force)

    if args.stage in ("all", "plots", "train", "docs"):
        print("=== Stage 5: plot documentation ===")
        make_plot_docs.run(config)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()