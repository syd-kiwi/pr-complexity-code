#!/usr/bin/env python3
"""Run preparation, RQ1 analysis, and RQ2/RQ3 classification from one command."""
import argparse
import subprocess
import sys
from pathlib import Path

from common import dump_json, fresh_output, setup_logging, LOG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="New directory for the whole experiment")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scope", choices=["open-source", "private", "all"], default="open-source")
    parser.add_argument("--splits", nargs="+", choices=["repository", "merge-time"], default=["repository"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--trees", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--permutation-rows", type=int, default=10000)
    parser.add_argument("--thresholds", nargs=2, type=float)
    parser.add_argument("--exclude-zero", action="store_true")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    args = parser.parse_args()
    if args.trials < 1 or args.trials > 36 or args.trees < 1 or args.threads < 1 or args.permutation_rows < 0:
        parser.error("trials must be 1..36, trees/threads positive, permutation-rows nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.splits)) != len(args.splits):
        parser.error("seeds and splits must not contain duplicates")
    setup_logging()
    root = fresh_output(args.output).resolve()
    dump_json(root / "pipeline_config.json", vars(args))
    here = Path(__file__).resolve().parent

    def run(script, *options):
        command = [sys.executable, str(here / script), *map(str, options)]
        LOG.info("Running %s", script)
        subprocess.run(command, check=True)

    run("prepare.py", "--input", str(Path(args.input).resolve()), "--output", root / "prepared",
        "--scope", args.scope, *(["--exclude-zero"] if args.exclude_zero else []))
    prepared = root / "prepared" / "prepared.parquet"
    run("analyze.py", "--input", prepared, "--output", root / "rq1", "--seed", args.seeds[0])
    for strategy in args.splits:
        for seed in args.seeds:
            run("train.py", "--input", prepared, "--output", root / f"{strategy}_seed{seed}",
                "--device", args.device, "--split", strategy, "--seed", seed,
                "--trials", args.trials, "--trees", args.trees, "--threads", args.threads,
                "--class-weight", args.class_weight, "--permutation-rows", args.permutation_rows,
                *(["--thresholds", *args.thresholds] if args.thresholds else []))
    LOG.info("All experiments completed: %s", root)


if __name__ == "__main__":
    main()
