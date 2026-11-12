#!/usr/bin/env python3
"""Prepare text once and launch the enhanced experiments on the supplied CSV."""
import argparse
import subprocess
import sys
from pathlib import Path

from common import dump_json, fresh_output, setup_logging, LOG


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Additional options are forwarded to experiment.py; see experiment.py --help.")
    parser.add_argument("--input", default="data/cleaned_public.csv")
    parser.add_argument("--prepared-input", help="Reuse an existing --include-text prepared Parquet; skips CSV loading")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=["open-source", "private", "all"], default="open-source")
    parser.add_argument("--exclude-zero", action="store_true")
    args, extra = parser.parse_known_args()
    if any(x == "--input" or x.startswith("--input=") or x == "--output" or x.startswith("--output=") for x in extra):
        parser.error("input and output must be supplied to the wrapper directly")
    setup_logging()
    root = fresh_output(args.output).resolve()
    dump_json(root / "research_config.json", {**vars(args), "experiment_options": extra,
                                              "prepared_input_note": "When reusing prepared data, scope/zero filtering comes from its preparation audit."})
    here = Path(__file__).resolve().parent
    if args.prepared_input:
        prepared = Path(args.prepared_input).resolve()
    else:
        command = [sys.executable, str(here / "prepare.py"), "--input", str(Path(args.input).resolve()),
                   "--output", str(root / "prepared"), "--scope", args.scope, "--include-text"]
        if args.exclude_zero:
            command.append("--exclude-zero")
        subprocess.run(command, check=True)
        prepared = root / "prepared" / "prepared.parquet"
    subprocess.run([sys.executable, str(here / "experiment.py"), "--input", str(prepared),
                    "--output", str(root / "models"), *extra], check=True)
    LOG.info("Research run completed: %s", root)


if __name__ == "__main__":
    main()
