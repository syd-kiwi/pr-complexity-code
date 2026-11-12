#!/usr/bin/env python3
"""Validate the supplied schema and save a compact, auditable modeling table."""
import argparse
from common import clean_data, dump_json, fingerprint, fresh_output, read_source, setup_logging, versions, LOG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/cleaned_public.csv", help="CSV, CSV.GZ, or Parquet file (default: data/cleaned_public.csv, relative to the working directory)")
    parser.add_argument("--output", required=True, help="New directory")
    parser.add_argument("--scope", choices=["open-source", "private", "all"], default="open-source")
    parser.add_argument("--exclude-zero", action="store_true")
    parser.add_argument("--include-text", action="store_true", help="Retain raw title/body for the enhanced experiments")
    args = parser.parse_args()
    setup_logging()
    LOG.info("Reading selected columns from %s", args.input)
    df, audit = clean_data(read_source(args.input, args.include_text), args.scope, args.exclude_zero)
    out = fresh_output(args.output)
    LOG.info("Fingerprinting input and writing %s valid PRs", len(df))
    audit.update(source=fingerprint(args.input), versions=versions())
    df.to_parquet(out / "prepared.parquet", index=False)
    dump_json(out / "audit.json", audit)
    LOG.info("Ready: %s", out / "prepared.parquet")


if __name__ == "__main__":
    main()
