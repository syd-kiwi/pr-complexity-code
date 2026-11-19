#!/usr/bin/env python3
"""Extract public, code-changing merged PRs from a cleaned CSV or CSV.GZ."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

KEYS = ("repository_id", "pull_request_id")
REQUIRED = (*KEYS, "is_open_source", "is_code_pr", "minutes_to_merge")
SPLIT_COLUMNS = ("original_split", "original_duration_class")
SPLITS = ("train", "validation", "test")
CLASSES = {"0": "fast", "1": "medium", "2": "slow"}
TRUE = {"true", "1", "1.0", "yes"}
FALSE = {"false", "0", "0.0", "no", "", "nan", "na", "n/a", "null", "none", "<na>"}


def open_csv(path):
    opener = gzip.open if path.name.endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8-sig", newline="")


def require_columns(reader, required, path):
    fields = reader.fieldnames or []
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}")
    if len(fields) != len(set(fields)):
        raise ValueError(f"{path} has duplicate column names.")
    return fields


def flag(value, column):
    token = value.strip().lower()
    if token in TRUE:
        return True
    if token in FALSE:
        return False
    raise ValueError(f"Unrecognized value for {column}: {value!r}")


def duration_is_valid(value):
    try:
        duration = float(value)
    except ValueError:
        return False
    return math.isfinite(duration) and duration >= 0


def read_reference(path):
    reference = {}
    repository_splits = {}
    with open_csv(path) as source:
        rows = csv.DictReader(source)
        require_columns(rows, (*KEYS, "split", "class"), path)
        for row in rows:
            key = tuple((row.get(field) or "").strip() for field in KEYS)
            split = row["split"]
            label = row["class"]
            if not all(key) or key in reference or split not in SPLITS or label not in CLASSES:
                raise ValueError(f"Invalid or duplicate reference row at line {rows.line_num}.")
            if repository_splits.setdefault(key[0], split) != split:
                raise ValueError(f"A repository appears across partitions at reference line {rows.line_num}.")
            reference[key] = split, label
    if not reference:
        raise ValueError("The reference split file is empty.")
    return reference


def write_subset(input_path, output_path, reference):
    totals = Counter()
    by_partition = Counter()
    reported = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Choose a new output path: {output_path}")

    # Write beside the destination and publish only a completed, compressed file.
    with TemporaryDirectory(dir=output_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / "subset.csv.gz"
        with open_csv(input_path) as source, gzip.open(
            temp_path, "wt", encoding="utf-8", newline="", compresslevel=3
        ) as destination:
            rows = csv.DictReader(source)
            original_fields = require_columns(rows, REQUIRED, input_path)
            if reference is not None and set(SPLIT_COLUMNS).intersection(original_fields):
                raise ValueError("The source CSV already has original split columns.")
            fields = original_fields + (list(SPLIT_COLUMNS) if reference is not None else [])
            writer = csv.DictWriter(destination, fields)
            writer.writeheader()
            for row in rows:
                totals["input"] += 1
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV row at line {rows.line_num}.")
                if not (flag(row["is_open_source"], "is_open_source")
                        and flag(row["is_code_pr"], "is_code_pr")):
                    continue
                key = tuple(row[field].strip() for field in KEYS)
                if not all(key) or not duration_is_valid(row["minutes_to_merge"]):
                    totals["invalid_key_or_duration"] += 1
                    continue
                if key in reported:
                    raise ValueError(f"Duplicate selected PR key at CSV line {rows.line_num}. Clean the input before subsetting.")
                reported.add(key)
                if reference is not None:
                    if key not in reference:
                        raise ValueError(f"Selected PR at CSV line {rows.line_num} is absent from the reference split.")
                    split, label = reference[key]
                    row.update(zip(SPLIT_COLUMNS, (split, label)))
                    by_partition[split, label] += 1
                writer.writerow(row)
                totals["retained"] += 1
        if not totals["retained"]:
            raise ValueError("No eligible code-changing PRs were found.")
        os.link(temp_path, output_path)

    return totals, by_partition, len(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/cleaned_public.csv"))
    parser.add_argument("--output", type=Path, help="New path ending in .csv.gz")
    parser.add_argument("--reference-splits", type=Path, help="Optional original splits.csv.gz")
    args = parser.parse_args()
    output_path = args.output or Path("data") / f"code_changing_prs_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%fZ}.csv.gz"
    if not output_path.name.endswith(".csv.gz"):
        parser.error("--output must end in .csv.gz")
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")

    csv.field_size_limit(2**31 - 1)
    try:
        reference = read_reference(args.reference_splits) if args.reference_splits else None
        totals, by_partition, columns = write_subset(args.input, output_path, reference)
    except (OSError, EOFError, ValueError, csv.Error) as error:
        parser.exit(1, f"Subset failed: {error}\n")

    print(f"Input: {args.input}")
    print(f"Rows read: {totals['input']:,}")
    print(f"Code-changing public PRs retained: {totals['retained']:,}")
    print(f"Rows excluded: {totals['input'] - totals['retained']:,}")
    print(f"Selected rows excluded for invalid keys or duration: {totals['invalid_key_or_duration']:,}")
    print(f"Output columns: {columns:,}")
    if reference is not None:
        print("Original partition class counts (fast, medium, slow):")
        for split in SPLITS:
            counts = ", ".join(f"{by_partition[split, label]:,}" for label in CLASSES)
            print(f"  {split}: {counts}")
    print(f"Compressed subset: {output_path.resolve()}")


if __name__ == "__main__":
    main()
