#!/usr/bin/env python3
"""GPU conversion of 4.calculate_pr_complexity_score.ipynb. Python 3.10+.

INSTALL (CUDA 12 Linux):
  python -m pip install 'cudf-cu12==25.10.*' pandas numpy matplotlib scipy tqdm
RUN:
  python -u 4.calculate_pr_complexity_score_gpu.py --input /workspace/data/df_title_body_with_categories.csv --output /workspace/data/prs_with_complexity.csv

GPU: float64 log1p and global average-tie percentile ranks using CuPy/cuDF.
CPU: CSV I/O, score combination/rounding, quartile labels, tables and plots.
All rows are retained in their original order; no sampling or deduplication.
Only narrow numeric columns are loaded globally; original wide text is streamed
through a second pass. Rank percentiles use the ENTIRE input, never chunks.
Do not modify the input while this two-pass script runs. Metadata change checks
are enforced before and after reading; this is not a content checksum.
Missing/non-numeric numeric features become zero; negatives are clipped to zero,
matching the notebook. Missing category counts still get the rank of zero (as in
its executed formula), not a literal zero contribution. Missing review columns
emit warnings, since they change the meaning of the post-review score.
Weights: code .30, files .20, review .25 (.60 comments/.40 reviewers),
configuration .10, keyword breadth .15. Score rounded to 2 decimals.
Quartiles use pandas.qcut. Degenerate ties produce fewer named groups instead
of the notebook's labels-length error. An all-tied score gets 'Tied'.
Output score is pr_complexity_score; existing complexity_score is preserved.
Full-data percentiles are descriptive; predictive studies must fit their rank
reference distributions using training data only.
--backend cpu provides the notebook-compatible reference implementation.
"""
from __future__ import annotations
import argparse
import ast
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

LOG = logging.getLogger("pr_complexity")
NUMERIC = ["code_lines_modified", "lines_modified", "files_touched", "files_added", "files_deleted", "files_modified",
           "num_review_comments", "num_reviewers", "config_lines_modified", "keyword_category_count",
           "minutes_to_merge", "complexity_score"]
SCORE_COLUMNS = ["complexity_code_size", "complexity_file_breadth", "complexity_review_comments", "complexity_reviewers",
                 "complexity_config", "complexity_category_breadth", "complexity_review_burden", "pr_complexity_score", "pr_complexity_level"]


def numeric_column(df, name, fallback=None):
    if name in df:
        values = pd.to_numeric(df[name], errors="coerce").fillna(0)
    elif fallback is not None and fallback in df:
        values = pd.to_numeric(df[fallback], errors="coerce").fillna(0)
    else:
        values = pd.Series(0., index=df.index)
    return values.clip(lower=0).astype("float64")


def calculate_scores(df, backend):
    if len(df) == 0:
        raise ValueError("Input CSV has no PR rows")
    if backend == "gpu":
        try:
            import cudf
            import cupy as cp
            cp.cuda.Device(0).use()
            probe = cudf.Series(cp.log1p(cp.asarray([0., 0., 1., 2.]))).rank(method="average", pct=True).to_pandas().to_numpy()
            if not np.allclose(probe, [.375, .375, .75, 1.]):
                raise RuntimeError("GPU rank startup parity check failed")
            LOG.info("GPU ready: cuDF %s, CuPy %s", cudf.__version__, cp.__version__)
        except Exception as exc:
            raise RuntimeError("GPU initialization failed; install cudf-cu12 in the CUDA 12 template, "
                               "or explicitly use --backend cpu. Original error: " + str(exc)) from exc
        def percentile(values):
            if np.isinf(values.to_numpy()).any():
                # Preserve exact pandas handling of infinities instead of relying
                # on device-specific infinity ordering.
                LOG.warning("Infinity in numeric feature: ranking this feature on CPU")
                return np.log1p(values).rank(method="average", pct=True).to_numpy()
            gpu = cp.log1p(cp.asarray(values.to_numpy(dtype=np.float64)))
            result = cudf.Series(gpu).rank(method="average", pct=True).to_pandas().to_numpy(dtype=np.float64)
            return result
    else:
        def percentile(values):
            return np.log1p(values).rank(method="average", pct=True).to_numpy()
    files = numeric_column(df, "files_touched") if "files_touched" in df else sum((numeric_column(df, c) for c in ["files_added", "files_deleted", "files_modified"]))
    features = {
        "complexity_code_size": numeric_column(df, "code_lines_modified", "lines_modified"),
        "complexity_file_breadth": files,
        "complexity_review_comments": numeric_column(df, "num_review_comments"),
        "complexity_reviewers": numeric_column(df, "num_reviewers"),
        "complexity_config": numeric_column(df, "config_lines_modified"),
        "complexity_category_breadth": numeric_column(df, "keyword_category_count"),
    }
    result = pd.DataFrame(index=df.index)
    for name, values in features.items():
        LOG.info("Global percentile ranking: %s (%s PRs)", name, f"{len(df):,}")
        result[name] = percentile(values)
    result["complexity_review_burden"] = .60*result.complexity_review_comments + .40*result.complexity_reviewers
    result["pr_complexity_score"] = (100*(.30*result.complexity_code_size + .20*result.complexity_file_breadth + .25*result.complexity_review_burden + .10*result.complexity_config + .15*result.complexity_category_breadth)).round(2)
    score = result.pr_complexity_score
    codes = pd.qcut(score, [0, .25, .5, .75, 1.], labels=False, duplicates="drop")
    count = int(codes.nunique())
    labels = {4: ["Low", "Moderate", "High", "Very High"], 3: ["Low", "Moderate", "High"], 2: ["Low", "High"], 1: ["Tied"]}
    if count == 0:
        result["pr_complexity_level"] = "Tied"
    else:
        result["pr_complexity_level"] = codes.map(dict(enumerate(labels[count])))
    if count < 4:
        LOG.warning("Score ties produced %s quartile bins; labels adjusted and documented", count)
    return result


def category_count(series):
    def parse(value):
        if not isinstance(value, str) or not value.strip():
            return 0
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError("Malformed keyword_categories; rerun script 3 or provide keyword_category_count")
        if not isinstance(parsed, list):
            raise ValueError("keyword_categories must contain lists")
        return len(parsed)
    lookup = {value: parse(value) for value in series.dropna().astype(str).unique()}
    return series.map(lookup).fillna(0).astype(int)


def reports(numeric, scores, path, side):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.mkdir(parents=True, exist_ok=True)
    scores.pr_complexity_score.describe().to_csv(path / "score_summary.csv", header=["value"])
    scores.pr_complexity_level.value_counts().to_csv(path / "level_distribution.csv", header=["PR_count"])
    scores[["pr_complexity_score"]].nlargest(20, "pr_complexity_score").assign(source_row=lambda d: d.index).to_csv(path / "top_20_complexity_rows.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores.pr_complexity_score, bins=30, color="green", edgecolor="black", alpha=.8)
    ax.set(title="Distribution of PR complexity scores", xlabel="PR complexity score", ylabel="PRs")
    fig.tight_layout(); fig.savefig(path / "score_distribution.png", dpi=160); plt.close(fig)
    if "minutes_to_merge" in numeric:
        validation = scores[["pr_complexity_score", "pr_complexity_level"]].copy()
        validation["minutes_to_merge"] = pd.to_numeric(numeric.minutes_to_merge, errors="coerce")
        validation = validation.loc[validation.minutes_to_merge.notna() & validation.minutes_to_merge.ge(0)].copy()
        validation["hours_to_merge"] = validation.minutes_to_merge / 60
        validation["log_hours_to_merge"] = np.log1p(validation.hours_to_merge)
        valid = validation.replace([np.inf, -np.inf], np.nan).dropna(subset=["pr_complexity_score", "hours_to_merge", "log_hours_to_merge"])
        correlations = pd.DataFrame({"metric": ["pearson_raw", "pearson_log", "spearman"], "value": [
            valid.pr_complexity_score.corr(valid.hours_to_merge), valid.pr_complexity_score.corr(valid.log_hours_to_merge),
            valid.pr_complexity_score.corr(valid.hours_to_merge, method="spearman")]})
        correlations.to_csv(path / "correlations.csv", index=False)
        LOG.info("Correlations:\n%s", correlations.to_string(index=False))
        validation["complexity_decile"] = pd.qcut(validation.pr_complexity_score, 10, labels=False, duplicates="drop") + 1
        summary = validation.groupby("complexity_decile").agg(median_complexity=("pr_complexity_score", "median"), median_hours_to_merge=("hours_to_merge", "median"), mean_hours_to_merge=("hours_to_merge", "mean"), pr_count=("hours_to_merge", "size")).reset_index()
        summary.to_csv(path / "decile_summary.csv", index=False)
        by_level = validation.groupby("pr_complexity_level", observed=True).agg(median_complexity=("pr_complexity_score", "median"), median_hours_to_merge=("hours_to_merge", "median"), mean_hours_to_merge=("hours_to_merge", "mean"), pr_count=("hours_to_merge", "size")).reset_index()
        order = {name: j for j, name in enumerate(["Low", "Moderate", "High", "Very High", "Tied"])}
        by_level = by_level.sort_values("pr_complexity_level", key=lambda s: s.map(order))
        by_level.to_csv(path / "level_merge_summary.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(summary.complexity_decile, summary.median_hours_to_merge, marker="o", color="green")
        ax.set(title="Median merge time by complexity decile", xlabel="Complexity decile", ylabel="Median hours to merge")
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path / "merge_time_by_decile.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(by_level.pr_complexity_level, by_level.median_hours_to_merge, color="green")
        ax.set(title="Median merge time by complexity level", ylabel="Median hours to merge")
        fig.tight_layout(); fig.savefig(path / "merge_time_by_level.png", dpi=160); plt.close(fig)
    for col, name in [("keyword_categories", "category"), ("languages_list", "language_combination")]:
        if col in side:
            table = pd.DataFrame({col: side[col], "score": scores.pr_complexity_score})
            summary = table.groupby(col, observed=True).score.agg(["median", "mean", "size"]).rename(columns={"size": "pr_count"}).sort_values("median", ascending=False)
            if col == "languages_list":
                summary = summary.query("pr_count >= 100")
            summary.to_csv(path / f"complexity_by_{name}.csv")
    if "merge_day" in side:
        monthly = pd.DataFrame({"month": pd.to_datetime(side.merge_day, errors="coerce").dt.to_period("M").astype("string"), "score": scores.pr_complexity_score}).groupby("month").score.median().reset_index(name="median_complexity")
        monthly.to_csv(path / "complexity_by_month.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(monthly.month.astype(str), monthly.median_complexity, marker="o", color="green")
        ax.set(title="Median PR complexity by month", xlabel="Month", ylabel="Median complexity")
        fig.tight_layout(); fig.savefig(path / "complexity_by_month.png", dpi=160); plt.close(fig)


def case_indices(numeric, scores):
    if "complexity_score" not in numeric:
        return []
    old = pd.to_numeric(numeric.complexity_score, errors="coerce")
    new = scores.pr_complexity_score
    old_pct, new_pct = old.rank(pct=True)*100, new.rank(pct=True)*100
    delta = (new_pct-old_pct).dropna()
    result = [(int(i), "New score much higher") for i in delta.nlargest(3).index]
    result += [(int(i), "Old score much higher") for i in delta.nsmallest(3).index]
    for name, mask in [("Both scores high", (old_pct>=90)&(new_pct>=90)), ("Both scores low", (old_pct<=10)&(new_pct<=10))]:
        candidates = old_pct.loc[mask].dropna()
        result += [(int(i), name) for i in candidates.sample(n=min(3, len(candidates)), random_state=42).index]
    return result


def signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="df_title_body_with_categories.csv")
    p.add_argument("--output", default="prs_with_complexity.csv")
    p.add_argument("--report-dir", default=None)
    p.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--chunksize", type=int, default=25000)
    p.add_argument("--skip-reports", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if source == output or output.exists():
        p.error("Choose a new output path; input and existing outputs are protected.")
    if args.chunksize <= 0:
        p.error("--chunksize must be positive")
    before = signature(source)
    columns = pd.read_csv(source, nrows=0).columns.tolist()
    if not any(c in columns for c in ["code_lines_modified", "lines_modified"]):
        p.error("Input needs code_lines_modified or lines_modified")
    if "keyword_category_count" not in columns and "keyword_categories" not in columns:
        p.error("Run script 3 first: keyword category features are missing")
    for col in ["num_review_comments", "num_reviewers", "config_lines_modified"]:
        if col not in columns:
            LOG.warning("Missing %s: using zeros as the original notebook does", col)
    extra = [c for c in ["keyword_categories", "languages_list", "merge_day"] if c in columns and (not args.skip_reports or c == "keyword_categories")]
    usecols = [c for c in NUMERIC if c in columns]+extra
    LOG.info("Reading narrow feature columns from %s", source)
    start = time.monotonic()
    frame = pd.read_csv(source, usecols=usecols, low_memory=False)
    if "keyword_category_count" not in frame:
        frame["keyword_category_count"] = category_count(frame.keyword_categories)
        LOG.info("Derived missing category counts from category lists")
    side = frame[extra].copy()
    numeric = frame.drop(columns=extra)
    del frame
    LOG.info("Loaded all %s PRs; computing global percentiles", f"{len(numeric):,}")
    scores = calculate_scores(numeric, args.backend)
    if signature(source) != before:
        raise RuntimeError("Input changed during feature reading. Use a completed immutable input.")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.csv")
    if partial.exists():
        p.error(f"Staging file already exists: {partial}. Rename it before rerunning.")
    case_selection = case_indices(numeric, scores)
    selected_indices = {i for i, _ in case_selection}
    case_rows = {}
    dtypes = {c: "string" for c in ["repository_id", "pull_request_id", "user_id", "customer_id"] if c in columns}
    offset = 0
    with tqdm(total=len(scores), desc="Saving scored PRs", unit="PR") as progress:
        for j, chunk in enumerate(pd.read_csv(source, chunksize=args.chunksize, dtype=dtypes, low_memory=False)):
            stop = offset+len(chunk)
            if stop > len(scores):
                raise RuntimeError("Row count changed between passes")
            if "keyword_category_count" not in columns:
                chunk["keyword_category_count"] = numeric.iloc[offset:stop]["keyword_category_count"].to_numpy()
            for col in SCORE_COLUMNS:
                chunk[col] = scores.iloc[offset:stop][col].to_numpy()
            chunk.to_csv(partial, mode="w" if j == 0 else "a", header=j == 0, index=False)
            if not args.skip_reports:
                for i in selected_indices.intersection(range(offset, stop)):
                    case_rows[i] = chunk.iloc[i-offset].to_dict()
            offset = stop
            progress.update(len(chunk))
            LOG.info("Saved %s / %s PRs", f"{offset:,}", f"{len(scores):,}")
    if offset != len(scores) or signature(source) != before:
        raise RuntimeError("Input changed between passes. Output remains .partial.csv.")
    partial.replace(output)
    report_dir = Path(args.report_dir).resolve() if args.report_dir else output.parent / "complexity_reports"
    if not args.skip_reports:
        LOG.info("Saving summary tables and plots (CPU)")
        reports(numeric, scores, report_dir, side)
        cases = []
        if case_selection:
            old_pct = pd.to_numeric(numeric.complexity_score, errors="coerce").rank(pct=True)*100
            new_pct = scores.pr_complexity_score.rank(pct=True)*100
            for i, label in case_selection:
                cases.append({"case_type": label, **case_rows[i], "old_complexity_percentile": old_pct.iloc[i],
                              "new_complexity_percentile": new_pct.iloc[i], "complexity_percentile_change": new_pct.iloc[i]-old_pct.iloc[i]})
            pd.DataFrame(cases).to_csv(report_dir / "old_vs_new_case_studies.csv", index=False)
    audit = {"input": str(source), "output": str(output), "rows_processed": len(scores), "backend": args.backend,
             "weights": {"code": .30, "files": .20, "review": .25, "configuration": .10, "keyword_breadth": .15},
             "review_subweights": {"comments": .60, "reviewers": .40}, "tie_method": "average", "rank_reference": "entire_input",
             "missing_numeric_rule": "fill zero, clip negative to zero", "rounding_decimals": 2,
             "no_rows_sampled_or_skipped": True, "level_counts": {str(k): int(v) for k,v in scores.pr_complexity_level.value_counts().items()}, "seconds": time.monotonic()-start,
             "note": "Post-review descriptive score; compute training-only ranks for prospective ML comparisons."}
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    LOG.info("DONE: saved all %s scored PRs to %s", f"{len(scores):,}", output)


if __name__ == "__main__":
    main()
