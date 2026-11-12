#!/usr/bin/env python3
"""GPU conversion of 3.categorize_prs_keywords.ipynb. Python 3.10+.

INSTALL (CUDA 12 Linux):
  python -m pip install 'cudf-cu12==25.10.*' pandas numpy matplotlib scipy tqdm
RUN:
  python -u 3.categorize_prs_keywords_gpu.py --input /workspace/data/cleaned_public.csv --output /workspace/data/df_title_body_with_categories.csv

GPU: cuDF regex matching in bounded chunks. CPU: CSV I/O, exact handling of
non-ASCII text (Python regex Unicode semantics), list serialization and reports.
--backend cpu is a fast, vectorized fallback with the original Python regex.
No rows sampled, skipped, reordered, or deduplicated. CSV identifiers are read
as strings. Parse errors fail visibly; incomplete CSVs are never accepted.
Outputs use a .partial.csv staging file; the final CSV appears only on success.
All original keywords and matching order are preserved. Category lists are JSON
strings in CSV, compatible with json.loads and ast.literal_eval.
Summary tables and PNGs replace notebook display()/plt.show(). No model training.
--skip-reports skips only tables/figures, never categorization or data rows.
"""
from __future__ import annotations
import argparse
import ast
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

LOG = logging.getLogger("categorize_prs")
PR_CATEGORIES = {
    "feature": ["add", "implement", "introduce", "support", "enable", "new feature", "create"],
    "bug_fix": ["fix", "bug", "resolve", "correct", "patch", "hotfix", "regression"],
    "refactor": ["refactor", "cleanup", "clean up", "simplify", "restructure", "reorganize"],
    "documentation": ["docs", "documentation", "readme", "comment", "guide", "tutorial"],
    "testing": ["test", "tests", "testing", "coverage", "unit test", "integration test"],
    "dependency": ["dependency", "dependencies", "bump", "upgrade", "downgrade", "package", "version"],
    "security": ["security", "vulnerability", "cve", "authentication", "authorization", "permission", "sanitize"],
    "performance": ["performance", "optimize", "optimization", "speed", "latency", "memory", "cache"],
    "configuration": ["config", "configuration", "settings", "environment variable", "yaml"],
    "release": ["release", "changelog", "publish", "tag", "prepare release"],
    "revert": ["revert", "rollback", "back out", "undo"],
}
TEXT_COLUMNS = ["title", "body", "pr_title", "pr_body", "clean_title", "clean_body"]


def patterns(keywords):
    alternatives = "|".join(re.escape(k.lower()).replace(r"\ ", r"\s+") for k in keywords)
    # Consuming boundaries are equivalent for boolean contains; no lookbehind is
    # needed on the GPU. CPU keeps the notebook's exact Python regex semantics.
    return (re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)", re.IGNORECASE),
            r"(?:^|\W)(?:" + alternatives + r")(?:\W|$)")


def gpu_backend(backend):
    if backend == "cpu":
        return None
    try:
        import cudf
        examples = ["fix a bug", "prefix", "bugfix", "fix_fix", "new\tfeature", "clean\nup",
                    "unit\vtest", "new\ffeature", "environment\rvariable", "", "Fix", "fix-fix",
                    "enable release rollback config security cache docs bump integration test"]
        normalized = [t.lower() for t in examples]
        probe = cudf.Series(normalized)
        for keywords in PR_CATEGORIES.values():
            cpu_pattern, gpu_pattern = patterns(keywords)
            expected = [bool(cpu_pattern.search(t)) for t in normalized]
            actual = probe.str.contains(gpu_pattern, regex=True).to_pandas().tolist()
            if actual != expected:
                raise RuntimeError("GPU regex startup parity check failed")
        LOG.info("GPU backend ready: cuDF %s", cudf.__version__)
        return cudf
    except Exception as exc:
        raise RuntimeError("GPU initialization failed. Install cudf-cu12 for the CUDA 12 template; "
                           "or explicitly use --backend cpu. Original error: " + str(exc)) from exc


def combine_text(chunk, columns):
    # Column-wise concatenation avoids pandas row-by-row .agg(axis=1).
    text = chunk[columns[0]].fillna("").astype(str)
    for col in columns[1:]:
        text = text.str.cat(chunk[col].fillna("").astype(str), sep=" ")
    return text.str.lower()


def match_chunk(chunk, columns, cudf):
    text = combine_text(chunk, columns)
    masks = np.zeros((len(chunk), len(PR_CATEGORIES)), dtype=bool)
    if cudf is None:
        for j, keywords in enumerate(PR_CATEGORIES.values()):
            masks[:, j] = text.str.contains(patterns(keywords)[0], na=False).to_numpy()
        return masks, 0
    # GPU regex character classes can differ from Python Unicode classes. Match
    # every non-ASCII row using Python's original regex, not just a sample.
    unicode_rows = text.str.contains(r"[^\x00-\x7f]", regex=True, na=False).to_numpy()
    ascii_rows = ~unicode_rows
    gpu_text = cudf.Series(text.loc[ascii_rows].reset_index(drop=True)) if ascii_rows.any() else None
    for j, keywords in enumerate(PR_CATEGORIES.values()):
        cpu_pattern, gpu_pattern = patterns(keywords)
        if gpu_text is not None:
            masks[ascii_rows, j] = gpu_text.str.contains(gpu_pattern, regex=True).fillna(False).to_pandas().to_numpy(dtype=bool)
        if unicode_rows.any():
            masks[unicode_rows, j] = text.loc[unicode_rows].str.contains(cpu_pattern, na=False).to_numpy()
    return masks, int(unicode_rows.sum())


def mask_lookups():
    categories = list(PR_CATEGORIES)
    lists = {mask: [category for j, category in enumerate(categories) if mask & (1 << j)]
             for mask in range(1 << len(categories))}
    serialized = {mask: json.dumps(categories, ensure_ascii=False) for mask, categories in lists.items()}
    return lists, serialized


def primary_languages(series):
    def parse(value):
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return str(parsed[0]).strip() if len(parsed) else None
        except (ValueError, SyntaxError):
            pass
        matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", value)
        if matches:
            return (matches[0][0] or matches[0][1]).strip()
        tokens = [v.strip().strip("'\"") for v in value.strip("[]").split(",") if v.strip()]
        return tokens[0] if tokens else None
    # Parse only distinct representations instead of every row.
    lookup = {s: parse(s) for s in series.dropna().astype(str).unique()}
    return series.map(lookup)


def save_reports(report_parts, counts, count_distribution, total, report_dir, min_language_prs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report_dir.mkdir(parents=True, exist_ok=True)
    report = pd.concat(report_parts, ignore_index=True)
    dist = pd.DataFrame({"category": list(PR_CATEGORIES), "PR_count": counts})
    dist["percent_of_PRs"] = dist.PR_count / total * 100
    dist.sort_values("PR_count", ascending=False).to_csv(report_dir / "category_distribution.csv", index=False)
    pd.DataFrame({"category_count": np.arange(len(count_distribution)), "PR_count": count_distribution}).to_csv(report_dir / "category_count_distribution.csv", index=False)
    def bars(frame, label, value, filename, title, xlabel):
        if frame.empty:
            return
        frame = frame.sort_values(value)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(frame[label].astype(str), frame[value], color="#0ea31b", edgecolor="black")
        ax.set(title=title, xlabel=xlabel)
        ax.grid(axis="x", alpha=.25)
        fig.tight_layout(); fig.savefig(report_dir / filename, dpi=160); plt.close(fig)
    bars(dist, "category", "PR_count", "category_counts.png", "Number of PRs by category", "PRs")
    for month, group in report.groupby("month", dropna=True):
        month_counts = [int(((group["mask"].to_numpy() & (1 << j)) != 0).sum()) for j in range(len(PR_CATEGORIES))]
        md = pd.DataFrame({"category": list(PR_CATEGORIES), "PR_count": month_counts})
        md.to_csv(report_dir / f"category_counts_{month}.csv", index=False)
        bars(md, "category", "PR_count", f"category_counts_{month}.png", f"PR categories: {month}", "PRs")
    valid = report.loc[report.minutes_to_merge.notna() & report.minutes_to_merge.ge(0)].copy()
    valid["hours_to_merge"] = valid.minutes_to_merge / 60
    pd.DataFrame({"metric": ["PRs with merge time", "Median hours to merge", "Mean hours to merge", "Median days to merge"],
                  "value": [len(valid), valid.hours_to_merge.median(), valid.hours_to_merge.mean(), valid.hours_to_merge.median() / 24]}).to_csv(report_dir / "overall_merge_summary.csv", index=False)
    rows, language_rows = [], []
    languages = ["JavaScript", "Python", "Java", "C++"]
    for j, category in enumerate(PR_CATEGORIES):
        selected = valid.loc[(valid["mask"].to_numpy() & (1 << j)) != 0]
        if selected.empty:
            continue
        values = selected.hours_to_merge
        rows.append({"category": category, "matched_prs": selected.pull_request_id.nunique(),
                     "median_hours_to_merge": values.median(), "mean_hours_to_merge": values.mean(),
                     "p75_hours_to_merge": values.quantile(.75), "median_days_to_merge": values.median()/24})
        for language in languages:
            group = selected.loc[selected.primary_language.eq(language)]
            if group.pull_request_id.nunique() >= min_language_prs:
                values = group.hours_to_merge
                language_rows.append({"primary_programming_language": language, "category": category,
                                      "matched_prs": group.pull_request_id.nunique(), "median_hours_to_merge": values.median(),
                                      "mean_hours_to_merge": values.mean(), "p75_hours_to_merge": values.quantile(.75)})
    summary = pd.DataFrame(rows, columns=["category", "matched_prs", "median_hours_to_merge", "mean_hours_to_merge", "p75_hours_to_merge", "median_days_to_merge"])
    summary.to_csv(report_dir / "category_merge_summary.csv", index=False)
    bars(summary, "category", "median_hours_to_merge", "merge_time_by_category.png", "Median time to merge by PR category", "Median hours to merge")
    lang = pd.DataFrame(language_rows, columns=["primary_programming_language", "category", "matched_prs", "median_hours_to_merge", "mean_hours_to_merge", "p75_hours_to_merge"])
    lang = lang.merge(summary[["category", "median_hours_to_merge"]].rename(columns={"median_hours_to_merge": "overall_median_hours_to_merge"}), on="category", how="left")
    lang.to_csv(report_dir / "language_category_summary.csv", index=False)
    pattern_rows = []
    for language in languages:
        selected = lang.loc[lang.primary_programming_language.eq(language)]
        rho = selected[["median_hours_to_merge", "overall_median_hours_to_merge"]].corr(method="spearman").iloc[0,1] if len(selected) >= 3 else np.nan
        pattern_rows.append({"language": language, "primary_language_prs": int((valid.primary_language.eq(language) & valid["mask"].gt(0)).sum()),
                             "categories_compared": len(selected), "spearman_r_vs_overall": rho,
                             "fastest_category": selected.loc[selected.median_hours_to_merge.idxmin(), "category"] if len(selected) else None,
                             "slowest_category": selected.loc[selected.median_hours_to_merge.idxmax(), "category"] if len(selected) else None,
                             "median_hour_range": selected.median_hours_to_merge.max()-selected.median_hours_to_merge.min() if len(selected) else np.nan})
        bars(selected, "category", "median_hours_to_merge", f"categories_{language.replace('+', 'p')}.png", f"{language}: median merge time by category", "Median hours to merge")
    pd.DataFrame(pattern_rows).to_csv(report_dir / "language_pattern_summary.csv", index=False)
    top = valid.primary_language.value_counts().head(12).index
    language_summary = valid.loc[valid.primary_language.isin(top)].groupby("primary_language").hours_to_merge.agg(["median", "mean", "count"]).reset_index()
    language_summary.to_csv(report_dir / "merge_time_by_language.csv", index=False)
    bars(language_summary, "primary_language", "median", "merge_time_by_language.png", "Median merge time by language", "Median hours to merge")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="cleaned_public.csv")
    p.add_argument("--output", default="df_title_body_with_categories.csv")
    p.add_argument("--report-dir", default=None)
    p.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--chunksize", type=int, default=25000)
    p.add_argument("--skip-reports", action="store_true")
    p.add_argument("--minimum-language-category-prs", type=int, default=30)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if source == output or output.exists():
        p.error("Choose a new output path; input and existing outputs are protected.")
    if args.chunksize < 1:
        p.error("--chunksize must be positive")
    columns = pd.read_csv(source, nrows=0).columns.tolist()
    text_columns = [c for c in TEXT_COLUMNS if c in columns]
    if not text_columns:
        p.error("No supported title/body text columns found")
    cudf = gpu_backend(args.backend)
    LOG.info("Input: %s; text columns: %s; chunk size: %s", source, text_columns, args.chunksize)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.csv")
    if partial.exists():
        p.error(f"Staging file exists from an earlier run: {partial}. Rename it before rerunning.")
    _, serialized = mask_lookups()
    counts = np.zeros(len(PR_CATEGORIES), dtype=np.int64)
    distribution = np.zeros(len(PR_CATEGORIES)+1, dtype=np.int64)
    report_parts, total, unicode_total = [], 0, 0
    dtypes = {c: "string" for c in ["repository_id", "pull_request_id", "user_id", "customer_id"] if c in columns}
    start = time.monotonic()
    with tqdm(desc="Categorized PRs", unit="PR") as progress:
        for chunk_number, chunk in enumerate(pd.read_csv(source, chunksize=args.chunksize, dtype=dtypes, low_memory=False), 1):
            chunk = chunk.reset_index(drop=True)
            masks, unicode_count = match_chunk(chunk, text_columns, cudf)
            codes = (masks.astype(np.uint16) * (1 << np.arange(len(PR_CATEGORIES), dtype=np.uint16))).sum(axis=1).astype(np.uint16)
            for j, category in enumerate(PR_CATEGORIES):
                chunk[f"kw_{category}"] = masks[:, j]
            chunk["keyword_categories"] = pd.Series(codes).map(serialized)
            category_counts = masks.sum(axis=1)
            chunk["keyword_category_count"] = category_counts
            chunk["has_keyword_match"] = category_counts > 0
            assert np.array_equal(chunk[[f"kw_{c}" for c in PR_CATEGORIES]].sum(axis=1).to_numpy(), category_counts)
            counts += masks.sum(axis=0)
            distribution += np.bincount(category_counts, minlength=len(distribution))
            chunk.to_csv(partial, mode="w" if chunk_number == 1 else "a", header=chunk_number == 1, index=False)
            if not args.skip_reports:
                report = pd.DataFrame({"mask": codes})
                report["pull_request_id"] = chunk.pull_request_id if "pull_request_id" in chunk else pd.Series(np.arange(total, total+len(chunk))).astype(str)
                report["minutes_to_merge"] = pd.to_numeric(chunk["minutes_to_merge"], errors="coerce") if "minutes_to_merge" in chunk else np.nan
                report["month"] = pd.to_datetime(chunk["merge_day"], errors="coerce").dt.to_period("M").astype("string") if "merge_day" in chunk else pd.NA
                report["primary_language"] = primary_languages(chunk.languages_list) if "languages_list" in chunk else None
                report_parts.append(report)
            total += len(chunk); unicode_total += unicode_count
            progress.update(len(chunk))
            LOG.info("Chunk %s complete: %s total PRs; %.1f seconds elapsed", chunk_number, f"{total:,}", time.monotonic()-start)
    if total == 0:
        pd.DataFrame(columns=columns+[f"kw_{c}" for c in PR_CATEGORIES]+["keyword_categories", "keyword_category_count", "has_keyword_match"]).to_csv(partial, index=False)
    partial.replace(output)
    if report_parts:
        LOG.info("Saving summary tables and plots (CPU)")
        save_reports(report_parts, counts, distribution, total, Path(args.report_dir).resolve() if args.report_dir else output.parent / "category_reports", args.minimum_language_category_prs)
    audit = {"input": str(source), "output": str(output), "backend": args.backend, "rows_processed": total,
             "unicode_rows_matched_on_cpu": unicode_total, "matched_prs": int(total-distribution[0]),
             "category_counts": dict(zip(PR_CATEGORIES, map(int, counts))), "text_columns": text_columns,
             "seconds": time.monotonic()-start, "no_rows_sampled_or_skipped": True,
             "cudf_version": getattr(cudf, "__version__", None), "pandas_version": pd.__version__}
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    LOG.info("DONE: saved all %s PRs to %s", f"{total:,}", output)


if __name__ == "__main__":
    main()
