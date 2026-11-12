#!/usr/bin/env python3
"""RQ1: describe PR characteristics, merge time, and Copilot access groups."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (TARGET, STRUCTURAL, REVIEW, TEXT, dump_json, fingerprint,
                    fresh_output, setup_logging, LOG)


def summary(df, group):
    measures = [TARGET] + STRUCTURAL + REVIEW + TEXT + ["language_count"]
    return df.groupby(group, dropna=False, observed=True)[measures].agg(["count", "median", "mean"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="prepare.py's prepared.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-group-prs", type=int, default=5, help="Minimum PRs per Copilot group within a repository")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.min_group_prs < 1 or args.bootstrap < 1:
        parser.error("min-group-prs and bootstrap must be positive")
    setup_logging()
    df = pd.read_parquet(args.input)
    out = fresh_output(args.output)
    LOG.info("RQ1: analyzing %s merged PRs", len(df))
    measures = [TARGET] + STRUCTURAL + REVIEW + TEXT + ["language_count"]
    df[measures].describe(percentiles=[.25, .5, .75, .9, .95, .99]).to_csv(out / "overall_summary.csv")
    for group in ["had_copilot_seat", "is_code_pr", "is_open_source", "first_language"]:
        table = summary(df, group)
        table.columns = [f"{c}_{stat}" for c, stat in table.columns]
        table.to_csv(out / f"by_{group}.csv")
    correlations = []
    for label, subset in [("all", df), ("copilot_access", df.loc[df.had_copilot_seat.eq(1)]),
                          ("no_copilot_access", df.loc[df.had_copilot_seat.eq(0)])]:
        for feature in STRUCTURAL + REVIEW + TEXT + ["language_count"]:
            pair = subset[[feature, TARGET]].dropna()
            rho = pair[feature].corr(pair[TARGET], method="spearman") if len(pair) > 2 and pair[feature].nunique() > 1 and pair[TARGET].nunique() > 1 else np.nan
            correlations.append(dict(group=label, feature=feature, n=len(pair), spearman=rho))
    pd.DataFrame(correlations).to_csv(out / "spearman_correlations.csv", index=False)

    # Equal repository weighting: differences in group medians within each repo.
    known = df.loc[df.had_copilot_seat.notna()]
    grouped = known.groupby(["repository_id", "had_copilot_seat"])[TARGET].agg(["size", "median"])
    counts = grouped["size"].unstack().reindex(columns=[0., 1.])
    medians = grouped["median"].unstack().reindex(columns=[0., 1.])
    eligible = counts.ge(args.min_group_prs).all(axis=1)
    differences = (medians.loc[eligible, 1.] - medians.loc[eligible, 0.]).rename("copilot_minus_no_copilot_median_minutes")
    paired = counts.loc[eligible].rename(columns={0.: "no_access_n", 1.: "access_n"}).join(differences)
    paired.to_csv(out / "within_repository_differences.csv")
    effect = {"eligible_repositories": len(differences), "minimum_prs_per_group": args.min_group_prs,
              "estimand": "median across repositories of within-repository differences in group median merge minutes",
              "interpretation": "Descriptive association. Repository stratification does not eliminate confounding.",
              "estimate_minutes": None, "bootstrap_95_ci_minutes": None}
    if len(differences):
        rng = np.random.default_rng(args.seed)
        values = differences.to_numpy()
        draws = [float(np.median(rng.choice(values, size=len(values), replace=True))) for _ in range(args.bootstrap)]
        effect.update(estimate_minutes=float(np.median(values)))
        if len(values) >= 2:
            effect["bootstrap_95_ci_minutes"] = np.quantile(draws, [.025, .975]).tolist()
    dump_json(out / "within_repository_effect.json", effect)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for value, label in [(0, "No Copilot access"), (1, "Copilot access")]:
        minutes = df.loc[df.had_copilot_seat.eq(value), TARGET]
        if len(minutes):
            quantiles = np.linspace(0, 1, 201)
            ax.plot(np.log1p(np.quantile(minutes, quantiles)), quantiles, label=f"{label} (n={len(minutes):,})")
    ax.set(xlabel="log(1 + minutes to merge)", ylabel="Cumulative proportion of merged PRs")
    if df.had_copilot_seat.notna().any():
        ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "copilot_merge_time_ecdf.png", dpi=180)
    fig.savefig(out / "copilot_merge_time_ecdf.pdf")
    plt.close(fig)
    dump_json(out / "analysis_manifest.json", {
        "source": fingerprint(args.input), "arguments": vars(args), "rows": len(df),
        "unknown_copilot_rows": int(df.had_copilot_seat.isna().sum()),
        "notes": ["All results concern merged PRs only.", "Seat status measures access, not use or code authorship.",
                  "first_language is the first supplied list entry, not necessarily the dominant language.",
                  "Review counts and other features may reflect final PR state.",
                  "These analyses do not measure separate review stages or establish causal effects."]})
    LOG.info("RQ1 outputs written to %s", out)


if __name__ == "__main__":
    main()
