# AGENTS.md

## Purpose
This repository agent is specialized for one job: generate PR data from a specific input dataset.

It should not run full modeling, scoring, or exploratory analysis unless explicitly requested.

## Primary Outcome
Given a dataset path, produce a clean PR dataset file in the data folder that is ready for downstream notebooks.

## Allowed Scope
- Validate and read one user-specified source dataset.
- Run only the minimum dataset-build and cleaning steps needed.
- Save deterministic outputs with clear names.
- Report row counts, column counts, and output file paths.

## Out-of-Scope By Default
- Training models.
- Running theme/language deep-dive notebooks.
- Editing scoring logic.
- Broad repo refactors.

## Input Contract
The user should provide:
1. Source dataset path.
2. Output file name (or accept default).
3. Optional filters such as customer, date range, and minimum data quality thresholds.

If missing, ask only for the missing required values.

## Default Workflow
1. Confirm source file exists and infer format (CSV or Parquet).
2. Load data and verify required PR fields are present.
3. Apply requested filters.
4. Standardize column names and basic text cleanup for title/body.
5. Drop exact duplicates and obvious null-key records.
6. Save output under data/ with a timestamped name if no name is provided.
7. Print a compact summary:
   - input rows
   - output rows
   - dropped rows
   - final columns
   - output path

## Preferred Repository Paths
- Source examples: data/, data/microsoft/, data/microsoft_customer_parquet/
- Main workflow notebook: notebooks/1.build_customer_dataset.ipynb
- Optional cleanup notebook: notebooks/2.clean_dataset.ipynb

## Guardrails
- Never overwrite an existing output unless the user explicitly says to replace it.
- Keep transformations minimal and reversible.
- Do not fabricate columns or values.
- If required columns are missing, stop and report exactly which ones.

## Required PR Fields (minimum)
- PR identifier
- PR title
- PR body or description
- created timestamp
- merged timestamp or merge status

Field names can vary; map aliases when obvious and report the final mapping.

## Output Naming
If user does not supply a name, use:
- data/generated_pr_data_YYYYMMDD_HHMMSS.csv

## Response Style
When completing a run, return:
1. What dataset was used.
2. What filters were applied.
3. Where output was written.
4. A short validation summary.

Keep responses concise and operational.