import json
import os
import time

import pandas as pd
import requests


INPUT_FILE = "pr_human_review_sample.csv"
OUTPUT_FILE = "pr_human_review_sample_gpt_labels.csv"

MODEL = "openai/gpt-4.1-mini"
BATCH_SIZE = 15

url = "https://models.github.ai/inference/chat/completions"

headers = {
    "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
    "Accept": "application/vnd.github+json",
    "Content-Type": "application/json",
    "X-GitHub-Api-Version": "2026-03-10",
}


# Load previous progress when available
try:
    df = pd.read_csv(OUTPUT_FILE)
    print("Resuming previous run.")
except FileNotFoundError:
    df = pd.read_csv(INPUT_FILE)
    df["gpt_category"] = ""
    df["gpt_reason"] = ""
    df["gpt_status"] = ""


def clean(value, limit):
    if pd.isna(value):
        return ""
    return str(value)[:limit]


for start in range(0, len(df), BATCH_SIZE):
    indices = list(range(start, min(start + BATCH_SIZE, len(df))))

    # Skip completed rows
    indices = [
        index
        for index in indices
        if df.at[index, "gpt_status"] != "ok"
    ]

    if not indices:
        continue

    prs = [
        {
            "row_id": index,
            "title": clean(df.at[index, "title"], 300),
            "body": clean(df.at[index, "body"], 700),
        }
        for index in indices
    ]

    prompt = f"""
Categorize each pull request using one category:

feature, bug_fix, refactor, documentation, testing, dependency,
ci_build, security, performance, configuration, release, revert,
other, unclear.

Return JSON only:

{{
  "results": [
    {{
      "row_id": 0,
      "category": "bug_fix",
      "reason": "Fixes incorrect behavior."
    }}
  ]
}}

Pull requests:
{json.dumps(prs)}
"""

    while True:
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You categorize software pull requests.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 1500,
                "temperature": 0,
            },
            timeout=120,
        )

        if response.status_code == 429:
            retry_after = int(
                response.headers.get("Retry-After", 60)
            )
            print(f"Rate limited. Retrying in {retry_after} seconds.")
            time.sleep(retry_after)
            continue

        response.raise_for_status()
        break

    content = response.json()["choices"][0]["message"]["content"]
    results = json.loads(content)["results"]

    for result in results:
        row_id = int(result["row_id"])

        df.at[row_id, "gpt_category"] = result["category"]
        df.at[row_id, "gpt_reason"] = result["reason"]
        df.at[row_id, "gpt_status"] = "ok"

    # Save after every request
    df.to_csv(OUTPUT_FILE, index=False)

    completed = df["gpt_status"].eq("ok").sum()
    print(f"Processed {completed}/{len(df)}")

    # Keep requests below the 15-per-minute limit
    time.sleep(5)


print(f"Done. Saved to {OUTPUT_FILE}")