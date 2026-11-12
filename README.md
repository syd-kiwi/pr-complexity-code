# PR Complexity Research

This repository contains code for two studies:

| Folder | Purpose |
| --- | --- |
| [paper1/](paper1/) | Calculate PR complexity and study its relationship with merge time |
| [paper2/](paper2/README.md) | Compare PRs by Copilot access and test merge time prediction models |

## Paper 1: PR complexity

Open the notebooks in [paper1/notebooks/](paper1/notebooks/) and follow this order:

1. Build the dataset.
2. Clean the data.
3. Assign keyword categories.
4. Calculate complexity scores.
5. Compare score components.

Already have PR data? Start with notebook 2. Notebook 1 requires internal GitHub data access.

Install the main notebook packages and open Jupyter:

```bash
python -m pip install jupyterlab pandas numpy matplotlib seaborn scipy scikit-learn joblib tqdm langdetect wordcloud
jupyter lab
```

Check input and output paths before running. The root `requirements.txt` currently has a merge conflict, so use the command above for the main notebook packages.

## Paper 2: Copilot and merge time

The study asks:

1. How do PR characteristics and merge time differ by Copilot access?
2. Can machine learning predict merge time using limited features?

Follow the [Paper 2 instructions](paper2/README.md).

## Manuscripts

- [PR complexity paper](https://github.com/syd-kiwi/pr-complexity-paper/tree/main/paper)
- [Copilot and merge time paper](https://github.com/syd-kiwi/An-Empirical-Study-of-the-Pull-Based-Software-Development-Model/tree/main/paper)

Data is not included. Copilot access does not confirm tool use.
