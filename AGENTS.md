# AGENTS.md

Guidance for AI coding agents (Claude Code, Cursor, etc.) working in this repo.
Human contributors should read it too — the conventions apply to everyone.

## What this project is

Binary classification of NASA Kepler Objects of Interest (KOI) as **planet
candidate** vs **false positive**. MSc COMP702 coursework.

Three tuned scikit-learn pipelines are compared: Logistic Regression, Random
Forest, and an MLP.

## Layout

| Path | What it is |
|---|---|
| `exoplanet_detection.ipynb` | The analysis. Loads the KOI table, cleans it, tunes and evaluates all three models, and writes `models/` + `results/`. Source of truth for every number. |
| `app.py` | Streamlit demo. Loads the saved artefacts and renders them. Trains nothing. |
| `models/` | Trained pipelines (`*_tuned_pipeline.joblib`) + `feature_columns.joblib`. Committed. |
| `results/` | Metrics CSVs, `summary.txt`, and the `outputs_*.png` figures. Committed. |
| `data/` | Raw KOI CSV. **Gitignored** — download `cumulative_koi.csv` from the NASA Exoplanet Archive. |
| `.streamlit/config.toml` | Demo theme. |
| `test_dataset_20_percent.csv` | Held-out test split, for trying the app's Predict tab. |

## Rules

### Never hardcode results
Every metric, table, and figure in `app.py` is read from `results/` or computed
from a pipeline in `models/`. If a number can't be traced back to a notebook
run, it doesn't belong in the app. Don't "fix" a chart by typing in a value.

### The notebook owns training; the app owns display
Model changes (features, hyperparameters, the split) go in the notebook, which
is then re-run to regenerate `models/` and `results/`. Don't train, refit, or
resample inside `app.py`.

### Don't leak the label
`target` is the label and `kepid` is only a grouping key — neither may reach a
model as an input feature. The train/test split is grouped by host star
(`kepid`) so observations of the same star can't straddle the split.
`results/leakage_columns_removed.csv` records the disposition columns dropped
for this reason. Preserve these guarantees.

### Keep the environment pinned
`requirements.txt` is pinned. The `.joblib` files are version-sensitive — a
different scikit-learn major version may fail to load them or fall back to an
approximate confidence. Don't loosen the pins casually.

### Don't commit data
`data/` stays gitignored. Don't add the raw KOI CSV, credentials, or `.env`.

## Running things

Install:

    pip install -r requirements.txt

Run the demo (it hot-reloads on save):

    streamlit run app.py

Re-run the analysis: open `exoplanet_detection.ipynb` in Jupyter and run all
cells. **Restart the kernel first** — an open Jupyter session does not pick up
edits made to the `.ipynb` on disk, and saving from a stale session will
overwrite them.

### Working on the notebook alongside an agent
Jupyter and an agent both write `exoplanet_detection.ipynb` directly, and
neither notices the other. Before asking an agent to edit the notebook: save
and **close** it in Jupyter. Afterwards, reopen it. Don't leave it open in a
browser tab while edits are happening.

## Checkpoints and reverting

A `PreToolUse` hook (`.claude/settings.json` → `.claude/checkpoint.sh`) runs
before any agent file edit. If the working tree is dirty it commits everything
as:

    checkpoint: before agent edit (YYYY-MM-DD HH:MM:SS)

So there is always a commit representing the state just before a change. A run
of consecutive edits produces one checkpoint, not one per edit.

Find them and go back:

    git log --oneline --grep='^checkpoint:'
    git revert <sha>        # safe: undoes it as a new commit
    git reset --hard <sha>  # discards everything after <sha>

Checkpoints are ordinary commits and are meant to be temporary. Before pushing,
squash them into real commits:

    git rebase -i <sha-before-the-checkpoints>

To disable the hook, delete the `hooks` block from `.claude/settings.json`.

## Commits

Write a real commit message for real work — what changed and why, not "update
files". Don't push to `main` without being asked. Don't commit `models/` or
`results/` churn unless the notebook was actually re-run.
