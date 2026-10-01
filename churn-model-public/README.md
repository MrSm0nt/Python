# Early churn model for a medical scribe product

A small, readable analysis of early churn among doctors using an AI medical scribe: which doctors and institutions are more likely to stop using it, and how early that can be detected.

**All data in this repository is synthetic.** `make_synthetic_data.py` simulates doctors, institutions and usage; no real people, organizations or results are included. The numbers in the notebook illustrate the method only.

## What the notebook does

`churn-model.ipynb` has two parts:

- **Part A — Monthly churn.** A doctor has churned if they have no consultation in the first 20 days of the month after their first session.
- **Part B — 4-week retention.** A doctor is retained if they have at least one consultation on days 22–28 after their first session.

Each part runs the same steps:

1. Filter the cohort and encode institution and specialty.
2. Fit an L2-regularized logistic regression on institution, specialty and first-week usage.
3. Compute an *adjusted* rate per institution: the predicted rate if every doctor belonged to that institution, keeping their own first-week usage. This separates an institution's own effect from how heavily its doctors use the tool.
4. Fit a shallow decision tree to get simple, readable rules.

Two details worth noting:

- **Leakage guard.** In Part A, doctors who start in the last days of a month are excluded, because their "first week" would overlap the period used to decide churn.
- **Validation.** The model is tested on a held-out month, then refitted on all months with 5-fold cross-validation for the institution comparison.

## Run it

```bash
pip install -r requirements.txt
```

```bash
python make_synthetic_data.py
```

Then open `churn-model.ipynb` and run all cells. The synthetic data is already in `data/`, so the second command is only needed to regenerate it.

## Files

| File | Purpose |
|---|---|
| `churn-model.ipynb` | The analysis, with outputs |
| `make_synthetic_data.py` | Simulates the dataset |
| `data/cohort_roster.csv` | One row per synthetic doctor |
| `data/weekly_usage.csv` | Weekly usage for each doctor's first four weeks |
