"""Generate a synthetic dataset with the same shape as the one the notebook expects.

No real doctors, institutions or usage are involved: every value is simulated.
Run from the repository root:  python make_synthetic_data.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

N_DAYS = 60  # days simulated after each doctor's first session
SPECIALTIES = ["generalMedicine"] * 8 + ["cardiology", "pediatrics", "orthopedics", "internalMedicine"]
MONTHS = pd.to_datetime(["2026-06-01", "2026-07-01", "2026-08-01"])
# institution sizes: one very large, several mid-sized, a few too small to analyse
SIZES = [700, 200, 180, 170, 120, 110, 100, 95, 90, 70, 60, 50, 40, 35, 30, 25, 22, 15, 12, 8, 5, 3]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


rows, weekly = [], []
doctor = 0
for i, size in enumerate(SIZES):
    name = f"Institution {chr(65 + i)}"
    engagement_mean = rng.normal(0, 0.8)      # how heavily this institution's doctors use the tool
    quit_effect = rng.normal(0, 0.5)          # extra tendency to quit, beyond usage
    month_weights = rng.dirichlet([1, 1, 1])  # institutions onboard in waves

    for _ in range(size):
        doctor += 1
        account_id = f"doc_{doctor:05d}"
        month = MONTHS[rng.choice(3, p=month_weights)]
        first = month + pd.Timedelta(days=int(rng.integers(0, month.days_in_month)))
        e = rng.normal(engagement_mean, 1)

        # how long the doctor keeps using the tool
        if rng.random() < sigmoid(-1.5 - 1.2 * e + quit_effect):
            lifetime = 1 + int(rng.exponential(4))          # early quitter
        else:
            lifetime = 7 * int(rng.geometric(0.015))         # stays for many weeks

        days = np.arange(N_DAYS)
        dates = first + pd.to_timedelta(days, unit="D")
        weekend = dates.dayofweek >= 5
        p_active = sigmoid(0.3 + 0.9 * e) * np.where(weekend, 0.15, 1.0)
        active = (rng.random(N_DAYS) < p_active) & (days < lifetime)
        active[0] = True
        sessions = np.where(active, 1 + np.minimum(rng.poisson(np.exp(1.0 + 0.8 * e), N_DAYS), 40), 0)

        next_month = month + pd.offsets.MonthBegin(1)
        cutoff = next_month + pd.Timedelta(days=20 if month.month == 6 else 19)
        active_next = active[(dates >= next_month) & (dates <= cutoff)].any()

        rows.append({
            "account_id": account_id,
            "cohort_month": month.date(),
            "institutional_name": name,
            "primary_specialty": rng.choice(SPECIALTIES),
            "first_session_date": first.date(),
            "sessions_day1": int(sessions[0]),
            "sessions_first7d": int(sessions[:7].sum()),
            "active_days_first7d": int(active[:7].sum()),
            "churned": int(not active_next),
        })
        week = {"account_id": account_id}
        for k in range(4):
            week[f"s_w{k + 1}"] = int(sessions[7 * k:7 * k + 7].sum())
            week[f"a_w{k + 1}"] = int(active[7 * k:7 * k + 7].sum())
        weekly.append(week)

roster = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
roster.to_csv("data/cohort_roster.csv", index=False)
pd.DataFrame(weekly).to_csv("data/weekly_usage.csv", index=False)

print(len(roster), "synthetic doctors,", roster["institutional_name"].nunique(), "institutions,",
      "churn rate", round(roster["churned"].mean(), 3))
