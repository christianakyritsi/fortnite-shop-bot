import pandas as pd
import numpy as np
from datetime import timedelta

PREDICTION_WINDOW_DAYS = 30

# Load history
history = pd.read_csv("shop_history.csv", parse_dates=["appearance_date"])
history = history.sort_values("appearance_date").reset_index(drop=True)

print(f"Loaded {len(history):,} appearances for {history['item_id'].nunique():,} items")
print(f"Date range: {history['appearance_date'].min().date()} to {history['appearance_date'].max().date()}")

# Build (item_id, reference_date) prediction rows
# For each item, generate a prediction row at every distinct shop date AFTER its first appearance
all_shop_dates = sorted(history["appearance_date"].unique())
print(f"Distinct shop dates: {len(all_shop_dates)}")

# To keep dataset manageable, sample ~weekly reference dates
sampled_dates = all_shop_dates[::7]
print(f"Using {len(sampled_dates)} sampled reference dates (weekly)")

rows = []
items = history["item_id"].unique()

for item_id in items:
    item_history = history[history["item_id"] == item_id].sort_values("appearance_date")
    item_dates = item_history["appearance_date"].tolist()
    item_name = item_history.iloc[0]["name"]
    item_rarity = item_history.iloc[0]["rarity"]
    item_type = item_history.iloc[0]["type"]
    first_seen = item_dates[0]

    for ref_date in sampled_dates:
        # Skip dates before this item ever existed
        if ref_date < first_seen:
            continue
        # Skip dates too close to the end (no future data to compute label)
        if ref_date > all_shop_dates[-1] - pd.Timedelta(days=PREDICTION_WINDOW_DAYS):
            continue

        # Past appearances (strictly before ref_date)
        past = [d for d in item_dates if d < ref_date]
        if not past:
            continue

        # Future appearances within window
        future_window_end = ref_date + pd.Timedelta(days=PREDICTION_WINDOW_DAYS)
        returned = any(ref_date <= d <= future_window_end for d in item_dates)

        # Features
        days_since_last = (ref_date - past[-1]).days
        total_so_far = len(past)
        item_age_days = (ref_date - first_seen).days

        if len(past) >= 2:
            gaps = [(past[i] - past[i-1]).days for i in range(1, len(past))]
            mean_gap = np.mean(gaps)
            median_gap = np.median(gaps)
            std_gap = np.std(gaps) if len(gaps) > 1 else 0
        else:
            mean_gap = median_gap = std_gap = np.nan

        last_90d_cutoff = ref_date - pd.Timedelta(days=90)
        last_365d_cutoff = ref_date - pd.Timedelta(days=365)
        appearances_last_90d = sum(1 for d in past if d >= last_90d_cutoff)
        appearances_last_365d = sum(1 for d in past if d >= last_365d_cutoff)

        rows.append({
            "item_id": item_id,
            "name": item_name,
            "rarity": item_rarity,
            "type": item_type,
            "ref_date": ref_date,
            "days_since_last": days_since_last,
            "total_appearances_so_far": total_so_far,
            "item_age_days": item_age_days,
            "mean_gap": mean_gap,
            "median_gap": median_gap,
            "std_gap": std_gap,
            "appearances_last_90d": appearances_last_90d,
            "appearances_last_365d": appearances_last_365d,
            "month": ref_date.month,
            "day_of_week": ref_date.dayofweek,
            "returned_within_30d": int(returned),
        })

df = pd.DataFrame(rows)
print(f"\nTraining rows: {len(df):,}")
print(f"Positive rate: {df['returned_within_30d'].mean():.3f}")
print(f"Date range of ref_date: {df['ref_date'].min().date()} to {df['ref_date'].max().date()}")
print()
print("Sample rows:")
print(df.head())

df.to_csv("training_data.csv", index=False)
print(f"\nSaved to training_data.csv")