import pandas as pd
import numpy as np
import requests
import joblib

# Load once at import time
HISTORY = pd.read_csv("shop_history.csv", parse_dates=["appearance_date"])
PREDICTOR = joblib.load("predictor.joblib")


def _run_gaps(dates):
    """Gaps in days between runs of consecutive shop days."""
    if len(dates) < 2:
        return []
    starts, ends = [dates[0]], []
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days > 1:
            ends.append(dates[i-1])
            starts.append(dates[i])
    ends.append(dates[-1])
    return [(starts[i] - ends[i-1]).days for i in range(1, len(ends))]


def lookup_item(name: str) -> dict:
    """Look up shop history for a single cosmetic by name (fuzzy match)."""
    matches = HISTORY[HISTORY["name"].str.lower().str.contains(name.lower(), na=False)]

    if matches.empty:
        return {"found": False, "message": f"No cosmetic found matching '{name}'."}

    top_id = matches["item_id"].value_counts().idxmax()
    matches = matches[matches["item_id"] == top_id]

    today = pd.Timestamp.now(tz="UTC")
    last_seen = matches["appearance_date"].max()
    days_since = (today - last_seen).days

    return {
        "found": True,
        "name": matches.iloc[0]["name"],
        "rarity": matches.iloc[0]["rarity"],
        "type": matches.iloc[0]["type"],
        "total_appearances": int(len(matches)),
        "first_seen": matches["appearance_date"].min().strftime("%Y-%m-%d"),
        "last_seen": last_seen.strftime("%Y-%m-%d"),
        "days_since_last_seen": int(days_since),
    }


def get_todays_shop() -> dict:
    """Get today's shop with scarcity info for each item."""
    r = requests.get("https://fortnite-api.com/v2/shop")
    data = r.json()
    today = pd.Timestamp.now(tz="UTC")

    items = []
    for entry in data["data"]["entries"]:
        if "brItems" not in entry:
            continue
        item = entry["brItems"][0]
        item_id = item["id"]
        item_history = HISTORY[HISTORY["item_id"] == item_id]

        if item_history.empty:
            scarcity_days = None
        else:
            past = item_history[item_history["appearance_date"] < today.normalize()]
            scarcity_days = int((today - past["appearance_date"].max()).days) if not past.empty else None

        items.append({
            "name": item["name"],
            "rarity": item.get("rarity", {}).get("value", "unknown"),
            "price_vbucks": entry["finalPrice"],
            "days_since_last_seen": scarcity_days,
        })

    return {"count": len(items), "items": items}


def find_rare_returns(min_days: int = 100) -> dict:
    """Find items in today's shop that haven't been seen in min_days+ days."""
    shop = get_todays_shop()
    rare = [i for i in shop["items"] if i["days_since_last_seen"] is not None and i["days_since_last_seen"] >= min_days]
    rare.sort(key=lambda x: x["days_since_last_seen"], reverse=True)
    return {"count": len(rare), "items": rare}


def predict_return(name: str) -> dict:
    """Predict probability that a Fortnite item returns to the shop within 30 days."""
    matches = HISTORY[HISTORY["name"].str.lower().str.contains(name.lower(), na=False)]
    if matches.empty:
        return {"found": False, "message": f"No item found matching '{name}'."}

    top_id = matches["item_id"].value_counts().idxmax()
    item_history = HISTORY[HISTORY["item_id"] == top_id].sort_values("appearance_date")
    item_name = item_history.iloc[0]["name"]

    today = pd.Timestamp.now(tz="UTC")
    past = item_history[item_history["appearance_date"] < today.normalize()]
    if past.empty:
        return {"found": False, "message": f"{item_name} has no past appearances to predict from."}

    dates = past["appearance_date"].tolist()
    days_since_last = (today - dates[-1]).days
    total_so_far = len(dates)
    item_age_days = (today - dates[0]).days

    gaps = _run_gaps(dates)
    if gaps:
        mean_gap = float(np.mean(gaps))
        median_gap = float(np.median(gaps))
        std_gap = float(np.std(gaps)) if len(gaps) > 1 else 0.0
    else:
        mean_gap = median_gap = std_gap = float("nan")

    last_90d_cutoff = today - pd.Timedelta(days=90)
    last_365d_cutoff = today - pd.Timedelta(days=365)
    appearances_last_90d = sum(1 for d in dates if d >= last_90d_cutoff)
    appearances_last_365d = sum(1 for d in dates if d >= last_365d_cutoff)

    rarity = item_history.iloc[0]["rarity"]
    item_type = item_history.iloc[0]["type"]

    features = {
        "days_since_last": days_since_last,
        "total_appearances_so_far": total_so_far,
        "item_age_days": item_age_days,
        "mean_gap": mean_gap,
        "median_gap": median_gap,
        "std_gap": std_gap,
        "appearances_last_90d": appearances_last_90d,
        "appearances_last_365d": appearances_last_365d,
        "month": today.month,
        "day_of_week": today.dayofweek,
    }
    for col in PREDICTOR["feature_columns"]:
        if col.startswith("rarity_"):
            features[col] = 1 if col == f"rarity_{rarity}" else 0
        elif col.startswith("type_"):
            features[col] = 1 if col == f"type_{item_type}" else 0

    X = pd.DataFrame([features])[PREDICTOR["feature_columns"]]
    prob = float(PREDICTOR["model"].predict_proba(X)[0, 1])

    return {
        "found": True,
        "name": item_name,
        "rarity": rarity,
        "probability_30d_return": round(prob, 3),
        "days_since_last_seen": int(days_since_last),
        "average_gap_days": round(mean_gap, 1) if not np.isnan(mean_gap) else None,
        "model_auc": 0.633,
        "model_disclaimer": "Moderate accuracy (AUC 0.63). Treat as informed estimate, not certainty.",
    }