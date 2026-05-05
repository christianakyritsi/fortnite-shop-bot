import pandas as pd
import requests

# Load once at import time
HISTORY = pd.read_csv("shop_history.csv", parse_dates=["appearance_date"])


def lookup_item(name: str) -> dict:
    """Look up shop history for a single cosmetic by name (fuzzy match)."""
    matches = HISTORY[HISTORY["name"].str.lower().str.contains(name.lower(), na=False)]

    if matches.empty:
        return {"found": False, "message": f"No cosmetic found matching '{name}'."}

    # Pick the most-appeared match
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