import requests
import pandas as pd

# responseFlags=4 enables INCLUDE_SHOP_HISTORY (3rd bit in the IntFlag)
# Use 7 to be safe (enables paths + gameplay_tags + shop_history)
r = requests.get(
    "https://fortnite-api.com/v2/cosmetics/br",
    params={"responseFlags": 7}
)
items = r.json()["data"]
print(f"Total cosmetics: {len(items)}")

with_history = [i for i in items if i.get("shopHistory")]
print(f"With shop history: {len(with_history)}")

rows = []
for item in with_history:
    for date in item["shopHistory"]:
        rows.append({
            "item_id": item["id"],
            "name": item["name"],
            "rarity": (item.get("rarity") or {}).get("value"),
            "type": (item.get("type") or {}).get("value"),
            "appearance_date": date,
        })

df = pd.DataFrame(rows)
df["appearance_date"] = pd.to_datetime(df["appearance_date"])
df = df.sort_values("appearance_date").reset_index(drop=True)

print(f"\n{len(df)} appearances")
print(f"Range: {df['appearance_date'].min().date()} → {df['appearance_date'].max().date()}")
df.to_csv("shop_history.csv", index=False)
print("Saved.")