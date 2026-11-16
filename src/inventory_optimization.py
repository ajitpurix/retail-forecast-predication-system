
import pandas as pd
import numpy as np
import os

# ----------------------------------
# 1. Load datasets
# ----------------------------------

if not os.path.exists("reports/future_predictions.csv"):
    raise FileNotFoundError(
        "reports/future_predictions.csv not found! "
        "Please run 'python src/predict_future.py' first to generate future predictions."
    )

predictions = pd.read_csv("reports/future_predictions.csv")
historical = pd.read_csv("data/raw/train.csv")

if not os.path.exists("data/raw/inventory.csv"):
    print("data/raw/inventory.csv not found. Auto-generating sample inventory data...")
    store_families = predictions[["store_nbr", "family"]].drop_duplicates().reset_index(drop=True)
    daily_mean = predictions.groupby(["store_nbr", "family"])["sales"].mean().reset_index()
    merged = store_families.merge(daily_mean, on=["store_nbr", "family"])
    
    np.random.seed(42)
    merged["lead_time_days"] = np.random.choice([2, 3, 4, 5, 7], size=len(merged))
    stock_multiplier = np.random.uniform(0.5, 10.0, size=len(merged))
    merged["current_stock"] = np.round(np.maximum(5, merged["sales"] * stock_multiplier)).astype(int)
    
    inventory = merged[["store_nbr", "family", "current_stock", "lead_time_days"]]
    os.makedirs("data/raw", exist_ok=True)
    inventory.to_csv("data/raw/inventory.csv", index=False)
    print("Auto-generated data/raw/inventory.csv with 1782 records.")
else:
    inventory = pd.read_csv("data/raw/inventory.csv")

historical["date"] = pd.to_datetime(
    historical["date"]
)

predictions["date"] = pd.to_datetime(
    predictions["date"]
)

# ----------------------------------
# 2. Calculate predicted daily demand
# ----------------------------------

predicted_demand = (
    predictions
    .groupby(["store_nbr", "family"])["sales"]
    .mean()
    .reset_index(name="average_predicted_demand")
)

# ----------------------------------
# 3. Calculate historical demand variation
# ----------------------------------

historical["physical_demand"] = (
    historical["sales"].clip(lower=0)
)

historical_daily = (
    historical
    .groupby(
        ["store_nbr", "family", "date"]
    )["physical_demand"]
    .sum()
    .reset_index()
)

demand_stats = (
    historical_daily
    .groupby(["store_nbr", "family"])["physical_demand"]
    .agg(
        historical_mean="mean",
        historical_std="std"
    )
    .reset_index()
)

# ----------------------------------
# 4. Merge datasets
# ----------------------------------

df = inventory.merge(
    predicted_demand,
    on=["store_nbr", "family"],
    how="left"
)

df = df.merge(
    demand_stats,
    on=["store_nbr", "family"],
    how="left"
)

# Check missing data
if df[
    [
        "average_predicted_demand",
        "historical_std"
    ]
].isnull().any().any():

    raise ValueError(
        "Some inventory combinations have no matching "
        "prediction or historical demand statistics."
    )

# ----------------------------------
# 5. Calculate safety stock
# ----------------------------------

SERVICE_LEVEL_Z = 1.65

df["safety_stock"] = (
    SERVICE_LEVEL_Z
    * df["historical_std"].fillna(0)
    * np.sqrt(df["lead_time_days"])
)

# ----------------------------------
# 6. Calculate reorder point
# ----------------------------------

df["lead_time_demand"] = (
    df["average_predicted_demand"]
    * df["lead_time_days"]
)

df["reorder_point"] = (
    df["lead_time_demand"]
    + df["safety_stock"]
)

# ----------------------------------
# 7. Calculate reorder quantity
# ----------------------------------

df["reorder_quantity"] = np.ceil(
    (
        df["reorder_point"]
        - df["current_stock"]
    ).clip(lower=0)
)

# ----------------------------------
# 8. Identify stockout risk
# ----------------------------------

df["stockout_risk"] = np.where(
    df["current_stock"] < df["lead_time_demand"],
    "HIGH",
    np.where(
        df["current_stock"] < df["reorder_point"],
        "MEDIUM",
        "LOW"
    )
)

# ----------------------------------
# 9. Generate recommendations
# ----------------------------------

df["recommendation"] = np.where(
    df["reorder_quantity"] > 0,
    "REORDER",
    "STOCK SUFFICIENT"
)

# ----------------------------------
# 10. Save output
# ----------------------------------

os.makedirs("reports", exist_ok=True)

df.to_csv(
    "reports/inventory_recommendations.csv",
    index=False
)

print("\nINVENTORY OPTIMIZATION RESULTS")
print("--------------------------------")

print(
    df[
        [
            "store_nbr",
            "family",
            "current_stock",
            "reorder_point",
            "reorder_quantity",
            "stockout_risk",
            "recommendation"
        ]
    ].to_string(index=False)
)

print("\nInventory optimization completed!")
print("Saved to reports/inventory_recommendations.csv")
