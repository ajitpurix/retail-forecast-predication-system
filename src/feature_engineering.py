
import pandas as pd
import os

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/train_cleaned.csv")

# 2. Convert date column
df["date"] = pd.to_datetime(df["date"])

# 3. Sort data chronologically
df = df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)

# 4. Create date features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.weekday

df["is_weekend"] = (
    df["weekday"] >= 5
).astype(int)

# 5. Group sales by store and product family
grouped_sales = df.groupby(
    ["store_nbr", "family"]
)["sales"]

# 6. Create lag features
df["lag_1"] = grouped_sales.shift(1)
df["lag_7"] = grouped_sales.shift(7)
df["lag_14"] = grouped_sales.shift(14)

# 7. Create rolling average features
df["rolling_mean_7"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

df["rolling_mean_30"] = (
    df.groupby(["store_nbr", "family"])["sales"]
    .transform(
        lambda x: x.shift(1).rolling(30).mean()
    )
)

# 8. Save engineered dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/train_features.csv",
    index=False
)

# 9. Display results
print("Feature engineering completed!")

print("\nNew columns:")
print(df.columns.tolist())

print("\nDataset preview:")
print(df.head(10))

print("\nMissing values in new features:")
print(
    df[
        [
            "lag_1",
            "lag_7",
            "lag_14",
            "rolling_mean_7",
            "rolling_mean_30"
        ]
    ].isnull().sum()
)
