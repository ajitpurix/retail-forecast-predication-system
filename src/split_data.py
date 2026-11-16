
import pandas as pd
import os

# 1. Load feature-engineered dataset
df = pd.read_csv("data/processed/train_features.csv")

# 2. Convert date column
df["date"] = pd.to_datetime(df["date"])

# 3. Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

# 4. Select validation period
validation_days = 16

unique_dates = sorted(df["date"].unique())

validation_start = unique_dates[-validation_days]

# 5. Split training and validation data
train_df = df[df["date"] < validation_start].copy()

validation_df = df[df["date"] >= validation_start].copy()

# 6. Separate input features and target
target = "sales"

drop_columns = ["id", "date", "sales"]

feature_columns = [
    col for col in df.columns
    if col not in drop_columns
]

X_train = train_df[feature_columns]
y_train = train_df[target]

X_validation = validation_df[feature_columns]
y_validation = validation_df[target]

# 7. Save datasets
os.makedirs("data/processed", exist_ok=True)

train_df.to_csv(
    "data/processed/train_split.csv",
    index=False
)

validation_df.to_csv(
    "data/processed/validation_split.csv",
    index=False
)

# 8. Display results
print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))

print("\nTraining date range:")
print(train_df["date"].min(), "to", train_df["date"].max())

print("\nValidation date range:")
print(
    validation_df["date"].min(),
    "to",
    validation_df["date"].max()
)

print("\nFeatures:")
print(feature_columns)

print("\nData splitting completed!")
