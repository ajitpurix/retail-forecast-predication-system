
import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/raw/train.csv")

print("Original dataset shape:", df.shape)

# 1. Convert date into datetime
df["date"] = pd.to_datetime(df["date"])

# 2. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3. Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())
    
# 4. Check data types
print("\nData types:")
print(df.dtypes)

# 5. Check sales statistics
print("\nSales statistics:")
print(df["sales"].describe())

# 6. Check negative sales
print("\nNegative sales records:", (df["sales"] < 0).sum())

# 7. Remove exact duplicate rows, if any
df = df.drop_duplicates()

# 8. Sort records chronologically
df = df.sort_values(
    by=["store_nbr", "family", "date"]
).reset_index(drop=True)

# 9. Create processed folder if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# 10. Save cleaned dataset
df.to_csv("data/processed/train_cleaned.csv", index=False)

print("\nPreprocessing completed!")
print("Final dataset shape:", df.shape)
print("Saved to data/processed/train_cleaned.csv")
