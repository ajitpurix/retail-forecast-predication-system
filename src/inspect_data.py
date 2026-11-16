
import pandas as pd
import os

# Project data directory
DATA_PATH = "data/raw"

files = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "holidays_events.csv"
]

for file in files:
    path = os.path.join(DATA_PATH, file)

    print("\n" + "=" * 60)
    print(f"FILE: {file}")
    print("=" * 60)

    if not os.path.exists(path):
        print("File not found:", path)
        continue

    df = pd.read_csv(path)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())
