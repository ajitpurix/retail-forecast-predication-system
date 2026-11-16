
import pandas as pd
import numpy as np
import os
import joblib

from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# ---------------------------------------
# 1. Load historical and future datasets
# ---------------------------------------

train_df = pd.read_csv("data/raw/train.csv")
test_df = pd.read_csv("data/raw/test.csv")

train_df["date"] = pd.to_datetime(train_df["date"])
test_df["date"] = pd.to_datetime(test_df["date"])

train_df = train_df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)

test_df = test_df.sort_values(
    ["date", "store_nbr", "family"]
).reset_index(drop=True)

# ---------------------------------------
# 2. Create training features
# ---------------------------------------

def create_features(df):

    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday

    df["is_weekend"] = (
        df["weekday"] >= 5
    ).astype(int)

    grouped = df.groupby(
        ["store_nbr", "family"]
    )["sales"]

    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)

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

    return df


train_features = create_features(train_df)

# ---------------------------------------
# 3. Select training features
# ---------------------------------------

target = "sales"

drop_columns = ["id", "date", "sales"]

feature_columns = [
    col for col in train_features.columns
    if col not in drop_columns
]

X_train = train_features[feature_columns]
y_train = train_features[target]

numeric_columns = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# ---------------------------------------
# 4. Preprocessing pipeline
# ---------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# ---------------------------------------
# 5. Create final XGBoost model
# ---------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    ))
])

print("Training final XGBoost model...")

model.fit(X_train, y_train)

print("Final model training completed!")

# ---------------------------------------
# 6. Build historical sales memory
# ---------------------------------------

history = defaultdict(list)

for row in train_df.itertuples(index=False):

    key = (row.store_nbr, row.family)

    history[key].append(float(row.sales))

# ---------------------------------------
# 7. Predict test dates recursively
# ---------------------------------------

all_predictions = []

for current_date, daily_df in test_df.groupby(
    "date", sort=True
):

    feature_rows = []

    for row in daily_df.itertuples(index=False):

        key = (row.store_nbr, row.family)
        past_sales = history[key]

        weekday = current_date.weekday()

        features = {
            "store_nbr": row.store_nbr,
            "family": row.family,
            "onpromotion": row.onpromotion,
            "year": current_date.year,
            "month": current_date.month,
            "day": current_date.day,
            "weekday": weekday,
            "is_weekend": int(weekday >= 5),

            "lag_1": (
                past_sales[-1]
                if len(past_sales) >= 1 else np.nan
            ),

            "lag_7": (
                past_sales[-7]
                if len(past_sales) >= 7 else np.nan
            ),

            "lag_14": (
                past_sales[-14]
                if len(past_sales) >= 14 else np.nan
            ),

            "rolling_mean_7": (
                np.mean(past_sales[-7:])
                if len(past_sales) >= 7 else np.nan
            ),

            "rolling_mean_30": (
                np.mean(past_sales[-30:])
                if len(past_sales) >= 30 else np.nan
            )
        }

        feature_rows.append(features)

    X_day = pd.DataFrame(feature_rows)
    X_day = X_day.reindex(columns=feature_columns)

    predictions = model.predict(X_day)

    for row, prediction in zip(
        daily_df.itertuples(index=False),
        predictions
    ):

        # Non-negative predicted demand
        prediction = max(0.0, float(prediction))

        all_predictions.append({
            "id": row.id,
            "date": current_date,
            "store_nbr": row.store_nbr,
            "family": row.family,
            "sales": prediction
        })

        # Add prediction to history for the next date
        key = (row.store_nbr, row.family)
        history[key].append(prediction)

    print("Predicted:", current_date.date())

# ---------------------------------------
# 8. Save model and predictions
# ---------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

joblib.dump(model, "models/final_xgboost.pkl")

predictions_df = pd.DataFrame(all_predictions)

predictions_df.to_csv(
    "reports/future_predictions.csv",
    index=False
)

# Kaggle-style submission file
submission = predictions_df[["id", "sales"]]

submission.to_csv(
    "reports/submission.csv",
    index=False
)

print("\nFuture sales prediction completed!")

print("Total predictions:", len(predictions_df))

print("\nPrediction preview:")
print(predictions_df.head())

print("\nSaved:")
print("reports/future_predictions.csv")
print("reports/submission.csv")
