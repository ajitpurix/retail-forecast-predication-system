
import pandas as pd
import numpy as np
import joblib
import os

from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load datasets
train_df = pd.read_csv("data/processed/train_split.csv")
val_df = pd.read_csv("data/processed/validation_split.csv")

train_df["date"] = pd.to_datetime(train_df["date"])
val_df["date"] = pd.to_datetime(val_df["date"])

# 2. Sort data
train_df = train_df.sort_values(
    ["store_nbr", "family", "date"]
)

val_df = val_df.sort_values(
    ["date", "store_nbr", "family"]
)

# 3. Define model files
model_files = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost_model.pkl"
}

# 4. Feature columns used during training
feature_columns = [
    col for col in train_df.columns
    if col not in ["id", "date", "sales"]
]

results = []

# 5. Evaluate each model
for model_name, model_path in model_files.items():

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        continue

    print(f"\nEvaluating {model_name}...")

    model = joblib.load(model_path)

    # Store historical sales separately for every
    # store and product-family combination.
    history = defaultdict(list)

    for row in train_df.itertuples(index=False):
        key = (row.store_nbr, row.family)
        history[key].append(float(row.sales))

    actual_all = []
    predicted_all = []
    prediction_records = []

    # 6. Predict one date at a time
    for current_date, daily_df in val_df.groupby("date", sort=True):

        feature_rows = []

        for row in daily_df.itertuples(index=False):

            key = (row.store_nbr, row.family)
            past_sales = history[key]

            # Calendar features
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

                # Lag features
                "lag_1": past_sales[-1] if len(past_sales) >= 1 else np.nan,
                "lag_7": past_sales[-7] if len(past_sales) >= 7 else np.nan,
                "lag_14": past_sales[-14] if len(past_sales) >= 14 else np.nan,

                # Rolling averages
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

        # Match the exact training feature order
        X_day = pd.DataFrame(feature_rows)
        X_day = X_day.reindex(columns=feature_columns)

        # 7. Predict current date
        predictions = model.predict(X_day)

        # 8. Save predictions and update history
        for row, prediction in zip(
            daily_df.itertuples(index=False),
            predictions
        ):

            key = (row.store_nbr, row.family)

            actual = float(row.sales)
            predicted = float(prediction)

            actual_all.append(actual)
            predicted_all.append(predicted)

            prediction_records.append({
                "date": current_date,
                "store_nbr": row.store_nbr,
                "family": row.family,
                "actual_sales": actual,
                "predicted_sales": predicted
            })

            # Future features use our prediction,
            # not the actual validation sales.
            history[key].append(predicted)

    # 9. Calculate evaluation metrics
    mae = mean_absolute_error(
        actual_all,
        predicted_all
    )

    rmse = np.sqrt(
        mean_squared_error(actual_all, predicted_all)
    )

    print("MAE:", mae)
    print("RMSE:", rmse)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse
    })

    # 10. Save model predictions
    output_file = (
        "reports/"
        + model_name.lower().replace(" ", "_")
        + "_recursive_predictions.csv"
    )

    pd.DataFrame(prediction_records).to_csv(
        output_file,
        index=False
    )

# 11. Compare all recursive results
if results:

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values("MAE")

    os.makedirs("reports", exist_ok=True)

    comparison.to_csv(
        "reports/recursive_model_comparison.csv",
        index=False
    )

    print("\nRECURSIVE MODEL COMPARISON")
    print(comparison.to_string(index=False))

    print(
        "\nBest model based on recursive MAE:",
        comparison.iloc[0]["Model"]
    )

    print("\nEvaluation completed!")
