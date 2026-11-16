
import pandas as pd
import numpy as np
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# 1. Define prediction file locations
prediction_files = {
    "Linear Regression":
        "reports/linear_regression_predictions.csv",

    "Random Forest":
        "reports/random_forest_predictions.csv",

    "XGBoost":
        "reports/xgboost_predictions.csv"
}

results = []

# 2. Evaluate each model
for model_name, file_path in prediction_files.items():

    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        continue

    df = pd.read_csv(file_path)

    actual = df["actual_sales"]
    predicted = df["predicted_sales"]

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse
    })

# 3. Create comparison table
comparison = pd.DataFrame(results)

if comparison.empty:
    print("No prediction files found.")
else:
    comparison = comparison.sort_values("MAE")

    print("\nMODEL PERFORMANCE COMPARISON")
    print("============================")
    print(comparison.to_string(index=False))

    # 4. Save comparison
    os.makedirs("reports", exist_ok=True)

    comparison.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    # 5. Display best preliminary model
    best_model = comparison.iloc[0]

    print("\nBest preliminary model:", best_model["Model"])
    print("Best MAE:", best_model["MAE"])
    print("Best RMSE:", best_model["RMSE"])

    print("\nComparison saved successfully!")
