import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
import xgboost as xgb

def evaluate_predictions(y_true, y_pred, name="Model"):
    # Ensure non-negative predictions for sales
    y_pred = np.clip(y_pred, 0, None)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    
    print(f"\n--- {name} Evaluation ---")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"RMSLE: {rmsle:.4f}")
    
    return {"MAE": mae, "RMSE": rmse, "RMSLE": rmsle}

def main():
    print("Loading split datasets...")
    train_df = pd.read_csv("data/processed/train_split.csv")
    val_df = pd.read_csv("data/processed/validation_split.csv")

    # Drop rows in train_df where lag/rolling features are NaN due to initial window
    print("Handling missing values from lag/rolling features...")
    train_df = train_df.dropna().reset_index(drop=True)
    val_df = val_df.fillna(0).reset_index(drop=True)

    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")

    target = "sales"
    drop_cols = ["id", "date", target]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    print("\nFeatures used for training:", feature_cols)

    # Encode categorical 'family' column
    print("\nEncoding categorical features...")
    encoder = LabelEncoder()
    train_df["family"] = encoder.fit_transform(train_df["family"].astype(str))
    val_df["family"] = encoder.transform(val_df["family"].astype(str))

    X_train = train_df[feature_cols]
    y_train = train_df[target]

    X_val = val_df[feature_cols]
    y_val = val_df[target]

    # Convert target to log1p to optimize for RMSLE
    y_train_log = np.log1p(y_train)

    # Train XGBoost Model
    print("\nTraining XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, np.log1p(y_val))],
        verbose=50
    )

    # Predictions
    print("\nEvaluating model on validation data...")
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    
    metrics = evaluate_predictions(y_val, val_preds, name="XGBoost Regressor")

    # Save trained model and artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    model_path = "models/xgboost_model.pkl"
    encoder_path = "models/family_encoder.pkl"
    features_path = "models/feature_columns.pkl"

    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(feature_cols, features_path)

    # Save metrics report
    report_path = "reports/evaluation_metrics.txt"
    with open(report_path, "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n")
        f.write(f"Model: XGBoost Regressor\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"\nArtifacts successfully saved:")
    print(f" - Model: {model_path}")
    print(f" - Encoder: {encoder_path}")
    print(f" - Features: {features_path}")
    print(f" - Report: {report_path}")

if __name__ == "__main__":
    main()

