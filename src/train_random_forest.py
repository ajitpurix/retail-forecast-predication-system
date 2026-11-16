
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load training and validation datasets
train_df = pd.read_csv("data/processed/train_split.csv")
val_df = pd.read_csv("data/processed/validation_split.csv")

# 2. Separate features and target
target = "sales"

drop_columns = ["id", "date", "sales"]

feature_columns = [
    col for col in train_df.columns
    if col not in drop_columns
]

X_train = train_df[feature_columns]
y_train = train_df[target]

X_val = val_df[feature_columns]
y_val = val_df[target]

# 3. Identify numerical and categorical columns
numeric_columns = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# 4. Handle missing numerical values
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# 5. Handle categorical values
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 6. Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 7. Create Random Forest pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=10,
        max_features=0.7,
        n_jobs=-1,
        random_state=42
    ))
])

# 8. Train model
print("Training Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")

# 9. Predict validation sales
predictions = model.predict(X_val)

# 10. Evaluate model
mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\nRandom Forest Results")
print("---------------------")
print("MAE:", mae)
print("RMSE:", rmse)

# 11. Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/random_forest.pkl"
)

# 12. Save predictions
os.makedirs("reports", exist_ok=True)

results = pd.DataFrame({
    "actual_sales": y_val,
    "predicted_sales": predictions
})

results.to_csv(
    "reports/random_forest_predictions.csv",
    index=False
)

print("\nModel saved successfully!")
