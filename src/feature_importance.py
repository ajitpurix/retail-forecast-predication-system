
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# 1. Load trained XGBoost pipeline
model = joblib.load("models/xgboost_model.pkl")

# 2. Extract preprocessing pipeline
preprocessor = model.named_steps["preprocessor"]

# 3. Extract XGBoost estimator
xgb_model = model.named_steps["regressor"]

# 4. Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# 5. Get feature importance using gain
importance = xgb_model.get_booster().get_score(
    importance_type="gain"
)

# 6. Convert importance into DataFrame
importance_df = pd.DataFrame(
    importance.items(),
    columns=["Feature", "Importance"]
)

# 7. Map XGBoost feature indices to actual names
importance_df["Feature"] = importance_df["Feature"].apply(
    lambda x: feature_names[int(x[1:])]
)

# 8. Sort features by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# 9. Select top 15 features
top_features = importance_df.head(15)

# 10. Print results
print("\nTOP 15 IMPORTANT FEATURES")
print(top_features.to_string(index=False))

# 11. Create reports folder
os.makedirs("reports", exist_ok=True)

# 12. Save feature importance
importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False
)

# 13. Plot feature importance
plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance (Gain)")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("reports/feature_importance.png")

plt.show()

print("\nFeature importance analysis completed!")
