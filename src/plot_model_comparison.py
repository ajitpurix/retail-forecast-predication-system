
import pandas as pd
import matplotlib.pyplot as plt

# Load comparison results
df = pd.read_csv("reports/model_comparison.csv")

# Plot MAE
plt.figure(figsize=(9, 5))

plt.bar(df["Model"], df["MAE"])

plt.title("Machine Learning Model Comparison")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (MAE)")

plt.tight_layout()

plt.savefig("reports/model_comparison.png")

plt.show()
