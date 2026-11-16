
import pandas as pd
import matplotlib.pyplot as plt

# Load training dataset
df = pd.read_csv("data/raw/train.csv")

# Convert date column into datetime format
df["date"] = pd.to_datetime(df["date"])

# 1. Top 10 product families by total sales
family_sales = (
    df.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
family_sales.plot(kind="bar")
plt.title("Top 10 Product Families by Sales")
plt.xlabel("Product Family")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/top_product_families.png")
plt.show()


# 2. Top 10 stores by total sales
store_sales = (
    df.groupby("store_nbr")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))
store_sales.plot(kind="bar")
plt.title("Top 10 Stores by Sales")
plt.xlabel("Store Number")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("reports/top_stores.png")
plt.show()


# 3. Daily sales trend
daily_sales = df.groupby("date")["sales"].sum()

plt.figure(figsize=(14, 5))
daily_sales.plot()
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("reports/daily_sales_trend.png")
plt.show()


# 4. Average sales by weekday
df["weekday"] = df["date"].dt.day_name()

weekday_sales = (
    df.groupby("weekday")["sales"]
    .mean()
    .reindex([
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ])
)

plt.figure(figsize=(10, 5))
weekday_sales.plot(kind="bar")
plt.title("Average Sales by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig("reports/weekday_sales.png")
plt.show()

print("EDA completed successfully!")
print("\nTop product families:")
print(family_sales)

print("\nTop stores:")
print(store_sales)

print("\nAverage sales by weekday:")
print(weekday_sales)
