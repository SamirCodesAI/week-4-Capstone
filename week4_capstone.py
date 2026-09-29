import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_excel("week4_data.xlsx")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Basic EDA
print("\nDataset Shape:", df.shape)
print("\nSummary Statistics:")
print(df.describe())

print("\nSales by Product:")
print(df.groupby("Product")["Sales"].sum().sort_values(ascending=False))

print("\nProfit by Product:")
print(df.groupby("Product")["Profit"].sum().sort_values(ascending=False))

print("\nSales by Year:")
print(df.groupby("Year")["Sales"].sum())

# Create charts
df.groupby("Product")["Sales"].sum().plot(kind="bar")
plt.title("Sales by Product")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("sales_by_product.png")
plt.show()

df.groupby("Year")["Sales"].sum().plot(kind="bar")
plt.title("Sales by Year")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("sales_by_year.png")
plt.show()

# Regression Model
X = df[["Units Sold", "Sale Price", "Gross Sales", "Discounts", "COGS"]]
y = df["Sales"]

# Remove rows with missing values
data = pd.concat([X, y], axis=1).dropna()

X = data[["Units Sold", "Sale Price", "Gross Sales", "Discounts", "COGS"]]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nRegression Results:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Save predictions
results = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": y_pred
})

results.to_csv("sales_predictions.csv", index=False)

# Save cleaned dataset
df.to_csv("cleaned_week4_data.csv", index=False)

print("\nWeek 4 Python analysis completed!")