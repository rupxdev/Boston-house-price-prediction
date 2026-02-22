# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("HousingData.csv")

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Data Preprocessing
print("\n--- Data Preprocessing ---")

# Checking for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values
data['CRIM'].fillna(data['CRIM'].mean(), inplace=True)
data['ZN'].fillna(data['ZN'].mean(), inplace=True)
data['INDUS'].fillna(data['INDUS'].mean(), inplace=True)
data['CHAS'].fillna(data['CHAS'].mode()[0], inplace=True)
data['AGE'].fillna(data['AGE'].mean(), inplace=True)

# Confirm missing values are handled
print("\nMissing Values After Handling:")
print(data.isnull().sum())

# Splitting data into features and target
X = data.drop(columns=["MEDV"])  # Features
y = data["MEDV"]  # Target variable (house price)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData Split:")
print(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")

# Model Training and Evaluation

print("\n--- Model Training and Evaluation ---")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Results:")
print(f"Mean Squared Error: {mse_lr:.2f}")
print(f"R^2 Score: {r2_lr:.2f}")

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Regressor Results:")
print(f"Mean Squared Error: {mse_dt:.2f}")
print(f"R^2 Score: {r2_dt:.2f}")

# Gradient Boosting Regressor
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("\nGradient Boosting Regressor Results:")
print(f"Mean Squared Error: {mse_gb:.2f}")
print(f"R^2 Score: {r2_gb:.2f}")

# Comparing Models
print("\n--- Model Comparison ---")
results = {
    "Model": ["Linear Regression", "Decision Tree", "Gradient Boosting"],
    "MSE": [mse_lr, mse_dt, mse_gb],
    "R^2 Score": [r2_lr, r2_dt, r2_gb],
}
results_df = pd.DataFrame(results)
print(results_df)

# Visualizing the Results
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="R^2 Score", data=results_df)
plt.title("Model Comparison: R^2 Score")
plt.show()

# Best Model
best_model = results_df.loc[results_df['R^2 Score'].idxmax()]
print("\nBest Model:")
print(best_model)
