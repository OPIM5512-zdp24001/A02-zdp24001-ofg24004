# ping_pong_model.py
# California Housing MLPRegressor Model

import os
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Features: {data.feature_names}")
print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# Create figures directory
os.makedirs("figures", exist_ok=True)

# Train MLPRegressor with early stopping and custom hyperparameters
model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    early_stopping=True,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Train R2: {train_r2:.4f}")
print(f"Test R2: {test_r2:.4f}")
