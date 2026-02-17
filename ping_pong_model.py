# ping_pong_model.py
# PR #1: Load California Housing dataset and split into train/test

import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

# Create figures directory for later
os.makedirs("figures", exist_ok=True)
