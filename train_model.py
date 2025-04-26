import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib  # Ensure consistency (Use joblib instead of pickle)

# Load the improved dataset
df = pd.read_csv("aqua.csv")

# Define Features (X) and Targets (y)
X_aerator = df.drop(columns=["Aerator_ON"])
X_pump = df.drop(columns=["Pump_Runtime (min/hr)"])

y_aerator = df["Aerator_ON"]
y_pump_runtime = df["Pump_Runtime (min/hr)"]

# Train-Test Split
X_train_aerator, X_test_aerator, y_train_aerator, y_test_aerator = train_test_split(
    X_aerator, y_aerator, test_size=0.2, stratify=y_aerator, random_state=42
)

X_train_pump, X_test_pump, y_train_pump, y_test_pump = train_test_split(
    X_pump, y_pump_runtime, test_size=0.2, random_state=42
)

# Standardizing features (Only for Pump Model)
scaler = StandardScaler()
X_train_pump_scaled = scaler.fit_transform(X_train_pump)
X_test_pump_scaled = scaler.transform(X_test_pump)

# Train Models
rf_aerator = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf_aerator.fit(X_train_aerator, y_train_aerator)

rf_pump = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
rf_pump.fit(X_train_pump_scaled, y_train_pump)

# Evaluate Performance
y_pred_aerator = rf_aerator.predict(X_test_aerator)
y_pred_pump = rf_pump.predict(X_test_pump_scaled)

accuracy_aerator = accuracy_score(y_test_aerator, y_pred_aerator)
mse_pump = mean_squared_error(y_test_pump, y_pred_pump)
r2_pump = r2_score(y_test_pump, y_pred_pump)

print(f"✅ Aerator Model Accuracy: {accuracy_aerator:.4f}")
print(f"✅ Pump Runtime Model MSE: {mse_pump:.4f}")
print(f"✅ Pump Runtime Model R² Score: {r2_pump:.4f}")

# Save Models (Use joblib for consistency)
joblib.dump(rf_aerator, "rf_aerator_model.pkl")
joblib.dump(rf_pump, "rf_pump_model.pkl")
joblib.dump(scaler, "scaler.pkl")