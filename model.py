import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load Dataset
data = pd.read_csv("Spectrum-Disorder.csv")

# Define Feature Columns
feature_cols = ["A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
                "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
                "age", "gender", "jaundice", "autism"]  # Use correct column names

# Select Features and Target
X = data[feature_cols]
y = data["Class"]  # Target Variable

# Convert Categorical Data
def convert_binary(value):
    return 1 if value == "Yes" else 0

data["gender"] = data["gender"].map({"Male": 1, "Female": 0})
data["jaundice"] = data["jaundice"].map(convert_binary)
data["autism"] = data["autism"].map(convert_binary)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "autism_model.pkl")

# Debugging: Print Sample Predictions for Manual Verification
sample_inputs = X_test[:5]
sample_predictions = model.predict(sample_inputs)

print("Sample Inputs:")
print(sample_inputs)
print("Predictions:")
print(sample_predictions)
