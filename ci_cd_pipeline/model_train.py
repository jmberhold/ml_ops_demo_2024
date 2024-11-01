# train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from joblib import dump
import os

# Load data
data = load_iris()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save model
dump(model, 'model/model.pkl')
print("Model trained and saved to model/model.pkl")