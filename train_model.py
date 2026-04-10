import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create folder
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("E:\machine learning all files\data preprocessing\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocess
df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']

# Train
model = RandomForestClassifier()
model.fit(X, y)

# Save
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ DONE")
