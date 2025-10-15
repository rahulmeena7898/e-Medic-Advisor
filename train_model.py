import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# === Load the dataset ===
# Make sure your Training.csv is inside the 'datasets' folder
df = pd.read_csv("datasets/Training.csv")

# Separate features (symptoms) and target (disease)
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# === Split into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train the SVM model ===
svc = SVC(kernel="linear", probability=True)
svc.fit(X_train, y_train)

# === Evaluate accuracy ===
y_pred = svc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

# === Save the model ===
os.makedirs("models", exist_ok=True)
with open("models/svc.pkl", "wb") as f:
    pickle.dump(svc, f)

print("💾 Model saved successfully at: models/svc.pkl")
