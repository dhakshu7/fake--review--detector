import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("reviews.csv")

# Features & target
X = df["review"]
y = df["label"]

# Convert text → numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained successfully!")