
import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Dataset files
DATA_FILES = [
    Path("hc3_balanced.csv"),
    Path("training_data.csv"),
    Path("additional_training_data.csv")
]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load available datasets
frames = []

for file in DATA_FILES:
    if file.exists():
        print("Loading:", file)
        frames.append(pd.read_csv(file))
    else:
        print("Skipping missing file:", file)

if not frames:
    raise FileNotFoundError("No training CSV files found.")

data = pd.concat(frames, ignore_index=True)

required = ["text", "label", "category"]

if not all(col in data.columns for col in required):
    raise ValueError("CSV must contain text, label, category.")

# Clean data
data = data.dropna(subset=required)

data["text"] = data["text"].astype(str).str.strip()
data["label"] = data["label"].astype(str).str.lower().str.strip()
data["category"] = data["category"].astype(str).str.lower().str.strip()

data = data[
    data["label"].isin(["human", "ai"])
    & data["category"].isin(["english", "math", "algorithm"])
    & (data["text"].str.len() >= 20)
]

# Remove exact duplicate text
data = data.drop_duplicates(subset=["text"])

print("\n=== OVERALL DATASET ===")
print("Total examples:", len(data))
print("\nLabels:")
print(data["label"].value_counts())
print("\nCategories:")
print(data["category"].value_counts())

counts = data["label"].value_counts()

if len(counts) != 2 or counts.min() < 5:
    raise ValueError("Not enough examples for both labels.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)

# Build model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        min_df=2
    )),
    ("classifier", LogisticRegression(
        max_iter=1500,
        class_weight="balanced"
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\n=== TEST RESULTS ===")
print("Test accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=["human", "ai"]))

# Save model
model_path = MODEL_DIR / "overall_model.joblib"
joblib.dump(model, model_path)

print("\nOverall model saved to:", model_path)
print("Training completed!")
