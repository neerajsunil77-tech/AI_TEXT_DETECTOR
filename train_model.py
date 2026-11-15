
import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = Path("training_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError("training_data.csv not found.")

data = pd.read_csv(DATA_FILE)

required = ["text", "label", "category"]

if not all(col in data.columns for col in required):
    raise ValueError("CSV must contain text, label, category.")

data = data.dropna(subset=required)
data["text"] = data["text"].astype(str).str.strip()
data["label"] = data["label"].str.lower().str.strip()
data["category"] = data["category"].str.lower().str.strip()

data = data[
    data["label"].isin(["human", "ai"])
    & data["category"].isin(["english", "math", "algorithm"])
    & (data["text"].str.len() >= 20)
]

print("\n=== OVERALL DATASET ===")
print("Total examples:", len(data))
print("\nLabels:")
print(data["label"].value_counts())
print("\nCategories:")
print(data["category"].value_counts())

counts = data["label"].value_counts()

if len(counts) != 2 or counts.min() < 5:
    print("\nNot enough examples to train and evaluate.")
    print("Collect at least 5 examples for each label first.")
    raise SystemExit

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.25,
    random_state=42,
    stratify=data["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n=== TEST RESULTS ===")
print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions, zero_division=0))

model_path = MODEL_DIR / "overall_model.joblib"
joblib.dump(model, model_path)

print("\nOverall model saved to:", model_path)
print("Training completed.")
