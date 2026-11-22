
import os
import json
import joblib

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from text_preprocessing import clean_text


# Create directory for saved models
os.makedirs("model_artifacts", exist_ok=True)


# 1. Load IMDb dataset
print("Downloading IMDb dataset...")

dataset = load_dataset("stanfordnlp/imdb")

X_train = dataset["train"]["text"]
y_train = dataset["train"]["label"]

X_test = dataset["test"]["text"]
y_test = dataset["test"]["label"]

print("Dataset loaded successfully!")


# 2. Preprocess reviews
print("Cleaning reviews...")

X_train = [clean_text(review) for review in X_train]
X_test = [clean_text(review) for review in X_test]


# 3. TF-IDF vectorization
print("Converting text into TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("TF-IDF conversion completed!")


# 4. Initialize algorithms
models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0
    ),

    "SVM": LinearSVC(
        C=1.0,
        max_iter=3000,
        random_state=42
    )
}


# 5. Train and evaluate
results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results[name] = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4)
    }

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)


# 6. Save vectorizer
joblib.dump(
    vectorizer,
    "model_artifacts/vectorizer.pkl"
)


# 7. Save trained models
joblib.dump(
    models,
    "model_artifacts/models.pkl"
)


# 8. Save evaluation metrics
with open("model_artifacts/metrics.json", "w") as file:
    json.dump(results, file, indent=4)


print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")

print("\nModel Performance:")

for name, scores in results.items():
    print(f"\n{name}: {scores}")

print("\nModels saved in model_artifacts/")
