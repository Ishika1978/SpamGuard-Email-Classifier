from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "email_dataset.csv"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)


def make_vectorizer():
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True
    )


def main():
    if not DATA.exists():
        raise SystemExit("Dataset CSV missing. Run python prepare_dataset.py first.")

    df = pd.read_csv(DATA).dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].astype(str)
    df = df[df["label"].isin(["ham", "spam"])]

    if df["label"].nunique() != 2:
        raise SystemExit("Dataset must contain both ham and spam.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    nb_pipeline = Pipeline([
        ("tfidf", make_vectorizer()),
        ("classifier", MultinomialNB(alpha=0.5))
    ])

    svm_pipeline = Pipeline([
        ("tfidf", make_vectorizer()),
        ("classifier", LinearSVC(class_weight="balanced"))
    ])

    # Calibrate the SVM so the website can display probability-based confidence.
    calibrated_svm = CalibratedClassifierCV(
        estimator=svm_pipeline,
        method="sigmoid",
        cv=3
    )

    candidates = {
        "Multinomial Naive Bayes": nb_pipeline,
        "Calibrated Linear SVM": calibrated_svm
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1.0

    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        spam_f1 = f1_score(
            y_test, predictions, pos_label="spam", zero_division=0
        )
        metrics = {
            "accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
            "spam_precision": round(
                precision_score(y_test, predictions, pos_label="spam", zero_division=0) * 100, 2
            ),
            "spam_recall": round(
                recall_score(y_test, predictions, pos_label="spam", zero_division=0) * 100, 2
            ),
            "spam_f1": round(spam_f1 * 100, 2),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=["ham", "spam"]
            ).tolist(),
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )
        }
        results[name] = metrics
        print(json.dumps(metrics, indent=2))

        if spam_f1 > best_f1:
            best_name, best_model, best_f1 = name, model, spam_f1

    joblib.dump(best_model, MODELS / "best_email_classifier.joblib")

    output = {
        "dataset_size": int(len(df)),
        "ham_count": int((df["label"] == "ham").sum()),
        "spam_count": int((df["label"] == "spam").sum()),
        "test_size": int(len(y_test)),
        "best_model": best_name,
        "models": results,
        "selected_metrics": results[best_name]
    }

    with (MODELS / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nTraining complete. Best model: {best_name}")
    print("Saved model and metrics in the models/ folder.")


if __name__ == "__main__":
    main()
