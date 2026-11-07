from pathlib import Path
import json
import joblib

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "best_email_classifier.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"


def load_artifacts():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return None, {
            "error": "Model not trained yet. Run python train.py.",
            "dataset_size": 0
        }

    model = joblib.load(MODEL_PATH)
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)
    return model, metrics


def predict_email(model, text):
    prediction = str(model.predict([text])[0])
    confidence = None

    # Both MultinomialNB and calibrated Linear SVM expose predict_proba.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = round(float(max(probabilities)) * 100, 2)

    return {
        "prediction": prediction,
        "confidence": confidence
    }
