from flask import Flask, render_template, request, jsonify
from src.model import load_artifacts, predict_email
app = Flask(__name__)
model, metrics = load_artifacts()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/metrics")
def api_metrics():
    return jsonify(metrics)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (str(data.get("subject", "")) + "\\n" + str(data.get("body", ""))).strip()
    if not text:
        return jsonify({"error": "Enter an email subject or body."}), 400
    if model is None:
        return jsonify({"error": "Model not trained. Run python train.py first."}), 503
    return jsonify(predict_email(model, text))

if __name__ == "__main__":
    app.run(debug=True)
