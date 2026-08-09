"""
Aviation Accident Prediction System using Quadratic Discriminant Analysis (QDA)
===============================================================================
Flask backend application.

Routes
------
GET  /            -> Home page
GET  /about       -> About / methodology page
GET  /prediction  -> Input form page
POST /predict     -> Runs the QDA model and renders the result page
GET  /dashboard   -> Analytics dashboard
GET  /contact     -> Project team / contact page
"""

import os
from typing import Any, Dict

import joblib
import numpy as np
from flask import Flask, render_template, request, session

# ------------------------------------------------------------------------------
# Application configuration
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "qda_model.pkl")

app = Flask(__name__)
# Secret key is only used for the session that stores the last prediction so
# the dashboard can visualise it. Override via the SECRET_KEY env var in prod.
app.secret_key = os.environ.get("SECRET_KEY", "aviation-qda-secret-key-change-me")

# ------------------------------------------------------------------------------
# Load the trained QDA model bundle once at startup
# ------------------------------------------------------------------------------
MODEL_BUNDLE: Dict[str, Any] = joblib.load(MODEL_PATH)
MODEL = MODEL_BUNDLE["model"]          # QuadraticDiscriminantAnalysis
SCALER = MODEL_BUNDLE["scaler"]        # StandardScaler fitted on training data
FEATURES = MODEL_BUNDLE["features"]    # order the model expects
FEATURE_LABELS = MODEL_BUNDLE["feature_labels"]
CLASSES = MODEL_BUNDLE["classes"]      # ["Low Risk", "Medium Risk", "High Risk"]
CLASS_MEANS = np.asarray(MODEL_BUNDLE["class_means"])  # scaled class centroids
FEATURE_STD = np.asarray(MODEL_BUNDLE["feature_std"])
ACCURACY = MODEL_BUNDLE["accuracy"]
CONFUSION = MODEL_BUNDLE["confusion_matrix"]
DATASET_STATS = MODEL_BUNDLE["dataset_stats"]

RISK_META = {
    "Low Risk": {
        "code": "low",
        "icon": "fa-circle-check",
        "summary": "Flight conditions are safe. Continue with standard operating procedures.",
        "recommendations": [
            "Proceed with the scheduled flight plan.",
            "Maintain standard communication with Air Traffic Control.",
            "Continue routine engine and system monitoring.",
            "Keep a close watch on any mid-flight weather changes.",
        ],
    },
    "Medium Risk": {
        "code": "medium",
        "icon": "fa-circle-exclamation",
        "summary": "Conditions are manageable but require increased vigilance.",
        "recommendations": [
            "Monitor weather conditions carefully throughout the flight.",
            "Reduce airspeed if turbulence or wind shear is encountered.",
            "Maintain extra separation and stay in touch with ATC.",
            "Review fuel reserves and identify alternate airports.",
            "Brief the crew on updated weather and contingency plans.",
        ],
    },
    "High Risk": {
        "code": "high",
        "icon": "fa-triangle-exclamation",
        "summary": "Potential accident detected. Immediate action is required.",
        "recommendations": [
            "Delay the flight until conditions improve.",
            "Notify Air Traffic Control immediately.",
            "Perform a complete pre-flight aircraft inspection.",
            "Re-check engine health, fuel and navigation systems.",
            "Consult the safety officer before take-off approval.",
        ],
    },
}

# Feature value ranges accepted by the form (matches the training envelope)
FEATURE_RANGES = DATASET_STATS["feature_ranges"]


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def compute_feature_contributions(values: Dict[str, float]) -> Dict[str, float]:
    """
    Estimate how strongly each feature pushes the flight towards the
    HIGH-RISK class. Works in the scaled feature space by measuring the
    signed deviation from the Low-Risk centroid in the direction of the
    High-Risk centroid. Returns a normalised percentage per feature.
    """
    low_centroid = CLASS_MEANS[0]
    high_centroid = CLASS_MEANS[-1]

    x = np.array([values[f] for f in FEATURES]).reshape(1, -1)
    x_scaled = SCALER.transform(x)[0]

    direction = np.sign(high_centroid - low_centroid)
    raw = (x_scaled - low_centroid) * direction
    raw = raw / (FEATURE_STD + 1e-9)

    total = float(np.abs(raw).sum())
    if total < 1e-9:
        return {f: 0.0 for f in FEATURES}
    return {f: float(abs(r) / total * 100) for f, r in zip(FEATURES, raw)}


def run_prediction(values: Dict[str, float]) -> Dict[str, Any]:
    """Scale the inputs, run the QDA model and bundle up the full result."""
    x = np.array([[values[f] for f in FEATURES]])
    x_scaled = SCALER.transform(x)

    probabilities = MODEL.predict_proba(x_scaled)[0]          # p(low), p(med), p(high)
    idx = int(np.argmax(probabilities))
    label = CLASSES[idx]
    confidence = float(probabilities[idx] * 100)

    probs = {cls: round(float(p * 100), 2) for cls, p in zip(CLASSES, probabilities)}
    contributions = compute_feature_contributions(values)
    top_contributors = sorted(
        contributions.items(), key=lambda kv: kv[1], reverse=True
    )[:3]

    return {
        "label": label,
        "risk_meta": RISK_META[label],
        "confidence": round(confidence, 2),
        "probabilities": probs,
        "probability_list": [
            {"class": cls, "value": round(float(p * 100), 2)}
            for cls, p in zip(CLASSES, probabilities)
        ],
        "features": values,
        "contributions": contributions,
        "top_contributors": top_contributors,
    }


# ------------------------------------------------------------------------------
# Context processor: expose model metadata to every template
# ------------------------------------------------------------------------------
@app.context_processor
def inject_globals() -> Dict[str, Any]:
    return {
        "model_name": "Quadratic Discriminant Analysis",
        "accuracy": ACCURACY,
        "classes": CLASSES,
        "feature_labels": FEATURE_LABELS,
        "dataset_stats": DATASET_STATS,
    }


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/prediction")
def prediction():
    return render_template(
        "prediction.html", feature_labels=FEATURE_LABELS, form_values={}
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts the flight parameters, validates them, runs the QDA model and
    renders the result page. Stores the prediction in the session so the
    dashboard can show it.
    """
    raw_values = {}
    errors = []

    for feat in FEATURES:
        raw = request.form.get(feat, "").strip()
        if raw == "":
            errors.append(f"{FEATURE_LABELS[feat]} is required.")
            continue
        try:
            raw_values[feat] = float(raw)
        except ValueError:
            errors.append(f"{FEATURE_LABELS[feat]} must be a number.")

    # Range validation against the training envelope
    for feat, value in raw_values.items():
        lo, hi = FEATURE_RANGES[feat]
        if not (lo <= value <= hi):
            errors.append(
                f"{FEATURE_LABELS[feat]} must be between {lo:g} and {hi:g}."
            )

    if errors:
        return (
            render_template(
                "prediction.html",
                feature_labels=FEATURE_LABELS,
                error="; ".join(errors),
                form_values=raw_values,
            ),
            400,
        )

    result = run_prediction(raw_values)

    # Persist for the dashboard (also visible after a fresh server start)
    session["last_prediction"] = result

    return render_template(
        "result.html",
        result=result,
        classes=CLASSES,
        zip=zip,
    )


@app.route("/dashboard")
def dashboard():
    last_prediction = session.get("last_prediction")
    return render_template(
        "dashboard.html",
        last_prediction=last_prediction,
        classes=CLASSES,
        zip=zip,
    )


@app.route("/contact")
def contact():
    return render_template("contact.html")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
