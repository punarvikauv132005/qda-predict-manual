"""
train_model.py
==============
Generates the aviation flight dataset, trains a Quadratic Discriminant
Analysis (QDA) classifier and persists everything the Flask application
needs into ``qda_model.pkl``.

Outputs produced by running this script:
  * dataset/aviation.csv            - labelled flight records (training data)
  * qda_model.pkl                   - pickled dict: model, scaler, metadata
  * static/images/confusion_matrix.png - confusion-matrix heatmap for the dashboard

Run:  python train_model.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMG_DIR = BASE_DIR / "static" / "images"
DATASET_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATASET_DIR / "aviation.csv"
MODEL_PATH = BASE_DIR / "qda_model.pkl"
CONFUSION_IMG_PATH = IMG_DIR / "confusion_matrix.png"

# --------------------------------------------------------------------------
# Feature specification (used both here and at prediction time)
# --------------------------------------------------------------------------
FEATURES = [
    "aircraft_age",
    "engine_health",
    "altitude",
    "airspeed",
    "fuel_level",
    "flight_duration",
    "turbulence",
    "visibility",
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "wind_gust",
    "air_pressure",
]

FEATURE_LABELS = {
    "aircraft_age": "Aircraft Age (years)",
    "engine_health": "Engine Health (%)",
    "altitude": "Altitude (ft)",
    "airspeed": "Airspeed (km/h)",
    "fuel_level": "Fuel Level (%)",
    "flight_duration": "Flight Duration (hours)",
    "turbulence": "Turbulence Level (0-10)",
    "visibility": "Visibility (km)",
    "temperature": "Temperature (degC)",
    "humidity": "Humidity (%)",
    "precipitation": "Precipitation (mm/h)",
    "wind_speed": "Wind Speed (km/h)",
    "wind_gust": "Wind Gust (km/h)",
    "air_pressure": "Air Pressure (hPa)",
}

CLASSES = ["Low Risk", "Medium Risk", "High Risk"]
RNG = np.random.default_rng(42)
N_SAMPLES = 8000

# Lower / upper bounds used to simulate realistic flight envelopes
BOUNDS = {
    "aircraft_age": (1.0, 40.0),
    "engine_health": (40.0, 100.0),
    "altitude": (0.0, 40000.0),
    "airspeed": (200.0, 1000.0),
    "fuel_level": (5.0, 100.0),
    "flight_duration": (0.5, 15.0),
    "turbulence": (0.0, 10.0),
    "visibility": (0.0, 15.0),
    "temperature": (-30.0, 50.0),
    "humidity": (10.0, 100.0),
    "precipitation": (0.0, 50.0),
    "wind_speed": (0.0, 80.0),
    "wind_gust": (0.0, 100.0),
    "air_pressure": (950.0, 1050.0),
}


def generate_dataset(n: int) -> pd.DataFrame:
    """Create a synthetic aviation flight dataset with a realistic risk signal."""
    data: dict[str, np.ndarray] = {}
    for feat in FEATURES:
        lo, hi = BOUNDS[feat]
        data[feat] = RNG.uniform(lo, hi, n)

    # Normalise every factor to roughly [0, 1] so weights are interpretable
    def norm(feat: str) -> np.ndarray:
        lo, hi = BOUNDS[feat]
        return (data[feat] - lo) / (hi - lo)

    score = (
        0.10 * norm("aircraft_age")
        + 0.16 * (1.0 - norm("engine_health"))
        + 0.07 * norm("altitude")
        + 0.05 * norm("airspeed")
        + 0.09 * (1.0 - norm("fuel_level"))
        + 0.07 * norm("flight_duration")
        + 0.13 * norm("turbulence")
        + 0.13 * (1.0 - norm("visibility"))
        + 0.04 * np.abs(data["temperature"]) / 40.0
        + 0.03 * norm("humidity")
        + 0.04 * norm("precipitation")
        + 0.12 * norm("wind_speed")
        + 0.05 * norm("wind_gust")
        + 0.04 * np.abs(data["air_pressure"] - 1013.25) / 40.0
        + RNG.normal(0.0, 0.055, n)  # measurement / modelling noise
    )

    # Class thresholds -> roughly 40% low, 30% medium, 30% high
    q40, q70 = np.quantile(score, [0.40, 0.70])
    labels = np.where(score < q40, 0, np.where(score < q70, 1, 2))

    # Inject a little label noise so the task is realistic rather than trivial
    flip = RNG.random(n) < 0.03
    labels[flip] = RNG.integers(0, 3, size=int(flip.sum()))

    df = pd.DataFrame(data)
    df["risk_code"] = labels
    df["risk_label"] = df["risk_code"].map(lambda c: CLASSES[c])
    return df


def build_confusion_image(cm: np.ndarray, path: Path) -> None:
    """Render the confusion matrix as a PNG used on the dashboard page."""
    cmap = LinearSegmentedColormap.from_list(
        "navy_sky", ["#e8f4fd", "#7fb8e8", "#0b2545"]
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=120)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted Risk",
        ylabel="True Risk",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "#0b2545",
                fontsize=12,
            )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    print(">> Generating synthetic aviation dataset ...")
    df = generate_dataset(N_SAMPLES)
    df.to_csv(DATASET_PATH, index=False)
    print(f"   saved -> {DATASET_PATH}")

    X = df[FEATURES].values
    y = df["risk_code"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisation improves the conditioning of the QDA covariance matrices
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(">> Training Quadratic Discriminant Analysis ...")
    qda = QuadraticDiscriminantAnalysis(reg_param=0.05)  # small regularisation
    qda.fit(X_train_s, y_train)

    y_pred = qda.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"   test accuracy = {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    cm = confusion_matrix(y_test, y_pred)
    build_confusion_image(cm, CONFUSION_IMG_PATH)
    print(f"   saved  -> {CONFUSION_IMG_PATH}")

    # Class means in the SCALED feature space (used for feature contributions)
    class_means = np.zeros((len(CLASSES), len(FEATURES)))
    for c in range(len(CLASSES)):
        class_means[c] = np.mean(X_train_s[y_train == c], axis=0)

    class_counts = df["risk_label"].value_counts().reindex(CLASSES, fill_value=0)

    model_bundle = {
        "model": qda,
        "scaler": scaler,
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "classes": CLASSES,
        "class_means": class_means,
        "feature_std": np.std(X_train_s, axis=0),
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "dataset_stats": {
            "total_samples": int(len(df)),
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "class_counts": class_counts.to_dict(),
            "class_percent": (class_counts / len(df) * 100).round(2).to_dict(),
            "feature_ranges": {f: list(BOUNDS[f]) for f in FEATURES},
            "model": "Quadratic Discriminant Analysis",
        },
    }

    joblib.dump(model_bundle, MODEL_PATH)
    print(f">> Model bundle saved -> {MODEL_PATH}")
    print("\nClass distribution:")
    print(class_counts.to_string())


if __name__ == "__main__":
    main()
