"""
train_model.py
==============
Generates the aviation flight dataset, trains an ensemble of machine-learning
classifiers (QDA, Random Forest, Gradient Boosting and Logistic Regression
combined with a soft-voting ensemble) and persists everything the Flask
application needs into ``qda_model.pkl``.

The pipeline covers three modelling goals:
  1. Risk classification  -> Low / Medium / High risk
  2. Accident detection   -> probability + indicative signatures
  3. Contribution analysis-> per-feature share of the high-risk score

Outputs produced by running this script:
  * dataset/aviation.csv            - labelled flight records (training data)
  * qda_model.pkl                   - pickled dict: models, scaler, metadata
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
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
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
    "runway_length",
    "airspeed",
    "fuel_level",
    "flight_duration",
    "turbulence",
    "visibility",
    "temperature",
    "dew_point",
    "humidity",
    "precipitation",
    "wind_speed",
    "wind_gust",
    "crosswind",
    "air_pressure",
    "night_flight",
    # --- In-flight / motion parameters (aircraft moving) -----------------
    "ground_speed",
    "vertical_speed",
    "wind_shear",
    "stall_margin",
    "g_load",
    "heading_change",
    "ice_accumulation",
    "maintenance_score",
    "pilot_hours",
    "vibration_level",
]

FEATURE_LABELS = {
    "aircraft_age": "Aircraft Age (years)",
    "engine_health": "Engine Health (%)",
    "altitude": "Altitude (ft)",
    "runway_length": "Runway Length (m)",
    "airspeed": "Airspeed (km/h)",
    "fuel_level": "Fuel Level (%)",
    "flight_duration": "Flight Duration (hours)",
    "turbulence": "Turbulence Level (0-10)",
    "visibility": "Visibility (km)",
    "temperature": "Temperature (degC)",
    "dew_point": "Dew Point (degC)",
    "humidity": "Humidity (%)",
    "precipitation": "Precipitation (mm/h)",
    "wind_speed": "Wind Speed (km/h)",
    "wind_gust": "Wind Gust (km/h)",
    "crosswind": "Crosswind (km/h)",
    "air_pressure": "Air Pressure (hPa)",
    "night_flight": "Night Flight (0/1)",
    "ground_speed": "Ground Speed (km/h)",
    "vertical_speed": "Vertical Speed (ft/min)",
    "wind_shear": "Wind Shear (km/h)",
    "stall_margin": "Stall Margin (km/h)",
    "g_load": "Vertical Acceleration (g)",
    "heading_change": "Heading Change (deg/min)",
    "ice_accumulation": "Ice Accumulation (0-1)",
    "maintenance_score": "Maintenance Score (%)",
    "pilot_hours": "Pilot Experience (log hours)",
    "vibration_level": "Vibration Level (0-10)",
}

CLASSES = ["Low Risk", "Medium Risk", "High Risk"]
RNG = np.random.default_rng(42)
N_SAMPLES = 8000

# Lower / upper bounds used to simulate realistic flight envelopes
BOUNDS = {
    "aircraft_age": (1.0, 40.0),
    "engine_health": (40.0, 100.0),
    "altitude": (0.0, 40000.0),
    "runway_length": (800.0, 4000.0),
    "airspeed": (200.0, 1000.0),
    "fuel_level": (5.0, 100.0),
    "flight_duration": (0.5, 15.0),
    "turbulence": (0.0, 10.0),
    "visibility": (0.0, 15.0),
    "temperature": (-30.0, 50.0),
    "dew_point": (-40.0, 30.0),
    "humidity": (10.0, 100.0),
    "precipitation": (0.0, 50.0),
    "wind_speed": (0.0, 80.0),
    "wind_gust": (0.0, 100.0),
    "crosswind": (0.0, 40.0),
    "air_pressure": (950.0, 1050.0),
    "night_flight": (0.0, 1.0),
    "ground_speed": (200.0, 1100.0),
    "vertical_speed": (-3000.0, 3000.0),
    "wind_shear": (0.0, 60.0),
    "stall_margin": (0.0, 120.0),
    "g_load": (0.5, 3.0),
    "heading_change": (0.0, 40.0),
    "ice_accumulation": (0.0, 1.0),
    "maintenance_score": (40.0, 100.0),
    "pilot_hours": (100.0, 20000.0),
    "vibration_level": (0.0, 10.0),
}


def generate_dataset(n: int) -> pd.DataFrame:
    """Create a synthetic aviation flight dataset with a realistic risk signal."""
    data: dict[str, np.ndarray] = {}
    for feat in FEATURES:
        lo, hi = BOUNDS[feat]
        if feat == "night_flight":
            data[feat] = RNG.integers(0, 2, n).astype(float)  # 0 = day, 1 = night
        elif feat == "g_load":
            # Vertical acceleration is usually near 1g with occasional excursions
            data[feat] = np.clip(RNG.normal(1.0, 0.22, n), lo, hi)
        else:
            data[feat] = RNG.uniform(lo, hi, n)

    # Normalise every factor to roughly [0, 1] so weights are interpretable
    def norm(feat: str) -> np.ndarray:
        lo, hi = BOUNDS[feat]
        return (data[feat] - lo) / (hi - lo)

    # Temperature - dew point spread (small spread -> fog / low visibility risk)
    dew_spread = np.abs(data["temperature"] - data["dew_point"])
    dew_spread_risk = 1.0 - np.clip(dew_spread / 90.0, 0.0, 1.0)

    # Ice accumulation: airframe icing is most likely in the icing band
    # (-12C..2C) combined with precipitation and moist air.
    icing_band = (
        (data["temperature"] >= -12.0)
        & (data["temperature"] <= 2.0)
        & (data["precipitation"] > 0.3)
        & (data["humidity"] > 55.0)
    )
    data["ice_accumulation"] = np.where(
        icing_band,
        np.clip(0.45 + 0.6 * RNG.random(n), 0.0, 1.0),
        data["ice_accumulation"],
    )

    # Vibration: degraded engines vibrate more
    data["vibration_level"] = np.clip(
        (1.0 - norm("engine_health")) * 6.0 + RNG.normal(0.8, 0.7, n), 0.0, 10.0
    )

    # Pilot experience risk uses a log scale (100h trainee vs 20000h veteran)
    log_hours = np.log(data["pilot_hours"])
    log_low, log_high = np.log(BOUNDS["pilot_hours"])
    inexperience = 1.0 - np.clip((log_hours - log_low) / (log_high - log_low), 0.0, 1.0)

    # Pilot / crew alertness is lower for very long night flights
    fatigue = (norm("flight_duration") * 0.6 + norm("night_flight") * 0.4)

    score = (
        0.09 * norm("aircraft_age")
        + 0.15 * (1.0 - norm("engine_health"))
        + 0.06 * norm("altitude")
        + 0.04 * (1.0 - norm("runway_length"))
        + 0.05 * norm("airspeed")
        + 0.08 * (1.0 - norm("fuel_level"))
        + 0.06 * norm("flight_duration")
        + 0.12 * norm("turbulence")
        + 0.12 * (1.0 - norm("visibility"))
        + 0.04 * np.abs(data["temperature"]) / 40.0
        + 0.04 * dew_spread_risk
        + 0.03 * norm("humidity")
        + 0.04 * norm("precipitation")
        + 0.11 * norm("wind_speed")
        + 0.05 * norm("wind_gust")
        + 0.05 * norm("crosswind")
        + 0.04 * np.abs(data["air_pressure"] - 1013.25) / 40.0
        + 0.04 * norm("night_flight")
        # --- In-flight / motion contributors --------------------------------
        + 0.03 * norm("ground_speed")
        + 0.06 * np.abs(data["vertical_speed"]) / 3000.0  # climb & sink risk
        + 0.09 * norm("wind_shear")
        + 0.08 * (1.0 - norm("stall_margin"))
        + 0.06 * np.clip(np.abs(data["g_load"] - 1.0) / 1.5, 0.0, 1.0)
        + 0.05 * norm("heading_change")
        + 0.08 * norm("ice_accumulation")
        + 0.08 * (1.0 - norm("maintenance_score"))
        + 0.06 * inexperience
        + 0.05 * norm("vibration_level")
        + 0.03 * fatigue
        + RNG.normal(0.0, 0.06, n)  # measurement / modelling noise
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

    # ------------------------------------------------------------------
    # Train the individual classifiers
    # ------------------------------------------------------------------
    print(">> Training individual classifiers ...")
    qda = QuadraticDiscriminantAnalysis(reg_param=0.05)  # small regularisation
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=220,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.9,
        random_state=42,
    )
    lr = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)

    models: dict[str, object] = {
        "Quadratic Discriminant Analysis": qda,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Logistic Regression": lr,
    }

    for name, model in models.items():
        model.fit(X_train_s, y_train)

    # Soft-voting ensemble combines the probability outputs of every model
    print(">> Building soft-voting ensemble ...")
    ensemble = VotingClassifier(
        estimators=[
            ("qda", qda),
            ("random_forest", rf),
            ("gradient_boosting", gb),
            ("logistic_regression", lr),
        ],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train_s, y_train)

    # ------------------------------------------------------------------
    # Evaluate every model + the ensemble
    # ------------------------------------------------------------------
    model_accuracies: dict[str, float] = {}
    model_reports: dict[str, str] = {}
    confusion_matrices: dict[str, list] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_s)
        model_accuracies[name] = float(accuracy_score(y_test, y_pred))
        model_reports[name] = classification_report(y_test, y_pred, target_names=CLASSES)
        confusion_matrices[name] = confusion_matrix(y_test, y_pred).tolist()
        print(f"   {name:33s} test accuracy = {model_accuracies[name]:.4f}")

    y_pred = ensemble.predict(X_test_s)
    acc = float(accuracy_score(y_test, y_pred))
    model_accuracies["Ensemble (Soft Voting)"] = acc
    model_reports["Ensemble (Soft Voting)"] = classification_report(
        y_test, y_pred, target_names=CLASSES
    )
    print(f"   {'Ensemble (Soft Voting)':33s} test accuracy = {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # Confusion matrix + heatmap for the ensemble (used on the dashboard)
    cm = confusion_matrix(y_test, y_pred)
    build_confusion_image(cm, CONFUSION_IMG_PATH)
    print(f"   saved  -> {CONFUSION_IMG_PATH}")

    # Global feature importance = average of tree-based importances
    rf_imp = rf.feature_importances_
    gb_imp = gb.feature_importances_
    imp = (rf_imp + gb_imp) / 2.0
    imp = imp / imp.sum() * 100.0  # normalise to percentages
    feature_importances = {
        f: round(float(v), 2) for f, v in zip(FEATURES, imp)
    }

    # Class means in the SCALED feature space (used for feature contributions)
    class_means = np.zeros((len(CLASSES), len(FEATURES)))
    for c in range(len(CLASSES)):
        class_means[c] = np.mean(X_train_s[y_train == c], axis=0)

    class_counts = df["risk_label"].value_counts().reindex(CLASSES, fill_value=0)

    model_bundle = {
        "model": ensemble,
        "ensemble": ensemble,
        "models": models,
        "model_order": list(models.keys()) + ["Ensemble (Soft Voting)"],
        "model_accuracies": {
            k: round(float(v) * 100, 2) for k, v in model_accuracies.items()
        },
        "model_reports": model_reports,
        "confusion_matrices": confusion_matrices,
        "scaler": scaler,
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "feature_importances": feature_importances,
        "classes": CLASSES,
        "class_means": class_means,
        "feature_std": np.std(X_train_s, axis=0),
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "dataset_stats": {
            "total_samples": int(len(df)),
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "class_counts": class_counts.to_dict(),
            "class_percent": (class_counts / len(df) * 100).round(2).to_dict(),
            "feature_ranges": {f: list(BOUNDS[f]) for f in FEATURES},
            "model": "QDA + Random Forest + Gradient Boosting + Logistic Regression (soft-voting ensemble)",
        },
    }

    # compress=True shrinks the ensemble pickle dramatically (gzip), keeping the
    # model count/size small enough for GitHub and Render's free-tier storage.
    joblib.dump(model_bundle, MODEL_PATH, compress=3)
    print(f">> Model bundle saved -> {MODEL_PATH}")
    print("\nClass distribution:")
    print(class_counts.to_string())
    print("\nPer-model accuracy comparison:")
    for name, v in model_accuracies.items():
        print(f"   {name:33s} {v * 100:.2f}%")


if __name__ == "__main__":
    main()
