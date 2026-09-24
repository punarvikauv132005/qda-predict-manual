"""
Aviation Accident Prediction System
===================================
Flask backend application. Predicts flight accident risk using a soft-voting
ensemble of machine-learning models:

  * Quadratic Discriminant Analysis (QDA)
  * Random Forest
  * Gradient Boosting
  * Logistic Regression
  * Ensemble (soft voting of the four above)

Beyond the risk label the app also reports:
  * an accident-detection probability with indicative warning signatures
  * per-model consensus
  * ALL risk contributors (including small ones)
  * targeted, parameter-specific safety measures for higher-risk flights

Routes
------
GET  /            -> Home page
GET  /about       -> About / methodology page
GET  /prediction  -> Input form page
POST /predict     -> Runs the ensemble model and renders the result page
GET  /dashboard   -> Analytics dashboard
GET  /contact     -> Project team / contact page
"""

import os
from typing import Any, Dict, List

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
# Load the trained model bundle once at startup
# ------------------------------------------------------------------------------
MODEL_BUNDLE: Dict[str, Any] = joblib.load(MODEL_PATH)
ENSEMBLE = MODEL_BUNDLE["ensemble"]            # VotingClassifier (soft voting)
MODELS = MODEL_BUNDLE["models"]                # individual classifiers
MODEL_ORDER = MODEL_BUNDLE["model_order"]      # display order incl. ensemble
MODEL_ACCURACIES = MODEL_BUNDLE["model_accuracies"]
SCALER = MODEL_BUNDLE["scaler"]                # StandardScaler
FEATURES = MODEL_BUNDLE["features"]            # order the models expect
FEATURE_LABELS = MODEL_BUNDLE["feature_labels"]
FEATURE_IMPORTANCES = MODEL_BUNDLE["feature_importances"]
CLASSES = MODEL_BUNDLE["classes"]              # ["Low Risk", "Medium Risk", "High Risk"]
CLASS_MEANS = np.asarray(MODEL_BUNDLE["class_means"])   # scaled class centroids
FEATURE_STD = np.asarray(MODEL_BUNDLE["feature_std"])
ACCURACY = MODEL_BUNDLE["accuracy"]
CONFUSION = MODEL_BUNDLE["confusion_matrix"]
DATASET_STATS = MODEL_BUNDLE["dataset_stats"]

HIGH_IDX = CLASSES.index("High Risk")

# ==============================================================================
# Safety / risk metadata
# ==============================================================================
RISK_META = {
    "Low Risk": {
        "code": "low",
        "icon": "fa-circle-check",
        "summary": "Flight conditions are safe. Continue with standard operating procedures.",
        "recommendations": [
            "Proceed with the scheduled flight plan.",
            "Maintain standard communication with Air Traffic Control.",
            "Continue routine engine and system monitoring.",
            "Keep a watch on any mid-flight weather changes.",
        ],
    },
    "Medium Risk": {
        "code": "medium",
        "icon": "fa-circle-exclamation",
        "summary": "Conditions are manageable but require increased vigilance.",
        "recommendations": [
            "Monitor weather and flight parameters continuously.",
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

# Accident-status bands based on the computed accident probability
ACCIDENT_STATUS = {
    "safe": {
        "code": "safe",
        "title": "Low Accident Risk",
        "icon": "fa-circle-check",
        "recommendations": [
            "No immediate accident signatures detected. Maintain normal monitoring.",
            "Continue the standard checklist and remain vigilant for changing conditions.",
        ],
    },
    "watch": {
        "code": "watch",
        "title": "Elevated Accident Risk",
        "icon": "fa-circle-exclamation",
        "recommendations": [
            "Review the listed warning signatures with the full crew.",
            "Increase cross-checks of flight instruments and engine parameters.",
            "Prepare diversion options and declare the situation to ATC early.",
            "Keep approach/go-around planning updated at all times.",
        ],
    },
    "danger": {
        "code": "danger",
        "title": "Accident Likely",
        "icon": "fa-triangle-exclamation",
        "recommendations": [
            "Treat the listed signatures as an emergency - execute the relevant checklist immediately.",
            "Declare PAN-PAN / MAYDAY as appropriate and inform ATC without delay.",
            "Initiate the wind-shear / stall-recovery / engine-out procedure for the dominant signature.",
            "Divert to the nearest suitable airfield and follow the commander's authority.",
        ],
    },
}

# ==============================================================================
# Hard accident-signature triggers (parameter band -> warning signature)
# ==============================================================================
def build_accident_triggers() -> List[Dict[str, Any]]:
    return [
        {
            "feature": "stall_margin",
            "test": lambda v: v["stall_margin"] <= 20.0,
            "title": "Near-stall / dangerously low speed",
            "weight": 15,
            "detail": lambda v: f"Stall margin is only {v['stall_margin']:.0f} km/h - aerodynamic stall is imminent.",
        },
        {
            "feature": "vertical_speed",
            "test": lambda v: v["vertical_speed"] <= -2000.0,
            "title": "High sink rate",
            "weight": 13,
            "detail": lambda v: f"Descent rate of {abs(v['vertical_speed']):.0f} ft/min - risk of terrain impact or unstabilised approach.",
        },
        {
            "feature": "vertical_speed",
            "test": lambda v: v["vertical_speed"] >= 2500.0,
            "title": "Uncommanded rapid climb",
            "weight": 9,
            "detail": lambda v: f"Climb rate of {v['vertical_speed']:.0f} ft/min - possible stall on the back side of the power curve.",
        },
        {
            "feature": "wind_shear",
            "test": lambda v: v["wind_shear"] >= 40.0,
            "title": "Severe wind shear",
            "weight": 14,
            "detail": lambda v: f"Wind shear of {v['wind_shear']:.0f} km/h - airspeed and altitude can be lost in seconds.",
        },
        {
            "feature": "g_load",
            "test": lambda v: v["g_load"] >= 2.0 or v["g_load"] < 0.7,
            "title": "Extreme vertical G-loading",
            "weight": 10,
            "detail": lambda v: f"Vertical acceleration of {v['g_load']:.2f} g - structural load limits may be exceeded.",
        },
        {
            "feature": "engine_health",
            "test": lambda v: v["engine_health"] <= 55.0,
            "title": "Engine health critical",
            "weight": 12,
            "detail": lambda v: f"Engine health at {v['engine_health']:.0f}% - uncontained failure or flame-out is possible.",
        },
        {
            "feature": "fuel_level",
            "test": lambda v: v["fuel_level"] <= 15.0,
            "title": "Fuel critically low",
            "weight": 12,
            "detail": lambda v: f"Fuel at {v['fuel_level']:.0f}% - declare fuel state and divert immediately.",
        },
        {
            "feature": "ice_accumulation",
            "test": lambda v: v["ice_accumulation"] >= 0.7,
            "title": "Severe airframe / engine icing",
            "weight": 11,
            "detail": lambda v: f"Ice accumulation severity {v['ice_accumulation']:.2f}/1.00 - risk of aerodynamic stall and lost thrust.",
        },
        {
            "feature": "heading_change",
            "test": lambda v: v["heading_change"] >= 25.0,
            "title": "Abrupt heading deviations",
            "weight": 7,
            "detail": lambda v: f"Track changing at {v['heading_change']:.0f} deg/min - possible autopilot or navigation fault.",
        },
        {
            "feature": "vibration_level",
            "test": lambda v: v["vibration_level"] >= 7.0,
            "title": "Excessive vibration",
            "weight": 9,
            "detail": lambda v: f"Vibration level {v['vibration_level']:.1f}/10 - engine or airframe imbalance warning.",
        },
        {
            "feature": "visibility",
            "test": lambda v: v["visibility"] <= 1.0,
            "title": "Near-zero visibility",
            "weight": 8,
            "detail": lambda v: f"Visibility only {v['visibility']:.1f} km - visual approach below minima.",
        },
        {
            "feature": "precipitation",
            "test": lambda v: v["precipitation"] >= 30.0,
            "title": "Heavy precipitation / storm cell",
            "weight": 8,
            "detail": lambda v: f"Precipitation at {v['precipitation']:.0f} mm/h - severe weather embedded.",
        },
        {
            "feature": "turbulence",
            "test": lambda v: v["turbulence"] >= 8.0,
            "title": "Severe turbulence",
            "weight": 9,
            "detail": lambda v: f"Turbulence {v['turbulence']:.1f}/10 - possible loss of control and injury risk.",
        },
        {
            "feature": "crosswind",
            "test": lambda v: v["crosswind"] >= 35.0,
            "title": "Crosswind beyond limits",
            "weight": 8,
            "detail": lambda v: f"Crosswind {v['crosswind']:.0f} km/h - exceedance of landing crosswind limit.",
        },
        {
            "feature": "air_pressure",
            "test": lambda v: v["air_pressure"] <= 985.0,
            "title": "Very low pressure (storm system)",
            "weight": 6,
            "detail": lambda v: f"Pressure {v['air_pressure']:.0f} hPa - deeply unstable airmass, severe weather likely.",
        },
        {
            "feature": "fuel_level",
            "test": lambda v: v["fuel_level"] <= 30.0 and v["flight_duration"] >= 8.0,
            "title": "Low fuel on a long sector",
            "weight": 6,
            "detail": lambda v: f"Fuel at {v['fuel_level']:.0f}% on a {v['flight_duration']:.1f}h flight - endurance is a concern.",
        },
    ]

ACCIDENT_TRIGGERS = build_accident_triggers()

# ==============================================================================
# Specific safety measures keyed by feature (targeted, not generic)
# ==============================================================================
SAFETY_PLAYBOOK: Dict[str, Dict[str, Any]] = {
    "aircraft_age": {
        "reason": "Aging airframe and systems",
        "actions": [
            "Review airworthiness directives, corrosion and structural-fatigue log for the airframe before dispatch.",
            "Cabin crew: report any pressurisation anomalies or unusual sounds in the climb.",
            "Engineers: verify life-limited components (landing gear, pylon attachments) are within compliance.",
        ],
    },
    "engine_health": {
        "reason": "Degraded engine performance",
        "actions": [
            "Monitor engine parameters (EGT, N1/N2, oil pressure and temperature) continuously.",
            "Review engine trend-monitoring and oil-consumption data at the next turnaround.",
            "Brief the crew on engine-out, drift-down and asymmetric-thrust procedures.",
            "Reduce thrust demand where performance margins and dispatch policy allow.",
        ],
    },
    "altitude": {
        "reason": "High operating altitude",
        "actions": [
            "Confirm cabin-pressurisation and oxygen systems are serviceable before departure.",
            "Define a descent profile that keeps time-of-useful-consciousness margins.",
            "Use supplemental oxygen if responding to cabin-decompression signs.",
        ],
    },
    "runway_length": {
        "reason": "Short runway / high take-off weight",
        "actions": [
            "Recompute take-off and landing performance with actual weight, wind and runway condition.",
            "Use a reduced take-off flap / assumed-thrust setting only if performance allows.",
            "Identify a go-around and an alternate take-off direction matching the wind.",
        ],
    },
    "airspeed": {
        "reason": "High airspeed (energy on impact)",
        "actions": [
            "Confirm speed limitations for the current flap/slat configuration.",
            "Plan the descent and approach to avoid excess speed at the threshold.",
            "Anticipate increased stopping distance - give extra runway margin.",
        ],
    },
    "fuel_level": {
        "reason": "Reduced fuel reserves",
        "actions": [
            "Declare minimum-fuel / PAN state and notify ATC of the fuel situation.",
            "Divert to the nearest suitable aerodrome without delay.",
            "Recompute endurance at current burn rate and identify the closest alternates.",
        ],
    },
    "flight_duration": {
        "reason": "Long sector / crew fatigue",
        "actions": [
            "Verify crew fatigue-risk management and duty-time limits are respected.",
            "Arrange in-flight rest rotation and extra rest at the destination.",
            "Carry contingency fuel for a possible longer routing or holding.",
        ],
    },
    "turbulence": {
        "reason": "Turbulence encountered",
        "actions": [
            "Reduce airspeed to the turbulence-penetration speed (VRA/VTURB) for the phase of flight.",
            "Turn on the fasten-seatbelt sign and brief cabin crew to secure the galley.",
            "Request a flight level change or re-routing from ATC.",
            "Check for structural load exceedance with the FDR/flight logs after landing.",
        ],
    },
    "visibility": {
        "reason": "Reduced visibility",
        "actions": [
            "Fly a precision approach (ILS / GLS) and brief the autoland / missed-approach plan.",
            "Confirm landing minima for the runway and divert to an alternate if below limits.",
            "Increase separation and keep continuous runway-contact monitoring.",
        ],
    },
    "temperature": {
        "reason": "Extreme temperature",
        "actions": [
            "Recompute take-off performance for hot-day (less air density) conditions.",
            "Verify tyre-pressure and brake-cooling guidance for high temperatures.",
            "For cold extremes follow cold-weather / de-icing starting procedures.",
        ],
    },
    "dew_point": {
        "reason": "Moist air near temperature (fog / mist)",
        "actions": [
            "Highly likely fog - carry extra fuel for a possible alternate approach.",
            "Plan for TCAS and ATC support early; use a precision approach.",
            "Brief a low-visibility take-off (LVTO) procedure if required.",
        ],
    },
    "humidity": {
        "reason": "Humid conditions",
        "actions": [
            "Watch for carburettor / inlet icing band conditions and use heat as needed.",
            "Use anti-icing in high-humidity low-temperature air.",
        ],
    },
    "precipitation": {
        "reason": "Precipitation along the route",
        "actions": [
            "Confirm anti-icing and rain-repellent systems are functioning.",
            "Plan for degraded braking action and reduced runway cleared length.",
            "Use weather radar to circumnavigate embedded storm cells.",
        ],
    },
    "wind_speed": {
        "reason": "Strong winds",
        "actions": [
            "Use the weather radar / wind data to plan the optimum flight level.",
            "Remember tailwind and crosswind components when computing landing distance.",
            "Anticipate turbulence and give extra ATC separation.",
        ],
    },
    "wind_gust": {
        "reason": "Gusty winds",
        "actions": [
            "Add a gust-factor to approach speed (Vref + gust correction, max limit).",
            "Brief go-around on each approach and discontinue on touchdown bounce.",
        ],
    },
    "crosswind": {
        "reason": "Strong crosswind",
        "actions": [
            "Use the maximum-crab or wing-low crosswind technique to the aircraft's limit.",
            "Consider changing runway or diverting to meet the crosswind component.",
            "Brief the go-around plan and use the crosswind limit line memorised by the crew.",
        ],
    },
    "air_pressure": {
        "reason": "Low-pressure / unstable airmass",
        "actions": [
            "Expect thunderstorms, turbulence and shear - brief the crew on avoidance strategy.",
            "Check winds-aloft and jet-stream location for the planned flight level.",
            "Carry contingency fuel for re-routing.",
        ],
    },
    "night_flight": {
        "reason": "Night operation",
        "actions": [
            "Confirm night lighting (landing, taxi, cockpit) and instrument readiness.",
            "Watch for disorientation near horizon-less terrain - rely on instruments.",
            "Extra fatigue monitoring and situation-awareness briefs.",
        ],
    },
    "ground_speed": {
        "reason": "High ground speed",
        "actions": [
            "Anticipate faster touchdown and longer roll-out - plan landing distance accordingly.",
            "Use speed brakes early on the approach to reduce energy.",
        ],
    },
    "vertical_speed": {
        "reason": "Excessive climb / descent rate",
        "actions": [
            "Arrest the sink rate with gentle pitch-up and added thrust; avoid an abrupt pull.",
            "For high climb rates, watch for the 'back side of the power curve' stall.",
            "Set a managed vertical-speed target and cross-check altimeter and VSI.",
        ],
    },
    "wind_shear": {
        "reason": "Wind shear present",
        "actions": [
            "Execute the wind-shear escape manoeuvre: maximum thrust, pitch to flight-path indicator, bypass warnings.",
            "If on final approach and a shear is reported or encountered, go around immediately.",
            "Postpone approach until the shear dissipates or the airport issues a special report.",
        ],
    },
    "stall_margin": {
        "reason": "Low margin above stall speed",
        "actions": [
            "Reduce angle of attack by lowering the nose and add maximum thrust.",
            "Disengage autopilot if it is not recovering airspeed.",
            "Use stick-pusher / stall-warning cues to manage recovery and request vectors.",
            "Declare PAN-PAN if recovery is not immediate.",
        ],
    },
    "g_load": {
        "reason": "High / low vertical acceleration",
        "actions": [
            "Reduce manoeuvre aggressiveness and limit bank/pull inputs.",
            "Document the load exceedance for an airframe structural check at the destination.",
        ],
    },
    "heading_change": {
        "reason": "Rapid heading / track changes",
        "actions": [
            "Cross-check heading vs. GPS track; a mismatch suggests an autopilot or sensor fault.",
            "If autopilot is erratic, disconnect and hand-fly the aircraft.",
            "Report the anomaly to maintenance with a logbook entry.",
        ],
    },
    "ice_accumulation": {
        "reason": "Airframe / engine icing",
        "actions": [
            "Activate wing and engine anti-ice immediately and monitor ice sensors.",
            "Leave the icing conditions - request an alternate flight level or routing.",
            "Increase speed margins and add power to offset drag and weight.",
        ],
    },
    "maintenance_score": {
        "reason": "Low maintenance / dispatch health",
        "actions": [
            "Review deferred-minimum-egress log and raise urgent items with dispatch engineering.",
            "Request a pre-flight scrutiny of the affected systems by a licensed engineer.",
            "Plan a maintenance visit at the next destination with an itemised fault list.",
        ],
    },
    "pilot_hours": {
        "reason": "Low crew experience on type",
        "actions": [
            "Pair the operating pilot with a senior check captain or type-rated instructor.",
            "Increase monitoring by the pilot monitoring (PM) and use call-outs at every gate.",
            "Carry extra fuel and margin for additional situation-awareness time.",
        ],
    },
    "vibration_level": {
        "reason": "Elevated vibration",
        "actions": [
            "Treat as an engine / rotor imbalance warning - monitor EGT and oil trends.",
            "Reduce engine power step-by-step to find the affected engine while monitoring.",
            "Divert and perform a precautionary shutdown or inspection as per QRH.",
        ],
    },
}

# Feature value ranges accepted by the form (matches the training envelope)
FEATURE_RANGES = DATASET_STATS["feature_ranges"]


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def detect_accident_signatures(values: Dict[str, float]) -> List[Dict[str, Any]]:
    """Return every accident signature triggered by the input parameters."""
    fired = []
    for trig in ACCIDENT_TRIGGERS:
        try:
            if trig["test"](values):
                fired.append(
                    {
                        "feature": trig["feature"],
                        "title": trig["title"],
                        "detail": trig["detail"](values),
                        "weight": trig["weight"],
                    }
                )
        except Exception:
            continue  # never let a trigger crash the prediction
    fired.sort(key=lambda t: t["weight"], reverse=True)
    return fired


def compute_feature_contributions(values: Dict[str, float], for_high_risk: bool = True) -> Dict[str, float]:
    """
    Per-feature share of the high-risk probability (as %).

    Works in scaled space around the scaled vector:
      * for a HIGH-RISK prediction each feature is moved one standard deviation
        in the "safer" direction - the resulting drop in the ensemble's
        high-risk probability is attributed to that feature (what is DRIVING
        the danger);
      * otherwise each feature is moved one standard deviation towards risk and
        the rise in high-risk probability is shown as a "watch-point"
        contributor (what COULD push the flight towards risk).

    Deltas are normalised over ALL features so even small contributors appear.
    """
    x = np.array([[values[f] for f in FEATURES]])
    x_scaled = SCALER.transform(x)[0]
    high_base = float(ENSEMBLE.predict_proba(x_scaled.reshape(1, -1))[0, HIGH_IDX])

    low_centroid = CLASS_MEANS[0]
    high_centroid = CLASS_MEANS[-1]
    # Sign of (high - low) centroid tells us which direction is riskier
    direction = np.sign(high_centroid - low_centroid)

    deltas = np.zeros(len(FEATURES))
    for i, feat in enumerate(FEATURES):
        mod = x_scaled.copy()
        if for_high_risk:
            mod[i] = mod[i] - direction[i] * FEATURE_STD[i]   # "safer"
            high_mod = float(ENSEMBLE.predict_proba(mod.reshape(1, -1))[0, HIGH_IDX])
            deltas[i] = max(0.0, high_base - high_mod)
        else:
            mod[i] = mod[i] + direction[i] * FEATURE_STD[i]   # "riskier" (watch)
            high_mod = float(ENSEMBLE.predict_proba(mod.reshape(1, -1))[0, HIGH_IDX])
            deltas[i] = max(0.0, high_mod - high_base)

    total = float(deltas.sum())
    if total < 1e-9:
        # None of the "what-if" moves changed the high-risk probability, so we
        # fall back to the model's global feature importances.
        return {f: float(v) / sum(FEATURE_IMPORTANCES.values()) * 100.0 for f, v in FEATURE_IMPORTANCES.items()}

    return {f: float(d / total * 100.0) for f, d in zip(FEATURES, deltas)}


def collect_targeted_measures(
    values: Dict[str, float],
    contributions: Dict[str, float],
    signatures: List[Dict[str, Any]],
    max_blocks: int = 8,
) -> List[Dict[str, Any]]:
    """
    Build SPECIFIC safety measures from the SAFETY_PLAYBOOK for the features
    that matter most: accident-signature features first (by trigger weight),
    then the strongest contributors. Capped so the report stays focused while
    the contributor list still shows every factor including small ones.
    """
    # Signature features first (ordered by trigger severity), then contributors
    contrib_ordered = sorted(
        ((f, pct) for f, pct in contributions.items() if pct >= 3.0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    ordered = []
    seen = set()
    for feat in [sig["feature"] for sig in signatures]:
        if feat not in seen:
            seen.add(feat)
            ordered.append(feat)
    for feat, _pct in contrib_ordered:
        if feat not in seen:
            seen.add(feat)
            ordered.append(feat)

    measures = []
    added = set()
    for feat in ordered[:max_blocks]:
        entry = SAFETY_PLAYBOOK.get(feat)
        if not entry or entry["reason"] in added:
            continue
        added.add(entry["reason"])
        measures.append(
            {
                "reason": entry["reason"],
                "feature_label": FEATURE_LABELS[feat],
                "actions": entry["actions"],
            }
        )
    return measures


def run_prediction(values: Dict[str, float]) -> Dict[str, Any]:
    """Scale the inputs, run the ensemble, and bundle up the full result."""
    x = np.array([[values[f] for f in FEATURES]])
    x_scaled = SCALER.transform(x)

    # --- Ensemble prediction ------------------------------------------------
    ensemble_probs = ENSEMBLE.predict_proba(x_scaled)[0]
    idx = int(np.argmax(ensemble_probs))
    label = CLASSES[idx]
    confidence = float(ensemble_probs[idx] * 100)

    probs = {cls: round(float(p * 100), 2) for cls, p in zip(CLASSES, ensemble_probs)}

    # --- Per-model consensus ------------------------------------------------
    model_predictions = []
    votes: Dict[str, int] = {cls: 0 for cls in CLASSES}
    for name, model in MODELS.items():
        p = model.predict_proba(x_scaled)[0]
        i = int(np.argmax(p))
        votes[CLASSES[i]] += 1
        model_predictions.append(
            {
                "name": name,
                "label": CLASSES[i],
                "confidence": round(float(p[i] * 100), 2),
                "code": RISK_META[CLASSES[i]]["code"],
            }
        )
    ensemble_idx = int(np.argmax(ensemble_probs))
    votes[CLASSES[ensemble_idx]] += 1
    model_predictions.append(
        {
            "name": "Ensemble (Soft Voting)",
            "label": CLASSES[ensemble_idx],
            "probability": probs[CLASSES[ensemble_idx]],
            "confidence": round(float(ensemble_probs[ensemble_idx] * 100), 2),
            "code": RISK_META[CLASSES[ensemble_idx]]["code"],
        }
    )
    consensus = {
        "votes": votes,
        "high_votes": votes.get("High Risk", 0),
        "models_total": len(MODELS) + 1,
    }

    # --- Accident detection --------------------------------------------------
    signatures = detect_accident_signatures(values)
    high_prob = float(ensemble_probs[HIGH_IDX] * 100)
    trigger_bonus = sum(sig["weight"] for sig in signatures)
    # Blend the model's high-risk probability with hard risk signatures
    accident_prob = float(np.clip(high_prob * 0.55 + trigger_bonus + 12.0, 0.0, 99.0))
    if accident_prob >= 65.0:
        accident_status = ACCIDENT_STATUS["danger"]
        accident_detected = True
    elif accident_prob >= 40.0:
        accident_status = ACCIDENT_STATUS["watch"]
        accident_detected = False
    else:
        accident_status = ACCIDENT_STATUS["safe"]
        accident_detected = False
    accident = {
        "probability": round(accident_prob, 2),
        "status": accident_status,
        "detected": accident_detected,
        "signatures": signatures,
        "trigger_count": len(signatures),
    }

    # --- Risk contributors (ALL, including small ones) -----------------------
    for_high_risk = label == "High Risk"
    contributions = compute_feature_contributions(values, for_high_risk=for_high_risk)
    contributors_sorted = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    contributors = [
        {
            "feature": feat,
            "pct": pct,
            "label": FEATURE_LABELS[feat],
            "tier": "major" if pct >= 6.0 else "minor",
        }
        for feat, pct in contributors_sorted
    ]
    top_contributors = contributors[:3]
    contributor_note = (
        "Factors currently driving this flight towards the High-Risk class."
        if for_high_risk
        else "Watch-points that would raise this flight's risk if they worsened."
    )

    targeted_measures = collect_targeted_measures(values, contributions, signatures)
    # Targeted, parameter-specific measures matter for Medium / High risk; a
    # Low-risk flight stays with the general "standard procedures" checklist.
    if label == "Low Risk":
        targeted_measures = []

    return {
        "label": label,
        "risk_meta": RISK_META[label],
        "confidence": round(confidence, 2),
        "probabilities": probs,
        "probability_list": [
            {"class": cls, "value": round(float(p * 100), 2)}
            for cls, p in zip(CLASSES, ensemble_probs)
        ],
        "model_predictions": model_predictions,
        "consensus": consensus,
        "accident": accident,
        "contributors": contributors,
        "top_contributors": top_contributors,
        "contributor_note": contributor_note,
        "targeted_measures": targeted_measures,
        "features": values,
    }


# ------------------------------------------------------------------------------
# Context processor: expose model metadata to every template
# ------------------------------------------------------------------------------
@app.context_processor
def inject_globals() -> Dict[str, Any]:
    return {
        "model_name": "QDA + Random Forest + Gradient Boosting + Logistic Regression (soft-voting ensemble)",
        "accuracy": ACCURACY,
        "classes": CLASSES,
        "feature_labels": FEATURE_LABELS,
        "feature_importances": FEATURE_IMPORTANCES,
        "model_accuracies": MODEL_ACCURACIES,
        "model_order": MODEL_ORDER,
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
    return render_template(
        "about.html",
        model_accuracies=MODEL_ACCURACIES,
        model_order=MODEL_ORDER,
        feature_importances=FEATURE_IMPORTANCES,
    )


@app.route("/prediction")
def prediction():
    return render_template(
        "prediction.html", feature_labels=FEATURE_LABELS, form_values={}
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts the flight parameters, validates them, runs the ensemble model and
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

    # Persist a SLIM snapshot for the dashboard (the full result model above is
    # too large to fit inside Flask's signed-session cookie limit).
    session["last_prediction"] = {
        "label": result["label"],
        "risk_meta": result["risk_meta"],
        "confidence": result["confidence"],
        "probability_list": result["probability_list"],
        "accident": {
            "probability": result["accident"]["probability"],
            "status": result["accident"]["status"],
        },
    }

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
        model_accuracies=MODEL_ACCURACIES,
        model_order=MODEL_ORDER,
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