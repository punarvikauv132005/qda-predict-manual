# Aviation Accident Prediction System using QDA + ML Ensemble

A complete full-stack web application that predicts the risk of an aviation
accident based on flight parameters, weather conditions and **in-flight
movement parameters** entered **manually by the user**. The backend uses a
**soft-voting ensemble of machine learning algorithms** — **Quadratic
Discriminant Analysis (QDA)**, Random Forest, Gradient Boosting and Logistic
Regression — to classify each flight into one of three risk categories and to
**detect accident probability** with specific warning signatures:

- 🟢 **Low Risk** — flight conditions are safe
- 🟡 **Medium Risk** — monitor weather, reduce speed if required
- 🔴 **High Risk** — delayed/denied departure, notify ATC, targeted safety measures

![Stack](https://img.shields.io/badge/Flask-3-blue) ![ML](https://img.shields.io/badge/Scikit--learn-QDA%2BEnsemble-orange) ![Frontend](https://img.shields.io/badge/Bootstrap-5-violet)

---

## Features

| Feature | Description |
| --- | --- |
| Manual Weather Entry | Temperature, wind speed, humidity, air pressure & visibility are typed directly into the form (no API required). |
| In-Flight Dynamics | Vertical speed, wind shear, stall margin, G-load, heading change, icing, vibration and more while the aircraft is moving. |
| QDA + ML Ensemble | QDA, Random Forest, Gradient Boosting and Logistic Regression combined with a soft-voting ensemble — no single algorithm decides alone. |
| Accident Detection | A dedicated accident probability gauge driven by the High-Risk posterior **plus hard safety signatures** (near-stall, wind shear, high sink rate, engine-critical, fuel-critical, severe icing, ...). |
| Model Consensus | Shows how each of the 5 models voted for every prediction. |
| All Risk Contributors | The full contributor breakdown — **including small (minor) contributors**, not just the top 3. |
| Targeted Safety Measures | High/Medium-risk flights get **parameter-specific** actions — e.g. wind-shear escape manoeuvre, stall recovery, anti-ice activation — instead of generic advice only. |
| Confidence Scores | Every prediction returns a probability for each risk class. |
| Dashboard | Per-model accuracy comparison chart, model accuracy, confusion matrix image and dataset statistics. |
| High-Risk Alert | Warning sound + pulsing emergency card for High Risk / accident-likely flights. |
| Responsive UI | Modern aviation dashboard theme (navy / white / sky blue), smooth scrolling, fade effects and hover animations. |

## Tech Stack

- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5, Font Awesome, Chart.js
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn — QDA + Random Forest + Gradient Boosting + Logistic Regression (soft-voting ensemble)
- **Model:** `qda_model.pkl` (loaded at app startup)
- **Deployment:** GitHub + Render

---

## Folder Structure

```
aviation-accident-predictor/
├── app.py                  # Flask application
├── train_model.py          # Dataset generation + QDA training script
├── requirements.txt
├── runtime.txt
├── Procfile
├── qda_model.pkl           # Trained model bundle
├── README.md
├── .gitignore
├── static/
│   ├── style.css
│   ├── script.js
│   └── images/
│       └── confusion_matrix.png
├── templates/
│   ├── base.html           # Shared layout (navbar + footer)
│   ├── index.html          # Home
│   ├── about.html          # About / methodology
│   ├── prediction.html     # Input form
│   ├── result.html         # Risk report
│   ├── dashboard.html      # Analytics dashboard
│   └── contact.html        # Project team
└── dataset/
    └── aviation.csv        # Labelled flight records
```

## Website Pages

| Route | Page | Description |
| --- | --- | --- |
| `/` | Home | Hero section with aviation image, risk classes, how-it-works and Start Prediction button. |
| `/about` | About | What is aviation accident prediction, what is QDA, why QDA, objectives and workflow diagram. |
| `/prediction` | Prediction | Full input form (reference dropdowns + optional airport + aircraft type + 28 features) with manually entered weather and in-flight dynamics. |
| `/predict` (POST) | Result | Colour-coded risk card, accident detection panel, model consensus, probability, confidence, targeted safety measures and ALL risk contributors. |
| `/dashboard` | Dashboard | Ensemble accuracy, per-model accuracy comparison, confusion matrix, class distribution and dataset statistics. |
| `/contact` | Contact | Project guide, team members and college details. |

## Model Features

The ensemble is trained on these **28 features** (manual entry):

**Aircraft / take-off**
1. Aircraft Age (years)
2. Engine Health (%)
3. Altitude (ft)
4. Runway Length (m)
5. Airspeed (km/h)
6. Fuel Level (%)
7. Flight Duration (hours)
8. Turbulence Level (0–10)

**Weather**
9. Visibility (km)
10. Temperature (°C)
11. Dew Point (°C)
12. Humidity (%)
13. Precipitation (mm/h)
14. Wind Speed (km/h)
15. Wind Gust (km/h)
16. Crosswind (km/h)
17. Air Pressure (hPa)
18. Night Flight (0 = day, 1 = night)

**Aircraft & in-flight dynamics (while moving)**
19. Ground Speed (km/h)
20. Vertical Speed (ft/min) — climb / sink rate
21. Wind Shear (km/h)
22. Stall Margin (km/h) above stall speed
23. Vertical Acceleration (g)
24. Heading Change (deg/min)
25. Ice Accumulation (0–1)
26. Maintenance Score (%)
27. Pilot Experience (log hours)
28. Vibration Level (0–10)

The form also accepts optional **reference fields** (Flight Number and Route as
dropdowns, plus Flight Date and Airport) that are displayed for record-keeping
but do **not** affect the prediction.

## Installation & Running Locally

Prerequisites: **Python 3.10+** and **pip**.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/aviation-accident-predictor.git
cd aviation-accident-predictor

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Retrain the model & regenerate the dataset
python train_model.py

# 5. Run the app
python app.py
```

Open <http://127.0.0.1:5000> in your browser.

> **Note:** The `qda_model.pkl` and `dataset/aviation.csv` files are already
> committed, so step 4 is optional. Every model uses a fixed seed
> (`random_state=42`) so a re-run reproduces the same ensemble. The dataset is
> synthetically generated (see `train_model.py`) because a public dataset
> containing every required field at flight level is not available.

## Manual Weather & In-Flight Entry

The prediction page has **no live weather API** — every field is entered by
hand. The **Aircraft & In-Flight Dynamics** section (ground speed, vertical
speed, wind shear, stall margin, G-load, heading change, icing, maintenance
score, pilot hours, vibration) models the aircraft while it is moving, so the
prediction reflects an in-motion flight rather than a static pre-flight check.

1. Visibility (km)
2. Temperature (°C)
3. Humidity (%)
4. Wind Speed (km/h)
5. Air Pressure (hPa)

Take the values from the METAR / weather report for your airport, or enter any
values to run "what-if" scenarios. Each field is validated against the model's
training ranges before prediction.

## Deploying to Render

1. Push the repository to GitHub.
2. In the Render dashboard, click **New > Web Service**.
3. Connect your GitHub repository.
4. Render auto-detects the Python environment. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Optionally set an environment variable `SECRET_KEY`.
6. Click **Create Web Service**. The app goes live at `https://<app-name>.onrender.com`.

Render uses the `.python-version` file (Python 3.12) and `Procfile` automatically. The `runtime.txt` file is kept for reference only — Render no longer reads it.

## GitHub Instructions

```bash
git init
git add .
git commit -m "Initial commit: Aviation Accident Prediction using QDA"
git branch -M main
git remote add origin https://github.com/<your-username>/aviation-accident-predictor.git
git push -u origin main
```

## Project Team

- **Project Guide:** Dr. Shridhar Kabbur
- **Team Members:** Inchara, Sinchana, Nidhi, Punarvika
- **College:** Global Academy of Technology, Department of ECE

## Disclaimer

This project is an academic demonstration. It uses a synthetically generated
dataset and should **not** be used for real flight-safety decisions.
