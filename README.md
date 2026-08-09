# Aviation Accident Prediction System using QDA

A complete full-stack web application that predicts the risk of an aviation
accident based on flight parameters and weather conditions **entered manually
by the user**. The backend uses **Quadratic Discriminant Analysis (QDA)** from
Scikit-learn to classify each flight into one of three risk categories:

- 🟢 **Low Risk** — flight conditions are safe
- 🟡 **Medium Risk** — monitor weather, reduce speed if required
- 🔴 **High Risk** — delay flight, notify ATC, inspect aircraft

![Stack](https://img.shields.io/badge/Flask-3-blue) ![ML](https://img.shields.io/badge/Scikit--learn-QDA-orange) ![Frontend](https://img.shields.io/badge/Bootstrap-5-violet)

---

## Features

| Feature | Description |
| --- | --- |
| Manual Weather Entry | Temperature, wind speed, humidity, air pressure & visibility are typed directly into the form (no API required). |
| QDA Risk Model | A trained `QuadraticDiscriminantAnalysis` model classifies flights into Low / Medium / High risk. |
| Confidence Scores | Every prediction returns a probability for each risk class. |
| Safety Recommendations | Colour-coded, action-oriented advice per risk level. |
| Dashboard | Charts (Chart.js), model accuracy, confusion matrix image and dataset statistics. |
| High-Risk Alert | Warning sound + pulsing emergency card for High Risk flights. |
| Responsive UI | Modern aviation dashboard theme (navy / white / sky blue), smooth scrolling, fade effects and hover animations. |

## Tech Stack

- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap 5, Font Awesome, Chart.js
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn — Quadratic Discriminant Analysis (QDA)
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
| `/prediction` | Prediction | Full input form (reference fields + optional airport + aircraft type + 14 features) with manually entered weather. |
| `/predict` (POST) | Result | Colour-coded risk card, probability, confidence, recommendations and top risk contributors. |
| `/dashboard` | Dashboard | Model accuracy, confusion matrix, class distribution and dataset statistics. |
| `/contact` | Contact | Project guide, team members and college details. |

## Model Features

The QDA model is trained on these 14 features:

1. Aircraft Age (years)
2. Engine Health (%)
3. Altitude (ft)
4. Airspeed (km/h)
5. Fuel Level (%)
6. Flight Duration (hours)
7. Turbulence Level (0–10)
8. Visibility (km)
9. Temperature (°C)
10. Humidity (%)
11. Precipitation (mm/h)
12. Wind Speed (km/h)
13. Wind Gust (km/h)
14. Air Pressure (hPa)

The form also accepts optional **reference fields** (Flight Number, Flight Date,
Route) that are displayed for record-keeping but do **not** affect the
prediction.

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
> committed, so step 4 is optional. The dataset is synthetically generated
> (see `train_model.py`) because a public dataset containing every required
> field at flight level is not available.

## Manual Weather Entry

The prediction page has **no live weather API** — every field is entered by
hand. An optional **Airport** dropdown is provided for reference only (it does
not affect the prediction). The five weather parameters are:

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
