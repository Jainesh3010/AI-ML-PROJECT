# ================================================================
#  app.py  –  AI ML Portfolio Backend
#  Author  : Jainesh Sanghavi
#  Purpose : Flask server connecting 4 ML models to the frontend
#            with dynamic graph generation after each prediction
# ================================================================
#
#  Models & PKL files:
#    model_promotion.pkl  – Promotion Prediction  (Classifier)
#    model_salary.pkl     – Salary Prediction     (Regressor)
#    model_churn.pkl      – Customer Churn        (Classifier)
#    model_house.pkl      – House Price           (Regressor)
#
#  Run:
#    pip install flask flask-cors scikit-learn matplotlib numpy
#    python app.py
#    Open  http://127.0.0.1:5000
# ================================================================

import os
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  FLASK SETUP
# ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # allow all origins (needed when opening index.html directly from browser)

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
os.makedirs(STATIC, exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  LOAD MODELS
# ──────────────────────────────────────────────────────────────
def load(filename):
    path = os.path.join(BASE, filename)
    if not os.path.exists(path):
        print(f"  ⚠  {filename} not found – endpoint will return 503")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

print("\n🔄  Loading models...")
MODELS = {
    "promotion": load("model1.pkl"),
    "salary":    load("model2.pkl"),
    "churn":     load("model3.pkl"),
    "house":     load("model4.pkl"),
}
print("✅  Models loaded\n")


# ──────────────────────────────────────────────────────────────
#  GRAPH STYLE  (matches the site's dark aesthetic)
# ──────────────────────────────────────────────────────────────
DARK_BG   = "#0e0e0e"
GRID_CLR  = "#1e1e1e"
TEXT_CLR  = "#a0a0a0"
BAR_CLR   = "#e0e0e0"
ACCENT    = "#ffffff"
HIGHLIGHT = "#888888"

def _base_style():
    """Apply dark theme to current matplotlib figure."""
    fig = plt.gcf()
    ax  = plt.gca()
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(ACCENT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.6, linestyle="--")


# ──────────────────────────────────────────────────────────────
#  HELPER – safe JSON field extractor
# ──────────────────────────────────────────────────────────────
def get(data, key, typ=float):
    if key not in data:
        raise ValueError(f"Missing field: '{key}'")
    try:
        return typ(data[key])
    except (ValueError, TypeError):
        raise ValueError(f"Field '{key}' must be a number. Got: {data[key]!r}")


# ================================================================
#  ROUTE – serve frontend
# ================================================================
@app.route("/")
def home():
    return render_template("index.html")


# ================================================================
#  HEALTH CHECK
# ================================================================
@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "models": {k: v is not None for k, v in MODELS.items()}
    })


# ================================================================
#  GRAPH ENDPOINT  – serves the latest generated PNG
# ================================================================
@app.route("/graph/<model_type>")
def serve_graph(model_type):
    """
    Returns the most recently generated graph for a given model.
    URL: GET /graph/salary  |  /graph/churn  |  /graph/promotion  |  /graph/house
    """
    path = os.path.join(STATIC, f"graph_{model_type}.png")
    if not os.path.exists(path):
        # Fall back to the static feature-importance chart shipped with the project
        fallback_map = {
            "salary":    "feature.png",
            "churn":     "churn_feature.png",
            "promotion": "promotion.png",
            "house":     "House.png",
        }
        fb = fallback_map.get(model_type)
        if fb:
            fb_path = os.path.join(STATIC, fb)
            if os.path.exists(fb_path):
                return send_file(fb_path, mimetype="image/png")
        return jsonify({"error": "Graph not yet generated. Run a prediction first."}), 404

    return send_file(path, mimetype="image/png")


# ================================================================
#  PREDICT – SALARY
#  Features : Age, Experience, Education (0-2), JobRole (0-4), Location (0-3)
#  Output   : Predicted annual salary in USD
# ================================================================
@app.route("/predict/salary", methods=["POST"])
def predict_salary():
    if MODELS["salary"] is None:
        return jsonify({"error": "Salary model not loaded."}), 503
    try:
        d = request.get_json(force=True)
        age  = get(d, "Age");        exp  = get(d, "Experience")
        edu  = get(d, "Education");  job  = get(d, "JobRole")
        loc  = get(d, "Location")

        X      = np.array([[age, exp, edu, job, loc]])
        result = float(MODELS["salary"].predict(X)[0])

        # ── Generate dynamic graph: Salary vs Experience curve ──
        exp_range = np.arange(0, 36)
        preds = [float(MODELS["salary"].predict(
                    np.array([[age, e, edu, job, loc]]))[0])
                 for e in exp_range]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(exp_range, preds, color=BAR_CLR, linewidth=2)
        ax.scatter([exp], [result], color=ACCENT, s=100, zorder=5,
                   label=f"Your input (₹{result:,.0f})")
        ax.fill_between(exp_range, preds, alpha=0.08, color=BAR_CLR)
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Predicted Salary")
        ax.set_title("Salary vs Experience")
        ax.legend(facecolor=DARK_BG, labelcolor=TEXT_CLR, fontsize=8)
        _base_style()
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC, "graph_salary.png"), dpi=120,
                    facecolor=DARK_BG, bbox_inches="tight")
        plt.close()

        return jsonify({
            "prediction": f"{result:,.0f} / month",
            "raw": round(result, 2),
            "show_graph_option": True,
            "graph_url": "/graph/salary"
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


# ================================================================
#  PREDICT – CHURN
#  Features : Gender, SeniorCitizen, Partner, Dependents,
#             Tenure, MonthlyCharges, TotalCharges,
#             Contract, PaymentMethod
#  Output   : "Churn" / "No Churn"
# ================================================================
@app.route("/predict/churn", methods=["POST"])
def predict_churn():
    if MODELS["churn"] is None:
        return jsonify({"error": "Churn model not loaded."}), 503
    try:
        d = request.get_json(force=True)
        gender  = get(d, "Gender");    senior  = get(d, "SeniorCitizen")
        partner = get(d, "Partner");   deps    = get(d, "Dependents")
        tenure  = get(d, "Tenure");    monthly = get(d, "MonthlyCharges")
        total   = get(d, "TotalCharges")
        contract= get(d, "Contract");  payment = get(d, "PaymentMethod")

        X      = np.array([[gender, senior, partner, deps,
                             tenure, monthly, total, contract, payment]])
        raw    = int(MODELS["churn"].predict(X)[0])
        label  = "Churn ⚠️" if raw == 1 else "No Churn ✅"

        # ── Generate graph: Monthly Charges vs Churn Risk ──
        mc_range = np.arange(20, 121, 5)
        risks = []
        for mc in mc_range:
            tc = mc * tenure
            xi = np.array([[gender, senior, partner, deps,
                             tenure, mc, tc, contract, payment]])
            prob = MODELS["churn"].predict_proba(xi)[0][1]
            risks.append(prob * 100)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(mc_range, risks, alpha=0.15, color=BAR_CLR)
        ax.plot(mc_range, risks, color=BAR_CLR, linewidth=2)
        ax.axvline(monthly, color=ACCENT, linestyle="--", linewidth=1.2,
                   label=f"Your charge (${monthly:.0f})")
        ax.set_xlabel("Monthly Charges ($)")
        ax.set_ylabel("Churn Risk (%)")
        ax.set_title("Monthly Charges vs Churn Risk")
        ax.legend(facecolor=DARK_BG, labelcolor=TEXT_CLR, fontsize=8)
        _base_style()
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC, "graph_churn.png"), dpi=120,
                    facecolor=DARK_BG, bbox_inches="tight")
        plt.close()

        return jsonify({
            "prediction": label,
            "raw": raw,
            "show_graph_option": True,
            "graph_url": "/graph/churn"
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


# ================================================================
#  PREDICT – PROMOTION
#  Features : Department (0-4), Education (0-2), Gender (0/1),
#             Age, Experience, KPI_Score, Awards (0/1), Previous_Rating (1-5)
#  Output   : "Promoted" / "Not Promoted"
# ================================================================
@app.route("/predict/promotion", methods=["POST"])
def predict_promotion():
    if MODELS["promotion"] is None:
        return jsonify({"error": "Promotion model not loaded."}), 503
    try:
        d = request.get_json(force=True)
        dept   = get(d, "Department");  edu    = get(d, "Education")
        gender = get(d, "Gender");      age    = get(d, "Age")
        exp    = get(d, "Experience");  kpi    = get(d, "KPI_Score")
        awards = get(d, "Awards");      rating = get(d, "Previous_Rating")

        X   = np.array([[dept, edu, gender, age, exp, kpi, awards, rating]])
        raw = int(MODELS["promotion"].predict(X)[0])
        label = "Promoted 🏆" if raw == 1 else "Not Promoted"

        # ── Generate graph: KPI Score vs Promotion Probability ──
        kpi_range = np.arange(0, 101, 2)
        probs = []
        for k in kpi_range:
            xi = np.array([[dept, edu, gender, age, exp, k, awards, rating]])
            prob = MODELS["promotion"].predict_proba(xi)[0][1]
            probs.append(prob * 100)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(kpi_range, probs, alpha=0.12, color=BAR_CLR)
        ax.plot(kpi_range, probs, color=BAR_CLR, linewidth=2)
        ax.axvline(kpi, color=ACCENT, linestyle="--", linewidth=1.2,
                   label=f"Your KPI ({kpi:.0f})")
        ax.set_xlabel("KPI Score")
        ax.set_ylabel("Promotion Probability (%)")
        ax.set_title("KPI Score vs Promotion Probability")
        ax.legend(facecolor=DARK_BG, labelcolor=TEXT_CLR, fontsize=8)
        _base_style()
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC, "graph_promotion.png"), dpi=120,
                    facecolor=DARK_BG, bbox_inches="tight")
        plt.close()

        return jsonify({
            "prediction": label,
            "raw": raw,
            "show_graph_option": True,
            "graph_url": "/graph/promotion"
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


# ================================================================
#  PREDICT – HOUSE PRICE
#  Features : Area (sq ft), Bedrooms, Age (years)
#  Output   : Predicted price in USD
# ================================================================
@app.route("/predict/house", methods=["POST"])
def predict_house():
    if MODELS["house"] is None:
        return jsonify({"error": "House model not loaded."}), 503
    try:
        d = request.get_json(force=True)
        area     = get(d, "Area")
        bedrooms = get(d, "Bedrooms")
        age      = get(d, "Age")

        X      = np.array([[area, bedrooms, age]])
        result = float(MODELS["house"].predict(X)[0])

        # ── Generate graph: Price vs Area curve ──
        area_range = np.arange(300, 5001, 100)
        preds = [float(MODELS["house"].predict(
                    np.array([[a, bedrooms, age]]))[0])
                 for a in area_range]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(area_range, preds, color=BAR_CLR, linewidth=2)
        ax.fill_between(area_range, preds, alpha=0.08, color=BAR_CLR)
        ax.scatter([area], [result], color=ACCENT, s=100, zorder=5,
                   label=f"Your property (₹{result:,.0f})")
        ax.set_xlabel("Area (sq ft)")
        ax.set_ylabel("Predicted Price")
        ax.set_title("House Price vs Area")
        ax.legend(facecolor=DARK_BG, labelcolor=TEXT_CLR, fontsize=8)
        _base_style()
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC, "graph_house.png"), dpi=120,
                    facecolor=DARK_BG, bbox_inches="tight")
        plt.close()

        return jsonify({
            "prediction": f"${result:,.0f}",
            "raw": round(result, 2),
            "show_graph_option": True,
            "graph_url": "/graph/house"
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


# ──────────────────────────────────────────────────────────────
#  RUN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀  Server running at http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
