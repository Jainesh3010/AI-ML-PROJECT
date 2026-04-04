# AI ML Portfolio – Jainesh Sanghavi
## Complete Flask + ML Web App

---

## Folder Structure
```
ai_ml_portfolio/
├── app.py                  ← Flask backend (main file)
├── requirements.txt        ← Python dependencies
├── model_promotion.pkl     ← Promotion model  (8 features)
├── model_salary.pkl        ← Salary model     (5 features)
├── model_churn.pkl         ← Churn model      (9 features)
├── model_house.pkl         ← House model      (3 features)
├── static/
│   ├── churn_feature.png   ← Feature importance charts
│   ├── feature.png
│   ├── House.png
│   └── promotion.png
└── templates/
    └── index.html          ← Frontend (auto-served by Flask)
```

---

## How to Run

### Step 1 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 – Start the server
```bash
python app.py
```

### Step 3 – Open in browser
```
http://127.0.0.1:5000
```

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET  | `/` | Serves index.html |
| GET  | `/health` | Check all models loaded |
| POST | `/predict/salary` | Salary prediction |
| POST | `/predict/churn` | Customer churn prediction |
| POST | `/predict/promotion` | Promotion prediction |
| POST | `/predict/house` | House price prediction |
| GET  | `/graph/<model>` | Latest generated graph PNG |

---

## How It Works
1. User clicks a model card → modal opens with input form
2. User fills fields and clicks "Predict"
3. JS `fetch()` sends POST request to Flask `/predict/<model>`
4. Flask loads the pkl model, runs prediction, generates Matplotlib graph
5. Result shown in modal; "Show Graph" button appears
6. Click "Show Graph" → fetches `/graph/<model>` → displays inline

---

## Graph Types (auto-generated per prediction)
- **Salary** – Salary vs Experience curve (your input highlighted)
- **Churn** – Monthly Charges vs Churn Risk probability curve
- **Promotion** – KPI Score vs Promotion Probability curve
- **House** – House Price vs Area curve (your property highlighted)
