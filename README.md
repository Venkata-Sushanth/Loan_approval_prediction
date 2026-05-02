# 🏦 LoanWise — AI-Powered Loan Eligibility Predictor

An ML-powered web application that predicts whether a bank loan application will be **approved or rejected**, built using the Loan Prediction Dataset with a clean banking-themed UI designed for all age groups.

---

## 📊 Dataset

- **File:** `loan_data.csv`
- **Source:** [Loan Prediction Dataset — Analytics Vidhya / Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Size:** 381 loan applications × 13 columns
- **Target:** `Loan_Status` — Y (Approved) or N (Rejected)
- **Approval rate in dataset:** ~71% approved, ~29% rejected

### Original Columns

| Column | Description |
|---|---|
| Loan_ID | Unique identifier (dropped before training) |
| Gender | Male / Female |
| Married | Yes / No |
| Dependents | 0 / 1 / 2 / 3+ |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| ApplicantIncome | Monthly income of main applicant |
| CoapplicantIncome | Monthly income of co-applicant |
| LoanAmount | Requested loan amount (₹ thousands) |
| Loan_Amount_Term | Repayment duration in months |
| Credit_History | 1 = Good history, 0 = Bad / No history |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | **Target** — Y (Approved) / N (Rejected) |

---

## 🤖 4 Models Trained & Compared

| Model | Accuracy | ROC-AUC | F1 Score | CV AUC Mean | CV AUC Std |
|---|---|---|---|---|---|
| 🔮 Gradient Boosting | 83.12% | **0.8339** ⭐ | 0.8870 | 0.8043 | ±0.0829 |
| 🌲 Random Forest | **85.71%** ⭐ | 0.7884 | **0.9060** ⭐ | 0.7714 | ±0.1118 |
| 📐 Logistic Regression | 81.82% | 0.8215 | 0.8750 | 0.7308 | ±0.0846 |
| 🔵 KNN Classifier | 74.03% | 0.6822 | 0.8413 | 0.6169 | ±0.0559 |

> ⭐ = Best in that metric

---

## ⚙️ Key ML Concepts Used

- **Data Cleaning** — Missing values in Gender (5), Self_Employed (21), Loan_Amount_Term (11), Credit_History (30), Dependents (8) filled using mode imputation
- **Label Encoding** — Categorical columns (Gender, Married, Education, Self_Employed, Property_Area) converted to numbers
- **KNN Imputer (k=5)** — Fills any remaining numeric missing values using 5 most similar loan applications — used inside ALL 4 pipelines
- **Robust Scaler** — Scales features using median + IQR (not mean), resistant to income outliers
- **Feature Engineering** — 5 new interaction features created from original columns
- **Feature Dependency Analysis** — Each feature dropped one at a time, all 4 models retrained, AUC drop measured (17 features × 4 models = **68 retraining experiments**)
- **5-Fold Stratified Cross Validation** — Tests model consistency across different data splits
- **Sklearn Pipeline** — Chains imputer → scaler → classifier so no data leakage during cross-validation

---

## ⚗️ Feature Engineering — 6 Engineered Features

| Feature | Formula | Why It Helps |
|---|---|---|
| `Total_Income` | ApplicantIncome + CoapplicantIncome | Combined repayment capacity |
| `Log_Total_Income` | log(Total_Income + 1) | Reduces income skewness |
| `EMI` | LoanAmount ÷ (Loan_Amount_Term + 1) | Estimated monthly burden |
| `Debt_Income_Ratio` | EMI ÷ (Total_Income + 1) | Repayment burden as % of income |
| `Income_Per_Dependent` | Total_Income ÷ (Dependents + 1) | Financial stress per family member |
| `Loan_Income_Ratio` | LoanAmount ÷ (Total_Income + 1) | How big is the loan vs income |

**Total features used for training: 17** (11 original + 6 engineered)

---

## 🧬 Feature Dependency Analysis — Top Results

| Rank | Feature | Type | Avg AUC Drop | Verdict |
|---|---|---|---|---|
| 1 | Credit_History | Original | +0.1636 | 🟢 Most important — removing it hurts badly |
| 2 | Property_Area | Original | +0.0342 | 🟢 Significant impact |
| 3 | Self_Employed | Original | +0.0132 | 🟡 Moderate impact |
| 4 | Dependents | Original | +0.0081 | 🟡 Moderate impact |
| 5 | EMI | Engineered | +0.0040 | 🟡 Slight positive |
| ... | Gender | Original | -0.0105 | 🔴 Removing it slightly helps — adds noise |
| ... | Loan_Amount_Term | Original | -0.0107 | 🔴 Removing it slightly helps — adds noise |

> A **positive drop** = feature was useful (model got worse without it).
> A **negative drop** = feature was adding noise (model got better without it).

**Credit_History is by far the most important feature** — removing it dropped AUC by 0.1636 on average across all 4 models.

---

## 🔬 Full ML Pipeline

```
loan_data.csv
      ↓
Drop Loan_ID (identifier, no predictive value)
      ↓
Label Encode categorical columns (Gender, Married, Education, Self_Employed, Property_Area)
      ↓
Feature Engineering (11 original + 6 new = 17 features)
      ↓
Train/Test Split — 80% train (304 samples) / 20% test (77 samples), stratified
      ↓
Pipeline: KNN Imputer (k=5) → Robust Scaler → Classifier
      ↓
Train 4 Models + 5-Fold Cross Validation
      ↓
Feature Dependency Analysis (68 retraining experiments)
      ↓
Save .pkl files → Flask REST API → Web UI
```

---

## 🌐 Web App Features

The web app has **4 tabs**:

| Tab | What It Shows |
|---|---|
| 🔍 Check Eligibility | Enter your details, choose a model, get instant approval probability |
| 📊 Model Comparison | Bar charts comparing Accuracy, AUC, F1, CV scores for all 4 models |
| 🧬 Feature Impact | Full ranked table showing AUC drop per feature per model (68 experiments) |
| ⚙️ How It Works | Full pipeline walkthrough, all 17 features explained |

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/loanwise.git
cd loanwise
```

### 2. Create virtual environment

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the models
```bash
# Mac/Linux
python3 train_model.py

# Windows
python train_model.py
```

This reads `loan_data.csv`, cleans data, engineers features, trains all 4 models, runs feature dependency analysis (68 experiments), and saves 4 `.pkl` files + `model_meta.json`.

Expected output:
```
DATASET LOADED FROM loan_data.csv
Shape: 381 rows × 13 columns
...
Training: Gradient Boosting  →  Accuracy: 83.12%  AUC: 0.8339
Training: Random Forest      →  Accuracy: 85.71%  AUC: 0.7884
Training: Logistic Regression→  Accuracy: 81.82%  AUC: 0.8215
Training: KNN Classifier     →  Accuracy: 74.03%  AUC: 0.6822
...
Feature dependency analysis complete (68 experiments)
ALL DONE — Run: python app.py
```

### 5. Run the web app
```bash
# Mac/Linux
python3 app.py

# Windows
python app.py
```

### 6. Open in browser
```
http://localhost:5000
```

---

## 📁 Project Structure

```
loanwise/
├── static/
│   └── index.html                    # Full banking web UI (4 tabs)
├── app.py                            # Flask REST API backend
├── train_model.py                    # Full ML pipeline & training
├── loan_data.csv                     # Real loan dataset (381 records)
├── model_gradient_boosting.pkl       # Trained GB pipeline
├── model_random_forest.pkl           # Trained RF pipeline
├── model_logistic_regression.pkl     # Trained LR pipeline
├── model_knn_classifier.pkl          # Trained KNN pipeline
├── model_meta.json                   # Metrics + feature dependency data
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 📦 Dependencies

```
flask
scikit-learn
pandas
numpy
joblib
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🆚 Mac vs Windows — Key Differences

| Action | Mac / Linux | Windows |
|---|---|---|
| Activate venv | `source venv/bin/activate` | `venv\Scripts\activate` |
| Python command | `python3` | `python` |
| Navigate folders | `cd ~/Desktop/loanwise` | `cd Desktop\loanwise` |

---

## 🎓 What to Tell Your Teacher

> "We load `loan_data.csv` directly using pandas. The dataset has 381 real bank loan records with 13 columns. We cleaned missing values, label-encoded categorical features, and engineered 6 new features based on financial domain knowledge — such as EMI, Debt-to-Income Ratio, and Income per Dependent. We then trained 4 classifiers using an sklearn Pipeline (KNN Imputer → Robust Scaler → Classifier), validated each with 5-fold stratified cross-validation, and performed Feature Dependency Analysis across all 4 models (68 retraining experiments). Results are served through a Flask REST API to a banking-themed web interface."

---

## ⚠️ Disclaimer

For **educational purposes only**. Not actual financial advice. Always consult a licensed financial institution for real loan decisions.
