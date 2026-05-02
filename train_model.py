# ================================================================
#  LoanWise — Loan Approval Predictor
#  Dataset : Loan Prediction Dataset (Banking)
#  File    : loan_data.csv  (381 rows × 13 columns)
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
import joblib, json, warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ────────────────────────────────────────────────────────────────
df = pd.read_csv('loan_data.csv')

print("=" * 60)
print("  DATASET LOADED FROM loan_data.csv")
print("=" * 60)
print(f"\n📌 Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📌 Columns       : {list(df.columns)}")
print(f"\n📌 First 5 rows  :\n{df.head()}")
print(f"\n📌 Class balance :\n{df['Loan_Status'].value_counts()}")
print(f"   → {(df['Loan_Status']=='Y').mean()*100:.1f}% loans approved in dataset")

# ────────────────────────────────────────────────────────────────
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2 : EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"\n📌 Missing values:\n{df.isnull().sum()}")
print(f"\n📌 Basic statistics:\n{df.describe().round(2)}")

# ────────────────────────────────────────────────────────────────
# STEP 3 — DATA CLEANING
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3 : DATA CLEANING")
print("=" * 60)

df_clean = df.copy()

# Drop Loan_ID — it's just an identifier, not a feature
df_clean.drop('Loan_ID', axis=1, inplace=True)
print("   ✅ Dropped Loan_ID (identifier, not a predictor)")

# Dependents: '3+' → 3 (numeric)
df_clean['Dependents'] = df_clean['Dependents'].replace('3+', '3')
df_clean['Dependents'] = pd.to_numeric(df_clean['Dependents'], errors='coerce')
print("   ✅ Dependents: '3+' converted to 3 (numeric)")

# Credit_History: fill missing with mode (most common = 1.0)
credit_mode = df_clean['Credit_History'].mode()[0]
df_clean['Credit_History'].fillna(credit_mode, inplace=True)
print(f"   ✅ Credit_History: {df['Credit_History'].isnull().sum()} missing → filled with mode ({credit_mode})")

# Loan_Amount_Term: fill missing with mode (360 months = 30 years)
term_mode = df_clean['Loan_Amount_Term'].mode()[0]
df_clean['Loan_Amount_Term'].fillna(term_mode, inplace=True)
print(f"   ✅ Loan_Amount_Term: {df['Loan_Amount_Term'].isnull().sum()} missing → filled with mode ({term_mode})")

# Gender, Married, Dependents, Self_Employed: fill with mode
for col in ['Gender', 'Married', 'Self_Employed', 'Dependents']:
    mode_val = df_clean[col].mode()[0]
    missing_count = df_clean[col].isnull().sum()
    df_clean[col].fillna(mode_val, inplace=True)
    print(f"   ✅ {col}: {missing_count} missing → filled with mode ({mode_val})")

print(f"\n📌 Missing values AFTER cleaning: {df_clean.isnull().sum().sum()} ✅")

# ────────────────────────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 : FEATURE ENGINEERING")
print("=" * 60)

# Total income = applicant + co-applicant
df_clean['Total_Income'] = df_clean['ApplicantIncome'] + df_clean['CoapplicantIncome']
print("   ✅ Total_Income = ApplicantIncome + CoapplicantIncome")

# Log of total income (reduces skew caused by outliers)
df_clean['Log_Total_Income'] = np.log1p(df_clean['Total_Income'])
print("   ✅ Log_Total_Income = log(Total_Income+1)  [reduces income skew]")

# EMI = LoanAmount / Loan_Amount_Term  (monthly payment proxy)
df_clean['EMI'] = df_clean['LoanAmount'] / df_clean['Loan_Amount_Term']
print("   ✅ EMI = LoanAmount ÷ Loan_Amount_Term  [monthly payment estimate]")

# Debt-to-Income ratio: EMI relative to income
df_clean['Debt_Income_Ratio'] = df_clean['EMI'] / (df_clean['Total_Income'] + 1)
print("   ✅ Debt_Income_Ratio = EMI ÷ Total_Income  [repayment burden]")

# Income per dependent (financial stress indicator)
df_clean['Income_Per_Dependent'] = df_clean['Total_Income'] / (df_clean['Dependents'] + 1)
print("   ✅ Income_Per_Dependent = Total_Income ÷ (Dependents+1)  [financial stress]")

# Loan amount to income ratio
df_clean['Loan_Income_Ratio'] = df_clean['LoanAmount'] / (df_clean['Total_Income'] + 1)
print("   ✅ Loan_Income_Ratio = LoanAmount ÷ Total_Income  [loan burden]")

# ────────────────────────────────────────────────────────────────
# STEP 5 — ENCODE TARGET + DEFINE FEATURES
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5 : PREPARE FEATURES & TARGET")
print("=" * 60)

df_clean['Loan_Status'] = (df_clean['Loan_Status'] == 'Y').astype(int)

categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
numeric_cols = [
    'Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History',
    'Total_Income', 'Log_Total_Income', 'EMI',
    'Debt_Income_Ratio', 'Income_Per_Dependent', 'Loan_Income_Ratio'
]

feature_cols = categorical_cols + numeric_cols
X = df_clean[feature_cols]
y = df_clean['Loan_Status']

print(f"\n📌 Total features : {len(feature_cols)}")
print(f"   Categorical    : {len(categorical_cols)} → {categorical_cols}")
print(f"   Numerical      : {len(numeric_cols)}")
print(f"\n📌 Target distribution: Approved={y.sum()}, Rejected={len(y)-y.sum()}")

# ────────────────────────────────────────────────────────────────
# STEP 6 — TRAIN/TEST SPLIT
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6 : TRAIN / TEST SPLIT  (80/20, stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\n📌 Training set : {X_train.shape[0]} samples")
print(f"📌 Test set     : {X_test.shape[0]} samples")

# ────────────────────────────────────────────────────────────────
# STEP 7 — BUILD PREPROCESSING PIPELINE
# ────────────────────────────────────────────────────────────────
numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    RobustScaler()
)
categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# ────────────────────────────────────────────────────────────────
# STEP 8 — TRAIN 4 MODELS
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 8 : TRAINING 4 MODELS")
print("=" * 60)

models = {
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4,
        subsample=0.8, min_samples_split=10, random_state=42),
    'random_forest': RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight='balanced', random_state=42),
    'logistic_regression': LogisticRegression(
        C=0.5, class_weight='balanced', max_iter=1000, random_state=42),
    'knn_classifier': KNeighborsClassifier(
        n_neighbors=11, weights='distance', metric='euclidean')
}

model_names = {
    'gradient_boosting': 'Gradient Boosting',
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
    'knn_classifier': 'KNN Classifier'
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for key, clf in models.items():
    print(f"\n   Training {model_names[key]}...")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', clf)
    ])
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)
    results[key] = {
        'accuracy': round(acc, 4),
        'auc': round(auc, 4),
        'f1': round(f1, 4),
        'cv_auc_mean': round(float(cv_scores.mean()), 4),
        'cv_auc_std': round(float(cv_scores.std()), 4)
    }
    joblib.dump(pipeline, f'model_{key}.pkl')
    print(f"   ✅ Accuracy: {acc*100:.2f}%  |  AUC: {auc:.4f}  |  F1: {f1:.4f}  |  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(classification_report(y_test, y_pred, target_names=['Rejected(0)','Approved(1)']))

# ────────────────────────────────────────────────────────────────
# STEP 9 — FEATURE DEPENDENCY ANALYSIS
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 9 : FEATURE DEPENDENCY ANALYSIS")
print("=" * 60)
print("   Dropping each feature one-at-a-time and measuring AUC drop...")

# Baseline AUC per model (already computed above)
baseline_aucs = {k: results[k]['auc'] for k in models}

feature_dependency = {}
for feat in feature_cols:
    remaining = [f for f in feature_cols if f != feat]
    X_tr_drop = X_train[remaining]
    X_te_drop  = X_test[remaining]

    num_remaining = [c for c in numeric_cols if c in remaining]
    cat_remaining = [c for c in categorical_cols if c in remaining]
    num_pipe_drop = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
    cat_pipe_drop = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    prep_drop = ColumnTransformer(transformers=[
        ("num", num_pipe_drop, num_remaining),
        ("cat", cat_pipe_drop, cat_remaining)
    ])

    drops = {}
    for key, clf_class in [
        ('gradient_boosting', GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)),
        ('random_forest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
        ('logistic_regression', LogisticRegression(C=0.5, class_weight='balanced', max_iter=500, random_state=42)),
        ('knn_classifier', KNeighborsClassifier(n_neighbors=11, weights='distance'))
    ]:
        pipe_drop = Pipeline([('preprocessor', prep_drop), ('model', clf_class)])
        pipe_drop.fit(X_tr_drop, y_train)
        prob_drop = pipe_drop.predict_proba(X_te_drop)[:, 1]
        auc_drop  = roc_auc_score(y_test, prob_drop)
        drops[key] = round(baseline_aucs[key] - auc_drop, 4)

    avg_drop = round(np.mean(list(drops.values())), 4)
    feature_dependency[feat] = {'drops': drops, 'avg_drop': avg_drop}
    print(f"   {feat:<28}: avg AUC drop = {avg_drop:+.4f}")

# Sort by avg drop descending
sorted_features = sorted(feature_dependency.items(), key=lambda x: x[1]['avg_drop'], reverse=True)

# ────────────────────────────────────────────────────────────────
# STEP 10 — SAVE METADATA
# ────────────────────────────────────────────────────────────────
meta = {
    'feature_cols': feature_cols,
    'categorical_cols': categorical_cols,
    'numeric_cols': numeric_cols,
    'models': results,
    'feature_dependency': {k: v for k, v in sorted_features},
    'dataset_file': 'loan_data.csv',
    'dataset_source': 'Loan Prediction Dataset — Banking',
    'train_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'total_features': len(feature_cols)
}
json.dump(meta, open('model_meta.json', 'w'), indent=2)
print("\n✅ All 4 models saved as .pkl files")
print("✅ model_meta.json saved")
print("\n" + "=" * 60)
print("  ALL DONE — Run: python3 app.py")
print("=" * 60)
