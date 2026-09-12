"""
CodeAlpha ML Internship - TASK 4: Disease Prediction from Medical Data
Datasets : Breast Cancer (REAL, sklearn/UCI) | Heart Disease & Diabetes (synthetic,
           built from real UCI feature schemas - no internet in this sandbox to
           fetch the actual UCI CSVs; swap-in loaders for the real files are below).
Algorithms: Logistic Regression, Random Forest, SVM, XGBoost (falls back to
           GradientBoosting if xgboost isn't installed).
Run: python3 task4_disease_prediction.py
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
try:
    from xgboost import XGBClassifier
    XGB = lambda: XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=42)
except ImportError:
    XGB = lambda: GradientBoostingClassifier(random_state=42)  # offline substitute

RNG = 42

# ---------- REAL dataset ----------
def load_breast_cancer_data():
    d = load_breast_cancer()
    return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target)

# ---------- SYNTHETIC (offline) datasets, mirroring real UCI schemas ----------
def load_heart_disease_data(n=600, seed=RNG):
    """UCI Heart Disease schema: age, sex, cp, trestbps, chol, fbs, thalach, exang, oldpeak."""
    rng = np.random.default_rng(seed)
    age = rng.integers(29, 77, n); sex = rng.integers(0, 2, n)
    chol = rng.normal(246, 52, n).clip(120, 450)
    trestbps = rng.normal(131, 17, n).clip(90, 200)
    thalach = rng.normal(150, 23, n).clip(70, 202)
    oldpeak = rng.exponential(1.0, n).clip(0, 6)
    exang = rng.integers(0, 2, n)
    risk = (0.03*age + 0.01*chol + 0.02*trestbps - 0.02*thalach + 5*oldpeak + 3*exang + rng.normal(0, 5, n))
    y = (risk > np.median(risk)).astype(int)
    X = pd.DataFrame({"age": age, "sex": sex, "chol": chol, "trestbps": trestbps,
                       "thalach": thalach, "oldpeak": oldpeak, "exang": exang})
    return X, pd.Series(y)

def load_diabetes_data(n=600, seed=RNG):
    """Pima Indians Diabetes schema: glucose, bmi, blood pressure, insulin, age."""
    rng = np.random.default_rng(seed)
    glucose = rng.normal(120, 32, n).clip(60, 250)
    bmi = rng.normal(32, 7, n).clip(15, 60)
    bp = rng.normal(70, 12, n).clip(40, 120)
    insulin = rng.exponential(80, n).clip(0, 500)
    age = rng.integers(21, 81, n)
    risk = (0.05*glucose + 0.1*bmi + 0.02*bp + 0.005*insulin + 0.03*age + rng.normal(0, 4, n))
    y = (risk > np.median(risk)).astype(int)
    X = pd.DataFrame({"glucose": glucose, "bmi": bmi, "blood_pressure": bp,
                       "insulin": insulin, "age": age})
    return X, pd.Series(y)

# To use REAL UCI files instead, once you have internet:
#   heart = pd.read_csv("heart.csv"); X, y = heart.drop(columns=["target"]), heart["target"]
#   diabetes = pd.read_csv("diabetes.csv"); X, y = diabetes.drop(columns=["Outcome"]), diabetes["Outcome"]

DATASETS = {
    "Breast Cancer (real)": load_breast_cancer_data,
    "Heart Disease (synthetic schema)": load_heart_disease_data,
    "Diabetes (synthetic schema)": load_diabetes_data,
}
MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=5000, random_state=RNG),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=RNG),
    "SVM": lambda: SVC(probability=True, random_state=RNG),
    "XGBoost": XGB,
}

all_results = []
for dname, loader in DATASETS.items():
    X, y = loader()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    print(f"\n=== {dname} | {X.shape[0]} patients, {X.shape[1]} features ===")
    for mname, mfactory in MODELS.items():
        model = mfactory().fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        proba = model.predict_proba(Xte_s)[:, 1]
        row = {"Dataset": dname, "Model": mname,
               "Accuracy": accuracy_score(yte, pred), "Precision": precision_score(yte, pred),
               "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred),
               "ROC-AUC": roc_auc_score(yte, proba),
               "CV_Acc": cross_val_score(mfactory(), scaler.transform(X), y, cv=5).mean()}
        all_results.append(row)
        print(f"  {mname:20s} Acc={row['Accuracy']:.3f} F1={row['F1']:.3f} AUC={row['ROC-AUC']:.3f} CV={row['CV_Acc']:.3f}")

results_df = pd.DataFrame(all_results)
results_df.to_csv("task4_all_results.csv", index=False)
print("\nSaved: task4_all_results.csv")

# Summary chart: F1-score per model per dataset
fig, ax = plt.subplots(figsize=(9, 5))
pivot = results_df.pivot(index="Model", columns="Dataset", values="F1")
pivot.plot(kind="bar", ax=ax)
ax.set_ylabel("F1-Score"); ax.set_title("Disease Prediction: F1-Score by Model & Dataset")
ax.set_ylim(0, 1.05); plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("task4_summary_chart.png", dpi=150)
print("Saved: task4_summary_chart.png")
print("\nBest overall (by F1):")
print(results_df.loc[results_df.groupby("Dataset")["F1"].idxmax(), ["Dataset", "Model", "F1", "ROC-AUC"]].to_string(index=False))
