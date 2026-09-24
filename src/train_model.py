
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "subscription_churn.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

categorical = [
    "contract_type", "payment_method", "autopay",
    "discount_active", "region"
]

numeric = [
    "tenure_months", "monthly_charge", "support_tickets_90d",
    "weekly_usage_hours", "num_products", "late_payments_12m",
    "satisfaction_score"
]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric),
    ("cat", categorical_pipe, categorical)
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

results = {}
best_name = None
best_auc = -1
best_pipeline = None

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }
    results[name] = metrics

    if metrics["roc_auc"] > best_auc:
        best_auc = metrics["roc_auc"]
        best_name = name
        best_pipeline = pipe

with open(OUT_DIR / "model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

joblib.dump(best_pipeline, OUT_DIR / "best_churn_model.joblib")

# Save scored customers for dashboard/demo use
scored = X_test.copy()
scored["actual_churn"] = y_test.values
scored["churn_probability"] = best_pipeline.predict_proba(X_test)[:, 1]
scored["risk_segment"] = pd.cut(
    scored["churn_probability"],
    bins=[-0.01, 0.35, 0.65, 1.0],
    labels=["Low", "Medium", "High"]
)
scored.to_csv(OUT_DIR / "scored_customers.csv", index=False)

print(f"Best model: {best_name} | ROC-AUC: {best_auc:.4f}")
print(json.dumps(results, indent=2))
