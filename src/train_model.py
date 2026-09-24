
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, confusion_matrix
)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Real dataset not found. Run `python src/download_data.py` first."
    )

df = pd.read_csv(DATA_PATH)

# Clean the IBM sample data.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})

# Keep customerID for analysis, but exclude it from modeling.
X = df.drop(columns=["customerID", "Churn", "ChurnFlag"])
y = df["ChurnFlag"]

numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical = [c for c in X.columns if c not in numeric]

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
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
trained = {}
best_name = None
best_auc = -1

for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    # Cross-validation on training data
    cv_scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist()
    }

    results[name] = metrics
    trained[name] = pipe

    if metrics["roc_auc"] > best_auc:
        best_auc = metrics["roc_auc"]
        best_name = name

best_model = trained[best_name]

# Model-agnostic importance on original input columns.
perm = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=8,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)

joblib.dump(best_model, MODEL_DIR / "best_churn_model.joblib")

with open(MODEL_DIR / "model_metrics.json", "w") as f:
    json.dump(
        {
            "dataset_rows": int(len(df)),
            "churn_rate": round(float(y.mean()), 4),
            "best_model": best_name,
            "models": results
        },
        f,
        indent=2
    )

importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

# Score the test set for portfolio analysis.
scored = df.loc[X_test.index, ["customerID"]].copy()
scored["actual_churn"] = y_test.values
scored["churn_probability"] = best_model.predict_proba(X_test)[:, 1]
scored["risk_segment"] = pd.cut(
    scored["churn_probability"],
    bins=[-0.01, 0.35, 0.65, 1.0],
    labels=["Low", "Medium", "High"]
)
scored.to_csv(MODEL_DIR / "scored_test_customers.csv", index=False)

print(f"Rows: {len(df):,}")
print(f"Churn rate: {y.mean():.1%}")
print(f"Best model: {best_name}")
print(f"Test ROC-AUC: {best_auc:.4f}")
print("\nModel comparison:")
print(json.dumps(results, indent=2))
print("\nTop permutation-importance features:")
print(importance_df.head(10).to_string(index=False))
