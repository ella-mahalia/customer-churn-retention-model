
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "best_churn_model.joblib"

MODEL = None
MODEL_ERROR = None

try:
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
    else:
        MODEL_ERROR = "Model file is missing. Download the real IBM dataset and run src/train_model.py before deploying."
except Exception as exc:
    MODEL_ERROR = str(exc)

FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {"ok": True})

    def do_POST(self):
        if MODEL is None:
            return self._send(503, {"error": MODEL_ERROR or "Model unavailable"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

            missing = [f for f in FEATURES if f not in payload]
            if missing:
                return self._send(400, {"error": f"Missing fields: {', '.join(missing)}"})

            row = pd.DataFrame([{
                "gender": str(payload["gender"]),
                "SeniorCitizen": int(payload["SeniorCitizen"]),
                "Partner": str(payload["Partner"]),
                "Dependents": str(payload["Dependents"]),
                "tenure": int(payload["tenure"]),
                "PhoneService": str(payload["PhoneService"]),
                "MultipleLines": str(payload["MultipleLines"]),
                "InternetService": str(payload["InternetService"]),
                "OnlineSecurity": str(payload["OnlineSecurity"]),
                "OnlineBackup": str(payload["OnlineBackup"]),
                "DeviceProtection": str(payload["DeviceProtection"]),
                "TechSupport": str(payload["TechSupport"]),
                "StreamingTV": str(payload["StreamingTV"]),
                "StreamingMovies": str(payload["StreamingMovies"]),
                "Contract": str(payload["Contract"]),
                "PaperlessBilling": str(payload["PaperlessBilling"]),
                "PaymentMethod": str(payload["PaymentMethod"]),
                "MonthlyCharges": float(payload["MonthlyCharges"]),
                "TotalCharges": float(payload["TotalCharges"])
            }])

            probability = float(MODEL.predict_proba(row)[0, 1])

            if probability >= 0.65:
                risk = "High"
                action = "Prioritize retention outreach and review contract, service, and billing friction."
            elif probability >= 0.35:
                risk = "Medium"
                action = "Monitor engagement and consider proactive retention messaging."
            else:
                risk = "Low"
                action = "No immediate retention intervention is indicated by the model."

            self._send(200, {
                "churn_probability": probability,
                "risk_segment": risk,
                "recommended_action": action
            })
        except Exception as exc:
            self._send(500, {"error": str(exc)})
