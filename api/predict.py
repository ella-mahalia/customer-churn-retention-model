
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "best_churn_model.joblib"
model = joblib.load(MODEL_PATH)

class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(200, {"ok": True})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))

            required = [
                "tenure_months",
                "monthly_charge",
                "contract_type",
                "payment_method",
                "autopay",
                "support_tickets_90d",
                "weekly_usage_hours",
                "num_products",
                "late_payments_12m",
                "discount_active",
                "satisfaction_score",
                "region",
            ]

            missing = [field for field in required if field not in data]
            if missing:
                return self._send(400, {"error": f"Missing fields: {', '.join(missing)}"})

            row = pd.DataFrame([{
                "tenure_months": int(data["tenure_months"]),
                "monthly_charge": float(data["monthly_charge"]),
                "contract_type": str(data["contract_type"]),
                "payment_method": str(data["payment_method"]),
                "autopay": str(data["autopay"]),
                "support_tickets_90d": int(data["support_tickets_90d"]),
                "weekly_usage_hours": float(data["weekly_usage_hours"]),
                "num_products": int(data["num_products"]),
                "late_payments_12m": int(data["late_payments_12m"]),
                "discount_active": str(data["discount_active"]),
                "satisfaction_score": int(data["satisfaction_score"]),
                "region": str(data["region"]),
            }])

            probability = float(model.predict_proba(row)[0, 1])

            if probability >= 0.65:
                risk = "High"
                action = "Prioritize this customer for retention outreach and review recent support or billing friction."
            elif probability >= 0.35:
                risk = "Medium"
                action = "Monitor engagement and consider proactive communication before the next billing cycle."
            else:
                risk = "Low"
                action = "No immediate intervention is needed; continue standard engagement."

            self._send(200, {
                "churn_probability": probability,
                "risk_segment": risk,
                "recommended_action": action
            })

        except Exception as exc:
            self._send(500, {"error": str(exc)})
