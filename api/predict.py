from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "model"
    / "best_churn_model.joblib"
)


model = joblib.load(MODEL_PATH)


FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


class handler(BaseHTTPRequestHandler):

    def send_json(
        self,
        status,
        payload
    ):

        body = json.dumps(
            payload
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(body)


    def do_GET(self):

        self.send_json(
            200,
            {
                "status": "ok",
                "message":
                    "Customer churn prediction API is running"
            }
        )


    def do_POST(self):

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            raw_body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw_body.decode(
                    "utf-8"
                )
            )


            missing = [
                field
                for field in FEATURES
                if field not in payload
            ]


            if missing:

                self.send_json(
                    400,
                    {
                        "error":
                            "Missing fields: "
                            + ", ".join(missing)
                    }
                )

                return


            row = pd.DataFrame(
                [
                    {
                        "gender":
                            str(
                                payload[
                                    "gender"
                                ]
                            ),

                        "SeniorCitizen":
                            int(
                                payload[
                                    "SeniorCitizen"
                                ]
                            ),

                        "Partner":
                            str(
                                payload[
                                    "Partner"
                                ]
                            ),

                        "Dependents":
                            str(
                                payload[
                                    "Dependents"
                                ]
                            ),

                        "tenure":
                            int(
                                payload[
                                    "tenure"
                                ]
                            ),

                        "PhoneService":
                            str(
                                payload[
                                    "PhoneService"
                                ]
                            ),

                        "MultipleLines":
                            str(
                                payload[
                                    "MultipleLines"
                                ]
                            ),

                        "InternetService":
                            str(
                                payload[
                                    "InternetService"
                                ]
                            ),

                        "OnlineSecurity":
                            str(
                                payload[
                                    "OnlineSecurity"
                                ]
                            ),

                        "OnlineBackup":
                            str(
                                payload[
                                    "OnlineBackup"
                                ]
                            ),

                        "DeviceProtection":
                            str(
                                payload[
                                    "DeviceProtection"
                                ]
                            ),

                        "TechSupport":
                            str(
                                payload[
                                    "TechSupport"
                                ]
                            ),

                        "StreamingTV":
                            str(
                                payload[
                                    "StreamingTV"
                                ]
                            ),

                        "StreamingMovies":
                            str(
                                payload[
                                    "StreamingMovies"
                                ]
                            ),

                        "Contract":
                            str(
                                payload[
                                    "Contract"
                                ]
                            ),

                        "PaperlessBilling":
                            str(
                                payload[
                                    "PaperlessBilling"
                                ]
                            ),

                        "PaymentMethod":
                            str(
                                payload[
                                    "PaymentMethod"
                                ]
                            ),

                        "MonthlyCharges":
                            float(
                                payload[
                                    "MonthlyCharges"
                                ]
                            ),

                        "TotalCharges":
                            float(
                                payload[
                                    "TotalCharges"
                                ]
                            ),
                    }
                ]
            )


            probability = float(
                model.predict_proba(
                    row
                )[0, 1]
            )


            if probability >= 0.65:

                risk = "High"

                action = (
                    "Prioritize retention outreach "
                    "and review contract, service, "
                    "and billing friction."
                )


            elif probability >= 0.35:

                risk = "Medium"

                action = (
                    "Monitor engagement and consider "
                    "proactive retention messaging."
                )


            else:

                risk = "Low"

                action = (
                    "No immediate retention intervention "
                    "is indicated by the model."
                )


            self.send_json(
                200,
                {
                    "churn_probability":
                        probability,

                    "risk_segment":
                        risk,

                    "recommended_action":
                        action,
                }
            )


        except Exception as error:

            self.send_json(
                500,
                {
                    "error":
                        str(error)
                }
            )