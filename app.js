
const form = document.getElementById("churn-form");
const result = document.getElementById("result");

const value = (id) => document.getElementById(id).value;
const num = (id) => Number(value(id));

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const button = form.querySelector("button");
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Running model...";

  const payload = {
    tenure_months: num("tenure_months"),
    monthly_charge: num("monthly_charge"),
    contract_type: value("contract_type"),
    payment_method: value("payment_method"),
    autopay: value("autopay"),
    support_tickets_90d: num("support_tickets_90d"),
    weekly_usage_hours: num("weekly_usage_hours"),
    num_products: num("num_products"),
    late_payments_12m: num("late_payments_12m"),
    discount_active: value("discount_active"),
    satisfaction_score: num("satisfaction_score"),
    region: value("region"),
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed");

    const pct = data.churn_probability * 100;
    document.getElementById("probability").textContent = `${pct.toFixed(1)}%`;
    document.getElementById("risk-badge").textContent = `${data.risk_segment} risk`;
    document.getElementById("meter-fill").style.width = `${Math.min(100, Math.max(0, pct))}%`;
    document.getElementById("action").textContent = data.recommended_action;
    result.classList.remove("hidden");
  } catch (error) {
    result.classList.remove("hidden");
    document.getElementById("probability").textContent = "Error";
    document.getElementById("risk-badge").textContent = "Unavailable";
    document.getElementById("meter-fill").style.width = "0%";
    document.getElementById("action").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
});
