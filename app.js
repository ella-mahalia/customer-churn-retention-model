
const $ = id => document.getElementById(id);
const form = $("form");
const ids = [
  "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
  "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
  "DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract",
  "PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"
];

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = form.querySelector("button");
  const old = button.textContent;
  button.disabled = true;
  button.textContent = "Running model...";

  const payload = {};
  ids.forEach(id => payload[id] = $(id).value);
  ["SeniorCitizen","tenure"].forEach(id => payload[id] = Number(payload[id]));
  ["MonthlyCharges","TotalCharges"].forEach(id => payload[id] = Number(payload[id]));

  try {
    const res = await fetch("/api/predict", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)
    });
    const data = await res.json();
    if(!res.ok) throw new Error(data.error || "Prediction failed");

    const pct = data.churn_probability * 100;
    $("probability").textContent = pct.toFixed(1) + "%";
    $("risk").textContent = data.risk_segment + " risk";
    $("fill").style.width = Math.max(0, Math.min(100, pct)) + "%";
    $("action").textContent = data.recommended_action;
    $("result").classList.remove("hidden");
  } catch(err) {
    $("probability").textContent = "Unavailable";
    $("risk").textContent = "Model error";
    $("fill").style.width = "0%";
    $("action").textContent = err.message;
    $("result").classList.remove("hidden");
  } finally {
    button.disabled = false;
    button.textContent = old;
  }
});
