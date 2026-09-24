const $ = (id) => document.getElementById(id);

const form = $("form");

const fields = [
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
];


/* =========================================================
   FORM SUBMIT
========================================================= */

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const button = form.querySelector(".predict-button");

  const originalButton = button.innerHTML;

  button.disabled = true;

  button.innerHTML = `
    Running model...
    <span>•••</span>
  `;


  /* ---------------------------------------------------------
     BUILD PAYLOAD
  --------------------------------------------------------- */

  const payload = {};

  fields.forEach((field) => {
    payload[field] = $(field).value;
  });


  payload.SeniorCitizen =
    Number(payload.SeniorCitizen);

  payload.tenure =
    Number(payload.tenure);

  payload.MonthlyCharges =
    Number(payload.MonthlyCharges);

  payload.TotalCharges =
    Number(payload.TotalCharges);


  /* ---------------------------------------------------------
     START LOADING UI
  --------------------------------------------------------- */

  let currentProgress = 4;

  renderLoading(currentProgress);

  const progressTimer =
    startProgressAnimation((progress) => {
      currentProgress = progress;

      updateLoadingProgress(progress);
    });


  /* ---------------------------------------------------------
     API REQUEST
  --------------------------------------------------------- */

  const controller =
    new AbortController();

  const timeout =
    setTimeout(() => {
      controller.abort();
    }, 30000);


  try {

    const response =
      await fetch("/api/predict", {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(payload),

        signal:
          controller.signal,
      });


    const rawResponse =
      await response.text();


    let data;


    try {

      data =
        JSON.parse(rawResponse);

    } catch {

      throw new Error(
        rawResponse ||
        "The prediction API returned an invalid response."
      );

    }


    if (!response.ok) {

      throw new Error(
        data.error ||
        `Prediction request failed with status ${response.status}.`
      );

    }


    /* ---------------------------------------------------------
       FINISH AT 100%
    --------------------------------------------------------- */

    clearInterval(progressTimer);

    updateLoadingProgress(100);


    await delay(450);


    renderResult(data);


  } catch (error) {

    clearInterval(progressTimer);


    if (error.name === "AbortError") {

      renderError(
        "The model took too long to respond. Check the Vercel terminal for the Python API error."
      );

    } else {

      renderError(
        error.message ||
        "The prediction could not be completed."
      );

    }

  } finally {

    clearTimeout(timeout);

    button.disabled =
      false;

    button.innerHTML =
      originalButton;

  }

});


/* =========================================================
   LOADING SCREEN
========================================================= */

function renderLoading(progress) {

  const result =
    $("result");


  result.className =
    "result";


  result.innerHTML = `
    <div class="loading-output">

      <div class="loading-model-icon">
        <span></span>
      </div>

      <span class="loading-label">
        RUNNING MODEL
      </span>

      <strong
        class="loading-percentage"
        id="loading-percentage"
      >
        ${Math.round(progress)}%
      </strong>

      <div class="loading-progress-track">

        <div
          class="loading-progress-fill"
          id="loading-progress-fill"
          style="width: ${progress}%"
        ></div>

      </div>

      <p
        class="loading-status"
        id="loading-status"
      >
        Preparing customer data...
      </p>

    </div>
  `;

}


/* =========================================================
   UPDATE LOADING %
========================================================= */

function updateLoadingProgress(progress) {

  const percentage =
    $("loading-percentage");

  const fill =
    $("loading-progress-fill");

  const status =
    $("loading-status");


  if (!percentage || !fill) {
    return;
  }


  const displayed =
    Math.min(
      100,
      Math.round(progress)
    );


  percentage.textContent =
    `${displayed}%`;


  fill.style.width =
    `${Math.min(progress, 100)}%`;


  if (!status) {
    return;
  }


  if (progress < 25) {

    status.textContent =
      "Preparing customer data...";

  }

  else if (progress < 55) {

    status.textContent =
      "Transforming model features...";

  }

  else if (progress < 80) {

    status.textContent =
      "Running churn classification...";

  }

  else if (progress < 97) {

    status.textContent =
      "Evaluating churn probability...";

  }

  else if (progress < 100) {

    status.textContent =
      "Finalizing prediction...";

  }

  else {

    status.textContent =
      "Prediction complete.";

  }

}


/* =========================================================
   PROGRESS ANIMATION
========================================================= */

function startProgressAnimation(onUpdate) {

  let progress = 4;


  return setInterval(() => {

    if (progress < 30) {

      progress +=
        randomBetween(4, 8);

    }

    else if (progress < 65) {

      progress +=
        randomBetween(2, 5);

    }

    else if (progress < 88) {

      progress +=
        randomBetween(1, 3);

    }

    else if (progress < 96) {

      progress +=
        randomBetween(
          0.5,
          1.5
        );

    }

    else if (progress < 98.5) {

      progress +=
        randomBetween(
          0.15,
          0.45
        );

    }

    else {

      /*
        Hold around 99%.
        Do NOT show 100%
        until the server actually responds.
      */

      progress =
        Math.min(
          progress + 0.05,
          99
        );

    }


    progress =
      Math.min(
        progress,
        99
      );


    onUpdate(progress);

  }, 180);

}


/* =========================================================
   SHOW MODEL RESULT
========================================================= */

function renderResult(data) {

  const result =
    $("result");


  const probability =
    Number(
      data.churn_probability
    ) * 100;


  const risk =
    data.risk_segment ||
    "Unknown";


  let riskClass =
    "risk-medium";


  if (
    risk.toLowerCase() ===
    "high"
  ) {

    riskClass =
      "risk-high";

  }

  else if (
    risk.toLowerCase() ===
    "low"
  ) {

    riskClass =
      "risk-low";

  }


  result.className =
    "result";


  result.innerHTML = `

    <div class="result-output">

      <span class="result-output-label">
        PREDICTED CHURN PROBABILITY
      </span>


      <strong class="result-probability">
        ${probability.toFixed(1)}%
      </strong>


      <div
        class="
          risk-label
          ${riskClass}
        "
      >
        ${risk.toUpperCase()} RISK
      </div>


      <div class="result-meter">

        <div
          style="
            width:
            ${Math.min(
              100,
              Math.max(
                0,
                probability
              )
            )}%
          "
        ></div>

      </div>


      <div class="result-divider"></div>


      <span class="result-action-label">
        RECOMMENDED ACTION
      </span>


      <p class="result-action">
        ${
          data.recommended_action ||
          "Review the customer profile and determine whether retention outreach is appropriate."
        }
      </p>


      <button
        type="button"
        class="reset-result-button"
        onclick="resetPrediction()"
      >
        Run another prediction
      </button>

    </div>

  `;

}


/* =========================================================
   ERROR
========================================================= */

function renderError(message) {

  const result =
    $("result");


  result.className =
    "result";


  result.innerHTML = `

    <div class="error-output">

      <div class="error-icon">
        !
      </div>


      <span class="loading-label">
        PREDICTION ERROR
      </span>


      <h3>
        Model unavailable
      </h3>


      <p>
        ${escapeHTML(message)}
      </p>


      <button
        type="button"
        class="reset-result-button"
        onclick="resetPrediction()"
      >
        Try again
      </button>

    </div>

  `;

}


/* =========================================================
   RESET PANEL
========================================================= */

function resetPrediction() {

  const result =
    $("result");


  result.className =
    "result empty-result";


  result.innerHTML = `

    <div class="empty-state">

      <div class="empty-icon">
        ↗
      </div>


      <h3>
        Ready to predict
      </h3>


      <p>
        Complete the customer profile
        and run the model to generate
        a churn probability.
      </p>

    </div>

  `;

}


/* =========================================================
   HELPERS
========================================================= */

function randomBetween(
  min,
  max
) {

  return (
    Math.random() *
    (max - min) +
    min
  );

}


function delay(ms) {

  return new Promise(
    (resolve) =>
      setTimeout(
        resolve,
        ms
      )
  );

}


function escapeHTML(value) {

  return String(value)

    .replaceAll(
      "&",
      "&amp;"
    )

    .replaceAll(
      "<",
      "&lt;"
    )

    .replaceAll(
      ">",
      "&gt;"
    )

    .replaceAll(
      '"',
      "&quot;"
    )

    .replaceAll(
      "'",
      "&#039;"
    );

}