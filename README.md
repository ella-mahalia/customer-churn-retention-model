# Customer Churn Prediction & Retention Modeling

A portfolio-ready machine-learning project built on IBM's public **Telco Customer Churn** sample dataset.

## Why this version is stronger

This version replaces the earlier synthetic dataset with the real public IBM sample. It includes:

- 7,043 customer records
- 21 raw columns
- real data cleaning (`TotalCharges` contains blank values)
- exploratory churn analysis
- Logistic Regression
- Random Forest
- Gradient Boosting
- stratified 5-fold cross-validation
- ROC-AUC, precision, recall, F1, accuracy
- model-agnostic permutation importance
- Low / Medium / High risk segmentation
- Vercel-ready interactive prediction page

## First-time setup

From the project folder:

```bash
chmod +x bootstrap_real_data.sh
./bootstrap_real_data.sh
```

That will:

1. install Python dependencies;
2. download the real IBM CSV;
3. train and compare the models;
4. save the selected model to `model/best_churn_model.joblib`;
5. create model metrics, feature importance, and scored test customers.

## Then push to GitHub

```bash
git add .
git commit -m "Replace synthetic churn data with IBM real dataset"
git push
```

Vercel will redeploy automatically if the repository is already connected.

## Dataset

See `DATA_SOURCE.md`.

Source: IBM Telco Customer Churn sample data, retrieved from the archived IBM GitHub repository.

## Important portfolio framing

Describe this as a **public IBM sample dataset**, not as private company customer data.

The project demonstrates customer churn classification and retention prioritization. It is a portfolio/educational analysis and is not intended to make automated real-world customer decisions.
