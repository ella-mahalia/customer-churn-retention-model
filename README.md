# Customer Churn Prediction & Retention Modeling

Vercel-ready portfolio project with:
- static frontend
- Python serverless prediction API
- pre-trained scikit-learn model
- reproducible notebook and training script

## Deploy with GitHub + Vercel

1. Create a new GitHub repository.
2. Push this folder to the repository.
3. In Vercel, choose **Add New → Project**.
4. Import the GitHub repository.
5. Leave the root directory as `./`.
6. Use the default/Other framework settings.
7. Deploy.

Vercel serves the static frontend and deploys `api/predict.py` as a Python Function.

## Local model training

```bash
pip install -r requirements.txt
python src/train_model.py
```

The deployed model file is already included at:

```text
model/best_churn_model.joblib
```

## Dataset

The project uses synthetic subscription-business data for portfolio and educational use.
