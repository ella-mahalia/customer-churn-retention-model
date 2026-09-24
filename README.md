# Customer Churn Prediction & Retention Modeling

## Project Overview

Customer churn is an important business problem because losing existing customers can directly affect recurring revenue and long-term growth. Instead of treating every customer the same, companies can use customer behavior and account information to identify which customers may be more likely to leave.

For this project, I used the public **IBM Telco Customer Churn dataset**, containing **7,043 customer records**, to build a machine learning model that estimates a customer's probability of churning.

The final project also includes an interactive web application where a user can enter a customer profile and receive a predicted churn probability, risk level, and suggested retention action.

## Why This Matters

A churn model can help a business move from reactive customer retention to a more proactive approach.

Rather than waiting until a customer cancels service, a company could use predicted churn risk to help:

- Prioritize customers for retention outreach
- Identify patterns associated with customer attrition
- Focus retention resources on higher-risk customers
- Better understand how contracts, pricing, tenure, and services relate to churn

The goal is not just to predict whether someone will leave, but to turn that prediction into information that can support business decisions.

## What I Found

The dataset had an overall churn rate of approximately **26.5%**.

I compared three classification models:

- Logistic Regression
- Random Forest
- Gradient Boosting

The selected **Logistic Regression model** achieved a test **ROC-AUC of 0.846** and a recall of approximately **79.4%**.

Recall was especially useful for this project because it measures how well the model identifies customers who actually churned. In a retention setting, missing a large number of high-risk customers could mean losing opportunities to intervene.

The analysis also showed that some of the most important predictors included:

- **Tenure**
- **Internet service**
- **Contract type**
- **Monthly charges**
- **Total charges**

Tenure had the strongest permutation importance in the selected model, suggesting that the length of the customer relationship contained particularly useful information when predicting churn.

## Project Outcome

This project demonstrates the full workflow of turning customer data into a usable machine learning application:

**Data preparation → Model development → Model evaluation → Churn prediction → Business interpretation**

The final application allows customer information to be entered directly into the model and returns:

- Predicted churn probability
- Low, medium, or high churn risk
- A recommended retention action

## Tools Used

**Python · Pandas · scikit-learn · Machine Learning · Data Analysis · HTML/CSS/JavaScript · Vercel**

## Dataset

IBM Telco Customer Churn public sample dataset.

This project was created for educational and portfolio purposes.
