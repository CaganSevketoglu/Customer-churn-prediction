# Customer Churn Prediction for a Telecom Company

## Project Overview

This project aims to predict customer churn for a fictional telecommunications company. By analyzing historical customer data, a machine learning model was developed to identify customers who are likely to cancel their subscriptions. The final model, a Logistic Regression classifier, achieved an accuracy of approximately 81.3%. This predictive model can help the company proactively target at-risk customers to reduce churn and retain revenue.

---

## Dataset

The dataset used for this project is the "Telco Customer Churn" dataset, publicly available on Kaggle. It contains data for over 7,000 customers and 21 features, including customer demographics, subscribed services, and account information.

- **Data Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Methodology

The project followed a standard data science pipeline:

1.  **Data Loading and Exploration:** The data was loaded into a Pandas DataFrame. Initial analysis was performed using `.info()`, `.describe()`, and `.value_counts()` to understand the data structure, types, and distributions.
2.  **Data Preprocessing and Cleaning:**
    - The `TotalCharges` column was converted to a numeric type, and 11 missing values were imputed with the column mean.
    - Binary categorical features (e.g., 'Yes'/'No') were encoded into `1`s and `0`s.
    - Multi-category features were transformed into numerical format using One-Hot Encoding (`pd.get_dummies`).
    - The non-predictive `customerID` column was dropped.
3.  **Feature Scaling:** All features were scaled using `StandardScaler` to ensure the model performs optimally without being biased by features with large value ranges. This step was crucial to prevent convergence warnings.
4.  **Model Training:** The data was split into training (75%) and testing (25%) sets. A **Logistic Regression** model was trained on the scaled training data.
5.  **Model Evaluation:** The model's performance was evaluated on the unseen test data using accuracy and a confusion matrix to analyze its predictive power and error types.

---

## Results

- **Accuracy:** The model achieved an accuracy of **81.32%** on the test set.
- **Confusion Matrix:**

[[1154  128]

[ 201  278]]

- **Interpretation:** The model is highly effective at identifying customers who will *not* churn (1154 True Negatives). It also successfully identified 278 customers who would churn (True Positives). The main area for improvement is reducing the 201 False Negatives, which represent at-risk customers the model failed to identify.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Git & GitHub
