# 📉 Customer Churn Analysis & Prediction

**Author:** Muhammad Waqas Waseem  
**GitHub:** [MuhammadWaqasds](https://github.com/MuhammadWaqasds)  
**Tools:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---

## 📌 Project Overview

An end-to-end data science project analyzing and predicting customer churn for a telecom company. Includes full EDA, machine learning classification models, ROC-AUC evaluation, and actionable business recommendations — the kind of project that directly maps to real DS internship work.

---

## 🎯 Problem Statement

Customer churn costs businesses millions annually. This project identifies which customers are likely to leave and **why**, enabling targeted retention strategies.

---

## 📊 Features Analyzed

| Feature | Description |
|---|---|
| Tenure_Months | How long the customer has been with the company |
| Monthly_Charges_PKR | Monthly bill in PKR |
| Total_Charges_PKR | Total amount paid |
| Num_Services | Number of subscribed services |
| Contract_Type | Month-to-month / 1yr / 2yr |
| Tech_Support | Whether customer has tech support |
| Num_Complaints | Number of complaints filed |
| Senior_Citizen | Whether customer is a senior citizen |

---

## 🤖 Models Used

- **Logistic Regression** — interpretable baseline
- **Random Forest Classifier** — best performing model

---

## 📈 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~78% | ~0.84 |
| Random Forest | ~85% | ~0.92 |

---

## 💡 Key Business Insights

1. **Month-to-month** contract customers churn 3x more than 2-year contract customers
2. Customers with **5+ complaints** have over 70% churn probability
3. **Short tenure** (under 12 months) is the strongest predictor of churn
4. **Tech support** subscribers have 20% lower churn rate

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the analysis
python customer_churn_analysis.py
```

---

## 📁 Project Structure

```
project2_churn/
│
├── customer_churn_analysis.py  # Full ML pipeline
├── churn_eda.png               # EDA visualizations
├── roc_curve.png               # Model ROC curve
└── README.md                   # Project documentation
```

---

## 🛠️ Skills Demonstrated

- Business problem framing
- Exploratory Data Analysis (EDA)
- Binary classification (Logistic Regression + Random Forest)
- Model evaluation (Accuracy, ROC-AUC, Confusion Matrix)
- Business insight generation from ML results
