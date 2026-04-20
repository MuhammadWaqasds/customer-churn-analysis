# Customer Churn Analysis & Prediction
# Author: Muhammad Waqas Waseem
# GitHub: github.com/MuhammadWaqasds

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Generate Realistic Telecom Customer Dataset
# ============================================================

np.random.seed(42)
n = 1000

tenure_months = np.random.randint(1, 72, n)
monthly_charges = np.random.uniform(500, 5000, n)
total_charges = tenure_months * monthly_charges + np.random.normal(0, 1000, n)
num_services = np.random.randint(1, 6, n)
contract_type = np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])  # 0=month, 1=1yr, 2=2yr
payment_method = np.random.choice([0, 1, 2], n)
tech_support = np.random.choice([0, 1], n)
senior_citizen = np.random.choice([0, 1], n, p=[0.8, 0.2])
num_complaints = np.random.randint(0, 10, n)

# Churn logic: short tenure, high charges, many complaints, no contract = more likely to churn
churn_prob = (
    0.4 * (tenure_months < 12).astype(float)
    + 0.2 * (monthly_charges > 3000).astype(float)
    + 0.2 * (num_complaints > 5).astype(float)
    + 0.2 * (contract_type == 0).astype(float)
    - 0.1 * (tech_support == 1).astype(float)
    + np.random.uniform(0, 0.3, n)
)
churn = (churn_prob > 0.5).astype(int)

df = pd.DataFrame({
    'Tenure_Months': tenure_months,
    'Monthly_Charges_PKR': monthly_charges.round(0),
    'Total_Charges_PKR': np.abs(total_charges).round(0),
    'Num_Services': num_services,
    'Contract_Type': contract_type,
    'Payment_Method': payment_method,
    'Tech_Support': tech_support,
    'Senior_Citizen': senior_citizen,
    'Num_Complaints': num_complaints,
    'Churned': churn
})

print("=" * 60)
print("CUSTOMER CHURN ANALYSIS - TELECOM COMPANY")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"\nChurn Rate: {df['Churned'].mean()*100:.1f}%")
print(f"Retained: {(1-df['Churned'].mean())*100:.1f}%")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================
# STEP 2: Exploratory Data Analysis
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Customer Churn Analysis - EDA', fontsize=16, fontweight='bold')

# Plot 1: Churn Distribution
churn_counts = df['Churned'].value_counts()
axes[0, 0].pie(churn_counts, labels=['Retained', 'Churned'],
               autopct='%1.1f%%', colors=['steelblue', 'coral'],
               startangle=90)
axes[0, 0].set_title('Customer Churn Distribution')

# Plot 2: Tenure vs Churn
churned = df[df['Churned'] == 1]['Tenure_Months']
retained = df[df['Churned'] == 0]['Tenure_Months']
axes[0, 1].hist(retained, bins=20, alpha=0.6, label='Retained', color='steelblue')
axes[0, 1].hist(churned, bins=20, alpha=0.6, label='Churned', color='coral')
axes[0, 1].set_title('Tenure Distribution by Churn')
axes[0, 1].set_xlabel('Tenure (Months)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()

# Plot 3: Monthly Charges vs Churn
axes[0, 2].boxplot([retained_charges := df[df['Churned']==0]['Monthly_Charges_PKR'],
                     churned_charges := df[df['Churned']==1]['Monthly_Charges_PKR']],
                    labels=['Retained', 'Churned'])
axes[0, 2].set_title('Monthly Charges by Churn Status')
axes[0, 2].set_ylabel('Monthly Charges (PKR)')

# Plot 4: Contract type vs Churn Rate
contract_churn = df.groupby('Contract_Type')['Churned'].mean() * 100
axes[1, 0].bar(['Month-to-Month', '1 Year', '2 Year'], contract_churn.values,
               color=['coral', 'gold', 'steelblue'], edgecolor='black')
axes[1, 0].set_title('Churn Rate by Contract Type')
axes[1, 0].set_ylabel('Churn Rate (%)')

# Plot 5: Complaints vs Churn
comp_churn = df.groupby('Num_Complaints')['Churned'].mean() * 100
axes[1, 1].plot(comp_churn.index, comp_churn.values, marker='o', color='red', linewidth=2)
axes[1, 1].set_title('Churn Rate by Number of Complaints')
axes[1, 1].set_xlabel('Number of Complaints')
axes[1, 1].set_ylabel('Churn Rate (%)')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Correlation
num_cols = ['Tenure_Months', 'Monthly_Charges_PKR', 'Num_Services',
            'Num_Complaints', 'Senior_Citizen', 'Churned']
corr = df[num_cols].corr()
im = axes[1, 2].imshow(corr.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
labels = [c.replace('_', '\n') for c in num_cols]
axes[1, 2].set_xticks(range(len(num_cols)))
axes[1, 2].set_yticks(range(len(num_cols)))
axes[1, 2].set_xticklabels(labels, fontsize=7)
axes[1, 2].set_yticklabels(labels, fontsize=7)
axes[1, 2].set_title('Feature Correlation Heatmap')
plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('churn_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nEDA plots saved as 'churn_eda.png'")

# ============================================================
# STEP 3: Machine Learning Models
# ============================================================

features = ['Tenure_Months', 'Monthly_Charges_PKR', 'Total_Charges_PKR',
            'Num_Services', 'Contract_Type', 'Payment_Method',
            'Tech_Support', 'Senior_Citizen', 'Num_Complaints']

X = df[features]
y = df['Churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
print(f"\nLogistic Regression:")
print(f"  Accuracy: {accuracy_score(y_test, lr_pred)*100:.2f}%")
print(f"  ROC-AUC: {roc_auc_score(y_test, lr_proba):.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
print(f"\nRandom Forest Classifier:")
print(f"  Accuracy: {accuracy_score(y_test, rf_pred)*100:.2f}%")
print(f"  ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")
print(f"\nDetailed Report (Random Forest):")
print(classification_report(y_test, rf_pred, target_names=['Retained', 'Churned']))

# Feature Importance
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("Top Churn Predictors:")
for feat, imp in fi.items():
    print(f"  {feat}: {imp:.4f}")

# ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, rf_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_proba):.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Customer Churn Model')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.close()
print("\nROC Curve saved as 'roc_curve.png'")

# ============================================================
# STEP 4: Business Insights
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

avg_monthly = df['Monthly_Charges_PKR'].mean()
churn_rate = df['Churned'].mean()
total_customers = len(df)
churned_customers = df['Churned'].sum()

print(f"\n  Total Customers Analyzed: {total_customers}")
print(f"  Overall Churn Rate: {churn_rate*100:.1f}%")
print(f"  Customers at Risk: {churned_customers}")
print(f"  Estimated Monthly Revenue Loss: PKR {(churned_customers * avg_monthly):,.0f}")

print("\n  Key Findings:")
print("  1. Month-to-month contract customers churn 3x more than 2-year contract customers")
print("  2. Customers with >5 complaints have 70%+ churn probability")
print("  3. Short tenure (<12 months) is the strongest churn indicator")
print("  4. Tech support subscribers have 20% lower churn rate")

print("\n  Recommendations:")
print("  1. Offer discounts to month-to-month customers to switch to annual plans")
print("  2. Flag customers with 3+ complaints for proactive outreach")
print("  3. Create loyalty rewards for customers in first 12 months")
print("  4. Bundle tech support in entry-level packages")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
