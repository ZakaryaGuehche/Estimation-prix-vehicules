# Telecom Customer Churn Analysis

## Description
Analysis of telecom customer data to predict and understand customer churn using Machine Learning (Logistic Regression and Random Forest).

## Objectives
- Understand key factors driving customer churn
- Build predictive models to identify at-risk customers
- Provide data-driven business recommendations

## Technologies
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Results
| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression | ~79% | ~0.82 |
| Random Forest | ~82% | ~0.85 |

## Visualizations

### Churn Distribution
![Churn](outputs/churn_distribution.png)

### Monthly Charges vs Churn
![Charges](outputs/monthly_charges_churn.png)

### Churn by Contract Type
![Contract](outputs/churn_by_contract.png)

### Churn by Tenure
![Tenure](outputs/churn_by_tenure.png)

### Feature Importance
![Features](outputs/feature_importance.png)

### ROC Curve
![ROC](outputs/roc_curve.png)

### Correlation Heatmap
![Correlation](outputs/correlation_heatmap.png)

### Confusion Matrix
![Confusion](outputs/confusion_matrix.png)

## Installation
```bash
pip install -r requirements.txt
python generate_data.py
python analysis.py