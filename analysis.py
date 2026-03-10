import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

COLORS = {
    'primary': '#3b82f6',
    'secondary': '#8b5cf6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#06b6d4',
    'churn': '#ef4444',
    'no_churn': '#10b981',
    'palette': ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b',
                '#ef4444', '#06b6d4', '#ec4899', '#84cc16']
}

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)


def separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ==========================================
# 1. DATA LOADING
# ==========================================
separator("1. DATA LOADING")

df = pd.read_csv('data/telecom_churn.csv')

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

print(f"\nDescriptive statistics:")
print(df.describe().round(2).to_string())


# ==========================================
# 2. EXPLORATORY DATA ANALYSIS
# ==========================================
separator("2. EXPLORATORY DATA ANALYSIS")

# Churn Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

churn_counts = df['churn'].value_counts()
colors_churn = [COLORS['no_churn'], COLORS['churn']]

axes[0].bar(churn_counts.index, churn_counts.values, color=colors_churn,
            edgecolor='white', linewidth=2, width=0.5)

for i, (label, val) in enumerate(zip(churn_counts.index, churn_counts.values)):
    axes[0].text(i, val + 50, f'{val:,}\n({val/len(df)*100:.1f}%)',
                 ha='center', fontsize=13, fontweight='bold')

axes[0].set_xlabel('Churn Status', fontsize=13)
axes[0].set_ylabel('Number of Customers', fontsize=13)
axes[0].set_title('Customer Churn Distribution', fontsize=15, fontweight='bold')
axes[0].set_ylim(0, max(churn_counts.values) * 1.2)

axes[1].pie(churn_counts.values, labels=['Retained', 'Churned'],
            colors=colors_churn, autopct='%1.1f%%',
            startangle=90, pctdistance=0.85,
            wedgeprops={'edgecolor': 'white', 'linewidth': 3},
            textprops={'fontsize': 13, 'fontweight': 'bold'})

centre = plt.Circle((0, 0), 0.55, fc='white')
axes[1].add_artist(centre)
axes[1].set_title('Churn Rate', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/churn_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/churn_distribution.png")

churn_rate = (df['churn'] == 'Yes').mean() * 100
print(f"Overall churn rate: {churn_rate:.1f}%")


# Monthly Charges vs Churn
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for label, color in zip(['No', 'Yes'], [COLORS['no_churn'], COLORS['churn']]):
    subset = df[df['churn'] == label]['monthly_charges']
    axes[0].hist(subset, bins=40, alpha=0.6, color=color,
                 label=f'Churn: {label}', edgecolor='white')

axes[0].set_xlabel('Monthly Charges (EUR)', fontsize=13)
axes[0].set_ylabel('Frequency', fontsize=13)
axes[0].set_title('Monthly Charges by Churn', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=12)

churn_data = [
    df[df['churn'] == 'No']['monthly_charges'],
    df[df['churn'] == 'Yes']['monthly_charges']
]
bp = axes[1].boxplot(churn_data, labels=['Retained', 'Churned'],
                     patch_artist=True, widths=0.5)

bp['boxes'][0].set_facecolor(COLORS['no_churn'])
bp['boxes'][1].set_facecolor(COLORS['churn'])
for box in bp['boxes']:
    box.set_alpha(0.7)

axes[1].set_ylabel('Monthly Charges (EUR)', fontsize=13)
axes[1].set_title('Monthly Charges: Retained vs Churned', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/monthly_charges_churn.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/monthly_charges_churn.png")


# Churn by Contract Type
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

contract_churn = df.groupby('contract_type')['churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).sort_values(ascending=False)

bars = axes[0].bar(range(len(contract_churn)), contract_churn.values,
                   color=[COLORS['danger'], COLORS['warning'], COLORS['success']],
                   edgecolor='white', linewidth=2, width=0.5)

axes[0].set_xticks(range(len(contract_churn)))
axes[0].set_xticklabels(contract_churn.index, fontsize=11)
axes[0].set_ylabel('Churn Rate (%)', fontsize=13)
axes[0].set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')

for bar, val in zip(bars, contract_churn.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 1,
                 f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

internet_churn = df.groupby('internet_service')['churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).sort_values(ascending=False)

bars2 = axes[1].bar(range(len(internet_churn)), internet_churn.values,
                    color=COLORS['palette'][:len(internet_churn)],
                    edgecolor='white', linewidth=2, width=0.5)

axes[1].set_xticks(range(len(internet_churn)))
axes[1].set_xticklabels(internet_churn.index, fontsize=11)
axes[1].set_ylabel('Churn Rate (%)', fontsize=13)
axes[1].set_title('Churn Rate by Internet Service', fontsize=14, fontweight='bold')

for bar, val in zip(bars2, internet_churn.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 1,
                 f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/churn_by_contract.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/churn_by_contract.png")


# Churn by Tenure
fig, ax = plt.subplots(figsize=(14, 7))

df['tenure_group'] = pd.cut(
    df['tenure_months'],
    bins=[0, 6, 12, 24, 36, 48, 72],
    labels=['0-6', '7-12', '13-24', '25-36', '37-48', '49-72']
)

tenure_churn = df.groupby('tenure_group', observed=True)['churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
)

bars = ax.bar(range(len(tenure_churn)), tenure_churn.values,
              color=COLORS['primary'], alpha=0.8, edgecolor='white', width=0.6)

for bar, val in zip(bars, tenure_churn.values):
    if val > 40:
        bar.set_color(COLORS['danger'])
    elif val > 25:
        bar.set_color(COLORS['warning'])
    else:
        bar.set_color(COLORS['success'])
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(tenure_churn)))
ax.set_xticklabels(tenure_churn.index, fontsize=10)
ax.set_xlabel('Tenure (months)', fontsize=13)
ax.set_ylabel('Churn Rate (%)', fontsize=13)
ax.set_title('Churn Rate by Customer Tenure', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/churn_by_tenure.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/churn_by_tenure.png")


# Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 10))

df_encoded = df.copy()
label_cols = ['gender', 'has_partner', 'has_dependents', 'contract_type',
              'internet_service', 'phone_service', 'online_security',
              'online_backup', 'streaming_tv', 'payment_method', 'churn']

for col in label_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

numeric_df = df_encoded.select_dtypes(include=[np.number])
cols_to_drop = [c for c in numeric_df.columns if 'customer' in c.lower()]
numeric_df = numeric_df.drop(columns=cols_to_drop, errors='ignore')

corr_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True,
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            ax=ax, vmin=-1, vmax=1)

ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)

plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/correlation_heatmap.png")

churn_corr = corr_matrix['churn'].drop('churn').abs().sort_values(ascending=False)
print("\nTop correlations with churn:")
for feature, corr_val in churn_corr.head(10).items():
    print(f"   {feature:<25} : {corr_val:.3f}")


# ==========================================
# 3. DATA PREPARATION
# ==========================================
separator("3. DATA PREPARATION")

df_ml = df.copy()
df_ml = df_ml.drop(['customer_id', 'tenure_group'], axis=1)
df_ml['churn'] = (df_ml['churn'] == 'Yes').astype(int)

categorical_features = ['gender', 'has_partner', 'has_dependents', 'contract_type',
                        'internet_service', 'phone_service', 'online_security',
                        'online_backup', 'streaming_tv', 'payment_method']

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    label_encoders[col] = le
    print(f"   Encoded: {col}")

X = df_ml.drop('churn', axis=1)
y = df_ml['churn']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nX_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"Churn rate train: {y_train.mean()*100:.1f}%")
print(f"Churn rate test:  {y_test.mean()*100:.1f}%")


# ==========================================
# 4. MODEL TRAINING
# ==========================================
separator("4. MODEL TRAINING")

print("> Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]

lr_cv = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"   CV AUC: {lr_cv.mean():.4f}")

print("\n> Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_split=5,
    min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"   CV AUC: {rf_cv.mean():.4f}")


# ==========================================
# 5. EVALUATION
# ==========================================
separator("5. MODEL EVALUATION")

def evaluate_model(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"\n{name}:")
    print(f"{'-' * 45}")
    print(f"   Accuracy  : {acc*100:.1f}%")
    print(f"   Precision : {prec*100:.1f}%")
    print(f"   Recall    : {rec*100:.1f}%")
    print(f"   F1-Score  : {f1*100:.1f}%")
    print(f"   AUC-ROC   : {auc:.4f}")
    print(f"{'-' * 45}")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}

lr_metrics = evaluate_model("LOGISTIC REGRESSION", y_test, lr_pred, lr_proba)
rf_metrics = evaluate_model("RANDOM FOREST", y_test, rf_pred, rf_proba)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred, target_names=['Retained', 'Churned']))


# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, pred, name in zip(axes, [lr_pred, rf_pred],
                            ['Logistic Regression', 'Random Forest']):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'],
                linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'fontweight': 'bold'}, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{name}\nAccuracy: {accuracy_score(y_test, pred)*100:.1f}%',
                 fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/confusion_matrix.png")


# ROC Curve
fig, ax = plt.subplots(figsize=(10, 8))

for name, proba, color in zip(
    ['Logistic Regression', 'Random Forest'],
    [lr_proba, rf_proba],
    [COLORS['primary'], COLORS['secondary']]
):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.5,
            label=f'{name} (AUC = {auc_val:.3f})')

ax.plot([0, 1], [0, 1], color='gray', linewidth=1.5, linestyle='--', label='Random')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/roc_curve.png")


# Feature Importance
fig, ax = plt.subplots(figsize=(12, 8))

rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_importance = rf_importance.sort_values(ascending=True)

colors_imp = [COLORS['danger'] if v > rf_importance.mean() * 1.5
              else COLORS['primary'] if v > rf_importance.mean()
              else COLORS['palette'][2] for v in rf_importance]

rf_importance.plot(kind='barh', ax=ax, color=colors_imp, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
ax.axvline(rf_importance.mean(), color=COLORS['warning'], linestyle='--', linewidth=2,
           label=f'Mean: {rf_importance.mean():.3f}')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/feature_importance.png")

print("\nTop 5 features:")
for feat, imp in rf_importance.tail(5).items():
    print(f"   > {feat:<25} : {imp:.4f}")


# ==========================================
# 6. SAVE MODELS
# ==========================================
separator("6. SAVING MODELS")

joblib.dump({'model': lr_model, 'scaler': scaler, 'metrics': lr_metrics},
            'models/logistic_regression.pkl')
print("Saved: models/logistic_regression.pkl")

joblib.dump({'model': rf_model, 'scaler': scaler, 'metrics': rf_metrics},
            'models/random_forest.pkl')
print("Saved: models/random_forest.pkl")


# ==========================================
# 7. BUSINESS INSIGHTS
# ==========================================
separator("7. BUSINESS INSIGHTS")

print(f"""
KEY FINDINGS:
{'-' * 50}

1. Overall churn rate: {churn_rate:.1f}%

2. TOP CHURN FACTORS:
   - Month-to-month contracts have highest churn
   - Short tenure customers leave more often
   - Fiber optic users churn more despite paying more
   - No online security = higher churn
   - Electronic check payment linked to higher churn

3. RECOMMENDATIONS:
   a) Target high-risk customers with retention offers
   b) Offer contract upgrades with discounts
   c) Bundle online security for free in first 6 months
   d) Investigate fiber optic service quality
   e) Implement early warning system using ML model

{'=' * 60}
  ANALYSIS COMPLETED SUCCESSFULLY
{'=' * 60}
""")

print("Outputs generated:")
for f in sorted(os.listdir('outputs')):
    print(f"   > outputs/{f}")