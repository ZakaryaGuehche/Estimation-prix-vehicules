import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    'palette': ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b',
                '#ef4444', '#06b6d4', '#ec4899', '#84cc16']
}

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)


def separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


separator("1. CHARGEMENT DES DONNEES")

df = pd.read_csv('data/vehicules.csv')

print(f"Shape : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print(f"\nColonnes :")
for col in df.columns:
    dtype = df[col].dtype
    n_unique = df[col].nunique()
    n_null = df[col].isnull().sum()
    print(f"   > {col:<20} | Type: {str(dtype):<10} | Uniques: {n_unique:<6} | Nulls: {n_null}")

print(f"\nStatistiques descriptives :")
print(df.describe().round(1).to_string())


separator("2. ANALYSE EXPLORATOIRE (EDA)")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(df['prix'], bins=50, color=COLORS['primary'],
             alpha=0.7, edgecolor='white')
axes[0].axvline(df['prix'].mean(), color=COLORS['danger'],
                linestyle='--', linewidth=2,
                label=f"Moyenne: {df['prix'].mean():,.0f} EUR")
axes[0].axvline(df['prix'].median(), color=COLORS['warning'],
                linestyle='--', linewidth=2,
                label=f"Mediane: {df['prix'].median():,.0f} EUR")
axes[0].set_xlabel('Prix (EUR)', fontsize=12)
axes[0].set_ylabel('Frequence', fontsize=12)
axes[0].set_title('Distribution des Prix', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)

df_sorted = df.groupby('marque')['prix'].median().sort_values()
order = df_sorted.index.tolist()
sns.boxplot(data=df, x='marque', y='prix', order=order,
            palette=COLORS['palette'], ax=axes[1])
axes[1].set_xlabel('Marque', fontsize=12)
axes[1].set_ylabel('Prix (EUR)', fontsize=12)
axes[1].set_title('Prix par Marque', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('outputs/distribution_prix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graphique sauvegarde : outputs/distribution_prix.png")


fig, ax = plt.subplots(figsize=(12, 7))

segments = df['marque'].map(
    lambda x: 'Premium' if x in ['BMW', 'Mercedes', 'Audi'] else 'Generaliste'
)

for segment, color in zip(['Generaliste', 'Premium'],
                           [COLORS['primary'], COLORS['secondary']]):
    mask = segments == segment
    ax.scatter(df.loc[mask, 'kilometrage'], df.loc[mask, 'prix'],
               alpha=0.4, s=30, c=color, label=segment,
               edgecolors='white', linewidth=0.5)

z = np.polyfit(df['kilometrage'], df['prix'], 2)
p = np.poly1d(z)
x_line = np.linspace(df['kilometrage'].min(), df['kilometrage'].max(), 100)
ax.plot(x_line, p(x_line), color=COLORS['danger'], linewidth=2.5,
        linestyle='--', label='Tendance')

ax.set_xlabel('Kilometrage (km)', fontsize=13)
ax.set_ylabel('Prix (EUR)', fontsize=13)
ax.set_title('Relation Prix vs Kilometrage', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/prix_vs_kilometrage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graphique sauvegarde : outputs/prix_vs_kilometrage.png")


fig, ax = plt.subplots(figsize=(10, 8))

numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, square=True,
            linewidths=1, linecolor='white',
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            ax=ax)

ax.set_title('Matrice de Correlation', fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graphique sauvegarde : outputs/correlation_matrix.png")

print("\nPrix moyen par carburant :")
print(df.groupby('carburant')['prix'].agg(['mean', 'median', 'count'])
      .sort_values('mean', ascending=False).round(0).to_string())

print("\nPrix moyen par marque :")
print(df.groupby('marque')['prix'].agg(['mean', 'median', 'count'])
      .sort_values('mean', ascending=False).round(0).to_string())


separator("3. FEATURE ENGINEERING")

df['age'] = 2024 - df['annee']
df['km_par_an'] = df['kilometrage'] / (df['age'] + 1)

segment_map = {
    'BMW': 'Premium', 'Mercedes': 'Premium', 'Audi': 'Premium',
    'Volkswagen': 'Premium_Accessible',
    'Renault': 'Generaliste', 'Peugeot': 'Generaliste',
    'Citroen': 'Generaliste', 'Toyota': 'Generaliste'
}
df['segment'] = df['marque'].map(segment_map)

df['puissance_categorie'] = pd.cut(
    df['puissance_cv'],
    bins=[0, 100, 150, 200, 400],
    labels=['Faible', 'Moyenne', 'Forte', 'Tres forte']
)

df['log_km'] = np.log1p(df['kilometrage'])

print("Nouvelles features creees :")
print(f"   > age")
print(f"   > km_par_an")
print(f"   > segment")
print(f"   > puissance_categorie")
print(f"   > log_km")
print(f"\nShape apres feature engineering : {df.shape}")


separator("4. PREPARATION DES DONNEES")

features = [
    'annee', 'kilometrage', 'puissance_cv', 'nb_portes',
    'nb_proprietaires', 'age', 'km_par_an', 'log_km',
    'marque', 'carburant', 'transmission', 'segment'
]

df_model = df[features + ['prix']].copy()

label_encoders = {}
categorical_cols = ['marque', 'carburant', 'transmission', 'segment']

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le
    print(f"   > {col} encode : {le.classes_}")

X = df_model.drop('prix', axis=1)
y = df_model['prix']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDimensions :")
print(f"   > X_train : {X_train.shape}")
print(f"   > X_test  : {X_test.shape}")


separator("5. MODELISATION - RANDOM FOREST")

print("> Entrainement du modele Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("Modele entraine avec succes !")

print("\n> Validation croisee (5-fold)...")
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
print(f"   R2 scores : {cv_scores.round(4)}")
print(f"   R2 moyen  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


separator("6. EVALUATION DU MODELE")

y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Metriques de performance :")
print(f"{'-' * 40}")
print(f"   R2 Score  : {r2:.4f}")
print(f"   MAE       : {mae:,.0f} EUR")
print(f"   RMSE      : {rmse:,.0f} EUR")
print(f"   MAPE      : {mape:.1f} %")
print(f"{'-' * 40}")


fig, ax = plt.subplots(figsize=(10, 8))

importance = pd.Series(rf_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=True)

colors = [COLORS['primary'] if v > importance.mean()
          else COLORS['palette'][2] for v in importance]

importance.plot(kind='barh', ax=ax, color=colors, edgecolor='white', linewidth=0.5)

ax.set_xlabel('Importance', fontsize=13)
ax.set_title('Importance des Variables (Random Forest)',
             fontsize=15, fontweight='bold')
ax.axvline(importance.mean(), color=COLORS['danger'], linestyle='--',
           alpha=0.7, label=f'Moyenne: {importance.mean():.3f}')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGraphique sauvegarde : outputs/feature_importance.png")


fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].scatter(y_test, y_pred, alpha=0.5, s=20,
                c=COLORS['primary'], edgecolors='white', linewidth=0.3)
lim_min = min(y_test.min(), y_pred.min()) * 0.9
lim_max = max(y_test.max(), y_pred.max()) * 1.1
axes[0].plot([lim_min, lim_max], [lim_min, lim_max],
             color=COLORS['danger'], linewidth=2, linestyle='--',
             label='Prediction parfaite')
axes[0].set_xlabel('Prix Reel (EUR)', fontsize=13)
axes[0].set_ylabel('Prix Predit (EUR)', fontsize=13)
axes[0].set_title(f'Predictions vs Reel (R2 = {r2:.3f})',
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

erreurs = y_test - y_pred
axes[1].hist(erreurs, bins=50, color=COLORS['secondary'],
             alpha=0.7, edgecolor='white')
axes[1].axvline(0, color=COLORS['danger'], linewidth=2, linestyle='--')
axes[1].axvline(erreurs.mean(), color=COLORS['warning'], linewidth=2,
                linestyle='--', label=f'Erreur moyenne: {erreurs.mean():,.0f} EUR')
axes[1].set_xlabel('Erreur (EUR)', fontsize=13)
axes[1].set_ylabel('Frequence', fontsize=13)
axes[1].set_title('Distribution des Erreurs', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('outputs/predictions_vs_reel.png', dpi=150, bbox_inches='tight')
plt.close()
print("Graphique sauvegarde : outputs/predictions_vs_reel.png")


separator("7. SAUVEGARDE")

model_path = 'models/random_forest_model.pkl'
joblib.dump({
    'model': rf_model,
    'label_encoders': label_encoders,
    'features': features,
    'metrics': {'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape}
}, model_path)

print(f"Modele sauvegarde : {model_path}")


separator("8. EXEMPLE DE PREDICTION")

exemple = pd.DataFrame([{
    'annee': 2020,
    'kilometrage': 45000,
    'puissance_cv': 130,
    'nb_portes': 5,
    'nb_proprietaires': 1,
    'age': 4,
    'km_par_an': 11250,
    'log_km': np.log1p(45000),
    'marque': label_encoders['marque'].transform(['Peugeot'])[0],
    'carburant': label_encoders['carburant'].transform(['Essence'])[0],
    'transmission': label_encoders['transmission'].transform(['Manuelle'])[0],
    'segment': label_encoders['segment'].transform(['Generaliste'])[0]
}])

prediction = rf_model.predict(exemple)[0]

print(f"Vehicule : Peugeot 308 Essence Manuelle")
print(f"   Annee       : 2020")
print(f"   Kilometrage : 45,000 km")
print(f"   Puissance   : 130 CV")
print(f"\nPrix estime : {prediction:,.0f} EUR")
print(f"\n{'=' * 60}")
print(f"  ANALYSE TERMINEE AVEC SUCCES")
print(f"{'=' * 60}")