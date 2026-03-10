\# Estimation du Prix de Vehicules d'Occasion



\## Description

Modele de Machine Learning (Random Forest) pour estimer le prix de revente de vehicules d'occasion en France. Le modele prend en compte le kilometrage, l'age, la motorisation, la puissance et la marque pour predire la decote.



\## Objectifs

\- Analyser les facteurs influencant le prix des vehicules d'occasion

\- Construire un modele predictif performant (Random Forest)

\- Identifier les variables les plus impactantes (Feature Importance)

\- Fournir un outil d'estimation fiable



\## Technologies

\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn



\## Resultats

| Metrique | Valeur |

|----------|--------|

| R2 Score | 0.74 |

| MAE | 3 592 EUR |

| RMSE | 4 765 EUR |

| MAPE | 20.3% |



\## Visualisations



\### Distribution des prix

!\[Distribution](outputs/distribution\_prix.png)



\### Prix vs Kilometrage

!\[Prix vs KM](outputs/prix\_vs\_kilometrage.png)



\### Matrice de correlation

!\[Correlation](outputs/correlation\_matrix.png)



\### Importance des variables

!\[Feature Importance](outputs/feature\_importance.png)



\### Predictions vs Reel

!\[Predictions](outputs/predictions\_vs\_reel.png)



\## Installation

```bash

pip install -r requirements.txt

python generate\_data.py

python analysis.py

