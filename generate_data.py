import pandas as pd
import numpy as np
import os

np.random.seed(42)

N_SAMPLES = 2000

CATALOGUE = {
    'Renault': {
        'models': ['Clio', 'Megane', 'Captur', 'Kadjar', 'Scenic'],
        'base_price': [18000, 24000, 23000, 28000, 30000],
        'segment': 'generaliste'
    },
    'Peugeot': {
        'models': ['208', '308', '2008', '3008', '5008'],
        'base_price': [19000, 25000, 24000, 32000, 35000],
        'segment': 'generaliste'
    },
    'Citroen': {
        'models': ['C3', 'C4', 'C3 Aircross', 'C5 Aircross'],
        'base_price': [17000, 24000, 22000, 30000],
        'segment': 'generaliste'
    },
    'Volkswagen': {
        'models': ['Polo', 'Golf', 'T-Roc', 'Tiguan', 'Passat'],
        'base_price': [21000, 28000, 27000, 35000, 38000],
        'segment': 'premium_accessible'
    },
    'Toyota': {
        'models': ['Yaris', 'Corolla', 'C-HR', 'RAV4'],
        'base_price': [19000, 26000, 28000, 35000],
        'segment': 'generaliste'
    },
    'BMW': {
        'models': ['Serie 1', 'Serie 3', 'X1', 'X3'],
        'base_price': [32000, 42000, 38000, 50000],
        'segment': 'premium'
    },
    'Mercedes': {
        'models': ['Classe A', 'Classe C', 'GLA', 'GLC'],
        'base_price': [34000, 45000, 40000, 52000],
        'segment': 'premium'
    },
    'Audi': {
        'models': ['A1', 'A3', 'A4', 'Q3', 'Q5'],
        'base_price': [28000, 33000, 40000, 36000, 48000],
        'segment': 'premium'
    }
}

CARBURANTS = {
    'Essence': {'proportion': 0.40, 'price_factor': 1.0},
    'Diesel': {'proportion': 0.30, 'price_factor': 1.05},
    'Hybride': {'proportion': 0.20, 'price_factor': 1.15},
    'Electrique': {'proportion': 0.10, 'price_factor': 1.25}
}

TRANSMISSIONS = {
    'Manuelle': {'proportion': 0.55, 'price_factor': 1.0},
    'Automatique': {'proportion': 0.45, 'price_factor': 1.08}
}

COULEURS = ['Blanc', 'Noir', 'Gris', 'Bleu', 'Rouge', 'Argent', 'Vert']


def generate_vehicle_data(n_samples):
    records = []

    for _ in range(n_samples):
        marque = np.random.choice(list(CATALOGUE.keys()))
        info = CATALOGUE[marque]
        model_idx = np.random.randint(len(info['models']))
        modele = info['models'][model_idx]
        prix_base = info['base_price'][model_idx]
        segment = info['segment']

        annee = np.random.randint(2015, 2025)
        age = 2024 - annee

        km_moyen_annuel = np.random.normal(15000, 5000)
        km_moyen_annuel = max(5000, min(km_moyen_annuel, 35000))
        kilometrage = int(age * km_moyen_annuel + np.random.normal(0, 3000))
        kilometrage = max(100, kilometrage)

        carburant = np.random.choice(
            list(CARBURANTS.keys()),
            p=[v['proportion'] for v in CARBURANTS.values()]
        )

        transmission = np.random.choice(
            list(TRANSMISSIONS.keys()),
            p=[v['proportion'] for v in TRANSMISSIONS.values()]
        )

        if segment == 'premium':
            puissance = int(np.random.normal(180, 50))
        elif segment == 'premium_accessible':
            puissance = int(np.random.normal(140, 35))
        else:
            puissance = int(np.random.normal(110, 30))
        puissance = max(70, min(puissance, 350))

        portes = np.random.choice([3, 5], p=[0.15, 0.85])
        couleur = np.random.choice(COULEURS)
        nb_proprietaires = np.random.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.2, 0.05])

        if age == 0:
            decote_age = 0.95
        elif age == 1:
            decote_age = 0.78
        else:
            decote_age = 0.78 * (0.90 ** (age - 1))

        km_ratio = kilometrage / (age * 15000 + 1)
        if km_ratio > 1.3:
            decote_km = 0.90
        elif km_ratio < 0.7:
            decote_km = 1.05
        else:
            decote_km = 1.0

        facteur_carburant = CARBURANTS[carburant]['price_factor']
        facteur_transmission = TRANSMISSIONS[transmission]['price_factor']
        facteur_proprio = 1.0 - (nb_proprietaires - 1) * 0.03
        facteur_puissance = 0.85 + (puissance / 200) * 0.30

        prix = (
            prix_base
            * decote_age
            * decote_km
            * facteur_carburant
            * facteur_transmission
            * facteur_proprio
            * facteur_puissance
        )

        bruit = np.random.normal(1.0, 0.08)
        prix = prix * bruit
        prix = max(1500, round(prix, -2))

        records.append({
            'marque': marque,
            'modele': modele,
            'annee': annee,
            'kilometrage': kilometrage,
            'carburant': carburant,
            'transmission': transmission,
            'puissance_cv': puissance,
            'nb_portes': portes,
            'couleur': couleur,
            'nb_proprietaires': nb_proprietaires,
            'prix': int(prix)
        })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("  GENERATION DU DATASET VEHICULES D'OCCASION")
    print("=" * 60)

    os.makedirs('data', exist_ok=True)

    print(f"\n> Generation de {N_SAMPLES} vehicules...")
    df = generate_vehicle_data(N_SAMPLES)

    filepath = 'data/vehicules.csv'
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"> Dataset sauvegarde : {filepath}")

    print(f"\n{'-' * 40}")
    print(f"  STATISTIQUES DU DATASET")
    print(f"{'-' * 40}")
    print(f"  Nombre de vehicules : {len(df)}")
    print(f"  Marques uniques     : {df['marque'].nunique()}")
    print(f"  Modeles uniques     : {df['modele'].nunique()}")
    print(f"  Prix moyen          : {df['prix'].mean():,.0f} EUR")
    print(f"  Prix median         : {df['prix'].median():,.0f} EUR")
    print(f"  Prix min            : {df['prix'].min():,.0f} EUR")
    print(f"  Prix max            : {df['prix'].max():,.0f} EUR")
    print(f"  Km moyen            : {df['kilometrage'].mean():,.0f} km")
    print(f"{'-' * 40}")

    print("\nDataset genere avec succes !")


if __name__ == '__main__':
    main()