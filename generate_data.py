import pandas as pd
import numpy as np
import os

np.random.seed(42)

N_CUSTOMERS = 5000

CONTRACTS = {
    'Month-to-month': {'proportion': 0.50, 'churn_rate': 0.42},
    'One year': {'proportion': 0.25, 'churn_rate': 0.11},
    'Two year': {'proportion': 0.25, 'churn_rate': 0.03}
}

INTERNET = {
    'Fiber optic': {'proportion': 0.40, 'churn_factor': 1.4, 'charge_base': 75},
    'DSL': {'proportion': 0.35, 'churn_factor': 0.8, 'charge_base': 50},
    'None': {'proportion': 0.25, 'churn_factor': 0.5, 'charge_base': 20}
}

PAYMENT_METHODS = {
    'Electronic check': {'proportion': 0.35, 'churn_factor': 1.3},
    'Bank transfer': {'proportion': 0.25, 'churn_factor': 0.8},
    'Credit card': {'proportion': 0.25, 'churn_factor': 0.7},
    'Mailed check': {'proportion': 0.15, 'churn_factor': 1.0}
}


def generate_telecom_data(n_customers):
    records = []

    for i in range(n_customers):
        customer_id = f"CUST-{i+1:05d}"

        gender = np.random.choice(['Male', 'Female'])
        age = int(np.random.normal(45, 15))
        age = max(18, min(age, 85))
        has_partner = np.random.choice(['Yes', 'No'], p=[0.48, 0.52])
        has_dependents = np.random.choice(['Yes', 'No'], p=[0.30, 0.70])

        contract = np.random.choice(
            list(CONTRACTS.keys()),
            p=[v['proportion'] for v in CONTRACTS.values()]
        )

        if contract == 'Month-to-month':
            tenure = int(np.random.exponential(18))
        elif contract == 'One year':
            tenure = int(np.random.normal(36, 15))
        else:
            tenure = int(np.random.normal(54, 18))
        tenure = max(1, min(tenure, 72))

        internet = np.random.choice(
            list(INTERNET.keys()),
            p=[v['proportion'] for v in INTERNET.values()]
        )

        phone = np.random.choice(['Yes', 'No'], p=[0.90, 0.10])

        if internet != 'None':
            online_security = np.random.choice(['Yes', 'No'], p=[0.35, 0.65])
            online_backup = np.random.choice(['Yes', 'No'], p=[0.35, 0.65])
            streaming_tv = np.random.choice(['Yes', 'No'], p=[0.40, 0.60])
        else:
            online_security = 'No'
            online_backup = 'No'
            streaming_tv = 'No'

        payment = np.random.choice(
            list(PAYMENT_METHODS.keys()),
            p=[v['proportion'] for v in PAYMENT_METHODS.values()]
        )

        base_charge = INTERNET[internet]['charge_base']
        if phone == 'Yes':
            base_charge += 20
        if online_security == 'Yes':
            base_charge += 10
        if online_backup == 'Yes':
            base_charge += 8
        if streaming_tv == 'Yes':
            base_charge += 12

        monthly_charges = base_charge * np.random.normal(1.0, 0.10)
        monthly_charges = max(18, round(monthly_charges, 2))

        total_charges = round(monthly_charges * tenure * np.random.normal(1.0, 0.05), 2)
        total_charges = max(monthly_charges, total_charges)

        num_tickets = np.random.poisson(2)
        if internet == 'Fiber optic':
            num_tickets += np.random.poisson(1)

        base_churn_prob = CONTRACTS[contract]['churn_rate']

        if tenure <= 6:
            tenure_factor = 1.5
        elif tenure <= 12:
            tenure_factor = 1.2
        elif tenure <= 24:
            tenure_factor = 1.0
        else:
            tenure_factor = 0.6

        internet_factor = INTERNET[internet]['churn_factor']
        payment_factor = PAYMENT_METHODS[payment]['churn_factor']

        if monthly_charges > 80:
            charge_factor = 1.3
        elif monthly_charges > 60:
            charge_factor = 1.1
        else:
            charge_factor = 0.8

        if online_security == 'No' and internet != 'None':
            security_factor = 1.3
        else:
            security_factor = 0.7

        if num_tickets >= 4:
            ticket_factor = 1.4
        elif num_tickets >= 2:
            ticket_factor = 1.1
        else:
            ticket_factor = 0.9

        if age < 30:
            age_factor = 1.2
        elif age > 60:
            age_factor = 0.8
        else:
            age_factor = 1.0

        family_factor = 1.0
        if has_partner == 'Yes':
            family_factor *= 0.85
        if has_dependents == 'Yes':
            family_factor *= 0.80

        churn_prob = (
            base_churn_prob
            * tenure_factor
            * internet_factor
            * payment_factor
            * charge_factor
            * security_factor
            * ticket_factor
            * age_factor
            * family_factor
        )

        churn_prob = min(0.95, max(0.02, churn_prob))
        churn = 'Yes' if np.random.random() < churn_prob else 'No'

        records.append({
            'customer_id': customer_id,
            'gender': gender,
            'age': age,
            'has_partner': has_partner,
            'has_dependents': has_dependents,
            'tenure_months': tenure,
            'contract_type': contract,
            'internet_service': internet,
            'phone_service': phone,
            'online_security': online_security,
            'online_backup': online_backup,
            'streaming_tv': streaming_tv,
            'payment_method': payment,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'num_support_tickets': num_tickets,
            'churn': churn
        })

    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("  TELECOM CHURN DATASET GENERATION")
    print("=" * 60)

    os.makedirs('data', exist_ok=True)

    print(f"\n> Generating {N_CUSTOMERS} customers...")
    df = generate_telecom_data(N_CUSTOMERS)

    filepath = 'data/telecom_churn.csv'
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"> Dataset saved: {filepath}")

    churn_count = df['churn'].value_counts()
    churn_rate = (churn_count.get('Yes', 0) / len(df)) * 100

    print(f"\n{'-' * 45}")
    print(f"  DATASET STATISTICS")
    print(f"{'-' * 45}")
    print(f"  Total customers     : {len(df):,}")
    print(f"  Churned (Yes)       : {churn_count.get('Yes', 0):,}")
    print(f"  Retained (No)       : {churn_count.get('No', 0):,}")
    print(f"  Churn rate          : {churn_rate:.1f}%")
    print(f"  Avg monthly charges : {df['monthly_charges'].mean():.2f} EUR")
    print(f"  Avg tenure          : {df['tenure_months'].mean():.1f} months")
    print(f"{'-' * 45}")

    print(f"\nDataset generated successfully!")


if __name__ == '__main__':
    main()