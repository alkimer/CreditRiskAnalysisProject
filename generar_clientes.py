import pandas as pd
import random

# Posibles valores codificados
sex_options = ["Male", "Female"]
marital_options = list(range(1, 8))  # 1–7
occupation_options = list(range(1, 6))  # 1–5
residence_options = list(range(1, 6))
flag_phone_options = ["Y", "N"]
product_options = ["Mortgage Loan", "Consumer Credit", "Vehicle Loan", "Credit Card"]
risk_labels = ["Low", "Medium", "High"]

# Ciudades y estados comunes en EE. UU.
cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "San Antonio", "Philadelphia", "San Diego"]
states = ["NY", "CA", "IL", "TX", "AZ", "PA"]

# Lista de registros sintéticos
records = []

for _ in range(200):
    record = {
        "AGE": random.randint(18, 80),
        "SEX": random.choice(sex_options),
        "MARITAL_STATUS": random.choice(marital_options),
        "OCCUPATION_TYPE": random.choice(occupation_options),
        "MONTHS_IN_RESIDENCE": random.randint(1, 240),
        "FLAG_RESIDENCIAL_PHONE": random.choice(flag_phone_options),
        "STATE_OF_BIRTH": random.choice(states),
        "RESIDENCIAL_STATE": random.choice(states),
        "RESIDENCE_TYPE": random.choice(residence_options),
        "RESIDENCIAL_CITY": random.choice(cities),
        "RESIDENCIAL_BOROUGH": f"Borough-{random.randint(1, 99)}",
        "RESIDENCIAL_PHONE_AREA_CODE": random.choice([212, 213, 312, 713, 602, 215]),
        "RESIDENCIAL_ZIP_3": random.randint(100, 999),
        "PROFESSIONAL_STATE": random.choice(states),
        "PROFESSIONAL_ZIP_3": random.randint(100, 999),
        "PRODUCT": random.choice(product_options),
        "RISK_LABEL": random.choice(risk_labels)
    }
    records.append(record)

# Guardar en CSV
df = pd.DataFrame(records)
df.to_csv("clientes_clean.csv", index=False)
print("✅ Archivo clientes_clean.csv generado con 200 registros sintéticos.")
