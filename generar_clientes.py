import pandas as pd
import random

# Valores codificados y compatibles
sex_options = ["M", "F"]
marital_options = list(range(1, 8))  # 1–7
occupation_options = list(range(1, 6))  # 1–5
residence_options = list(range(1, 6))  # 1–5
flag_phone_options = ["Y", "N"]
product_options = [1, 2, 7]  # Mortgage (1), Vehicle (2), Student (7)
risk_labels = ["Low", "Medium", "High"]

# Estados y ciudades de Brasil
states = ["SP", "RJ", "MG", "BA", "RS", "PE"]
cities = ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador", "Porto Alegre", "Recife"]

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
        "RESIDENCIAL_BOROUGH": f"Bairro-{random.randint(1, 99)}",
        "RESIDENCIAL_PHONE_AREA_CODE": random.choice([11, 21, 31, 71, 51, 81]),
        "RESIDENCIAL_ZIP_3": random.randint(100, 999),
        "PROFESSIONAL_STATE": random.choice(states),
        "PROFESSIONAL_ZIP_3": random.randint(100, 999),
        "PRODUCT": random.choice(product_options),
        "RISK_LABEL": random.choice(risk_labels)
    }
    records.append(record)

df = pd.DataFrame(records)
df.to_csv("clientes_clean.csv", index=False)
print("✅ Archivo clientes_clean.csv generado con datos brasileños y productos compatibles.")
