import pandas as pd
import random

# 🔁 Parámetros posibles
sexos = ["M", "F"]
estados_civiles = ["Casado", "Soltero", "Unión libre", "Divorciado"]
ocupaciones = ["Empleado público", "Empleado privado", "Independiente", "Comerciante", "Técnico", "Tecnólogo"]
residencias = ["Propia", "Arrendada", "Familiar", "Empresa"]
estados = ["Bogotá", "Antioquia", "Cundinamarca", "Valle del Cauca", "Santander", "Tolima", "Atlántico"]
ciudades = {
    "Bogotá": ["Chía", "Teusaquillo", "Chapinero", "Suba"],
    "Antioquia": ["Medellín", "Envigado", "Bello"],
    "Cundinamarca": ["Soacha", "Zipaquirá", "Fusagasugá"],
    "Valle del Cauca": ["Cali", "Palmira"],
    "Santander": ["Bucaramanga"],
    "Atlántico": ["Barranquilla"],
    "Tolima": ["Ibagué"]
}
productos = ["Crédito Hipotecario", "Crédito Consumo", "Crédito Vehicular", "Tarjeta de Crédito"]

def generar_cliente():
    state = random.choice(estados)
    ciudad = random.choice(ciudades[state])

    return {
        "AGE": random.randint(21, 65),
        "SEX": random.choice(sexos),
        "MARITAL_STATUS": random.choice(estados_civiles),
        "OCCUPATION_TYPE": random.choice(ocupaciones),
        "MONTHS_IN_RESIDENCE": random.randint(6, 120),
        "FLAG_RESIDENCIAL_PHONE": random.randint(0, 1),
        "STATE_OF_BIRTH": state,
        "RESIDENCIAL_STATE": state,
        "RESIDENCE_TYPE": random.choice(residencias),
        "RESIDENCIAL_CITY": ciudad,
        "RESIDENCIAL_BOROUGH": f"Barrio {random.randint(1, 15)}",
        "RESIDENCIAL_PHONE_AREA_CODE": random.choice([1, 2, 4, 5, 7]),
        "RESIDENCIAL_ZIP_3": random.randint(100, 999),
        "PROFESSIONAL_STATE": state,
        "PROFESSIONAL_ZIP_3": random.randint(100, 999),
        "PRODUCT": random.choice(productos),
        "RISK_LABEL": random.choice(["Bajo", "Alto"])
    }

# 🧠 Generar 100 filas
clientes = pd.DataFrame([generar_cliente() for _ in range(100)])
clientes.to_csv("clientes.csv", index=False)
print("✅ Archivo clientes.csv generado con 100 registros")
