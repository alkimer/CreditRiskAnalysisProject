import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 👈 Escalador
import joblib

# Cargar dataset
df = pd.read_csv("clientes.csv")

# Codificar variables categóricas
df["education"] = df["education"].map({
    "Bachillerato": 0,
    "Universitario": 1,
    "Postgrado": 2,
    "Otro": 3
})
df["credit_type"] = df["credit_type"].map({
    "Auto": 0,
    "Casa": 1,
    "Educación": 2,
    "Tarjeta de Crédito": 3
})

X = df.drop("risk", axis=1)
y = df["risk"]

# Escalar las variables numéricas y categóricas
scaler = StandardScaler()                           # 👈 Crear el escalador
X_scaled = scaler.fit_transform(X)                  # 👈 Escalar los datos

# Entrenar el modelo con datos escalados
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_scaled, y)

# Guardar modelo y escalador juntos
joblib.dump((model, scaler), "model.pkl")           # 👈 Paquete completo
print("✅ Modelo y escalador guardados como 'model.pkl'")