import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. Cargar el dataset
df = pd.read_csv("clientes_clean.csv")

# 2. Variables
target = "RISK_LABEL"

numeric_features = [
    "AGE", "MARITAL_STATUS", "OCCUPATION_TYPE", "MONTHS_IN_RESIDENCE",
    "RESIDENCIAL_PHONE_AREA_CODE", "RESIDENCIAL_ZIP_3",
    "PROFESSIONAL_ZIP_3", "RESIDENCE_TYPE", "PRODUCT"
]

X = df[numeric_features]
y = df[target]

# 3. Pipeline: escalado + modelo
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 4. Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 5. Resultados
print("\n📊 Resultados del modelo:")
print(classification_report(y_test, y_pred))

# 6. Guardar modelo
joblib.dump(pipeline, "model.pkl")
print("✅ Modelo guardado como model.pkl")
