import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Cargar el CSV codificado
df = pd.read_csv("clientes_clean.csv")  # Asegúrate de que esté codificado

# 2. Variables
target = "RISK_LABEL"
features = [
    'AGE', 'SEX', 'MARITAL_STATUS', 'OCCUPATION_TYPE', 'MONTHS_IN_RESIDENCE',
    'FLAG_RESIDENCIAL_PHONE', 'STATE_OF_BIRTH', 'RESIDENCIAL_STATE',
    'RESIDENCE_TYPE', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH',
    'RESIDENCIAL_PHONE_AREA_CODE', 'RESIDENCIAL_ZIP_3',
    'PROFESSIONAL_STATE', 'PROFESSIONAL_ZIP_3', 'PRODUCT'
]

X = df[features]
y = df[target]

# 3. Preprocesador
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
binary_features = ["FLAG_RESIDENCIAL_PHONE"]
preprocessor = Pipeline(steps=[
    ("scaler", StandardScaler())
])

# 4. Pipeline completo
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Entrenar y evaluar
X_train, X_test, y_train, y_test = train_test_split(X[numeric_features], y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n📊 Resultados del modelo:\n")
print(classification_report(y_test, y_pred))

# 6. Guardar modelo entrenado
joblib.dump(pipeline, "model.pkl")
print("✅ Modelo guardado como model.pkl")
