import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 📥 1. Carga de datos
df = pd.read_csv("clientes.csv")

# 🎯 2. Definir variables
target = "RISK_LABEL"  # ajusta según tu dataset
features = [
    'AGE', 'SEX', 'MARITAL_STATUS', 'OCCUPATION_TYPE', 'MONTHS_IN_RESIDENCE',
    'FLAG_RESIDENCIAL_PHONE', 'STATE_OF_BIRTH', 'RESIDENCIAL_STATE',
    'RESIDENCE_TYPE', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH',
    'RESIDENCIAL_PHONE_AREA_CODE', 'RESIDENCIAL_ZIP_3',
    'PROFESSIONAL_STATE', 'PROFESSIONAL_ZIP_3', 'PRODUCT'
]

X = df[features]
y = df[target]

# 🧠 3. Columnas categóricas y numéricas
cat_features = [col for col in features if X[col].dtype == "object"]
num_features = [col for col in features if X[col].dtype in ["int64", "float64"]]

# 🔧 4. Preprocesador
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# 🌲 5. Pipeline completo
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 🧪 6. Entrenar y evaluar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("\n📊 Resultados del modelo:\n")
print(classification_report(y_test, y_pred))

# 💾 7. Guardar modelo completo (pipeline + modelo + encoder)
joblib.dump(pipeline, "model.pkl")
print("\n✅ Modelo guardado como model.pkl")
