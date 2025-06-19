# Credit Risk Analysis Project

## Local Run

### Backend

For project local running , run in this same path

`docker compose up`

This will start three docker containers:

after successful execution , you should be able to access the API Gui in `localhost:8000/docs`

#### This API will orchestrate request back and forth with the REDIS message broker container and Postgres Container.

ALL prediction requests will be stored in te postgres-db container , db CREDIT_RISK, table PREDICTIONS

## SAMPLE REQUEST TO GET A PREDICTION

### id_client is MANDATORY , all other fields are optional or new ones can be included

curl -X POST "http://localhost:8000/model/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "client_information": {
      "id_client" : 10001,
      "age": 35,
      "income": 42000,
      "marital_status": "single",
      "has_credit_card": true,
      "dependents": 2
    }
}'
git push -u origin# Credit Risk Evaluation App 💼📊

Este proyecto es una aplicación de evaluación de riesgo crediticio que permite:

- Ingresar datos del usuario como edad, ingresos, deuda, educación y tipo de crédito.
- Predecir el nivel de riesgo utilizando un modelo de machine learning.
- Visualizar los resultados y explicaciones usando gráficos interactivos con SHAP.
- Ejecutar inspecciones manuales desde consola para analizar decisiones del modelo.

---
### Frontend

## 🛠 Estructura del front
.
credit_risk_app/
├── app.py              # Interfaz principal con Streamlit
├── main.py             # Backend de prueba con FastAPI que expone la API de predicción
├── model.pkl           # Modelo de prueba entrenado + scaler 
├── styles.css          # Estilos visuales para Streamlit en modo oscuro
├── requirements.txt    # Lista de dependencias para instalar en el entorno
└── README.md           # Documentación completa del proyecto
├── screenshots/        # Imágenes para el README o la presentación
│   ├── newplot3.jpg
│   └── newplot4.png

---

## ⚙️ Entorno virtual

1. Crear un entorno virtual:

python -m venv env
source env/bin/activate   # Windows: .\env\Scripts\activate


## ⚙️ Instalar dependencias

2. Instalar:

pip install -r requirements.txt


## ⚙️ Levantar Backend  de prueba


## ⚙️ Levantar Frontend

4. Bash

streamlit run app.py

---
Si no se ejecuta automatica abrirla en el navegador  en la dirección
http://localhost:8501.
---



