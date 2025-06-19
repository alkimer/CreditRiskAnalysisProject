git push -u origin# Credit Risk Evaluation App 💼📊

Este proyecto es una aplicación de evaluación de riesgo crediticio que permite:

- Ingresar datos del usuario como edad, ingresos, deuda, educación y tipo de crédito.
- Predecir el nivel de riesgo utilizando un modelo de machine learning.
- Visualizar los resultados y explicaciones usando gráficos interactivos con SHAP.
- Ejecutar inspecciones manuales desde consola para analizar decisiones del modelo.

---

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

3. bash

uvicorn main:app --reload

---
Esto crea una API local en:
http://localhost:8000/predict
---

## ⚙️ Levantar Frontend

4. Bash

streamlit run app.py

---
Si no se ejecuta automatica abrirla en el navegador  en la dirección
http://localhost:8501.
---



