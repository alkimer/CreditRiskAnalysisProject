# 🧠 Credit Risk Analysis Project

## Problema

El acceso al crédito es fundamental para el desarrollo económico, pero representa un riesgo para las instituciones financieras. Tradicionalmente, la evaluación del riesgo crediticio (credit scoring) se ha basado en reglas rígidas y modelos estadísticos simples, que pueden dejar fuera a personas con perfiles no tradicionales o generar decisiones poco precisas.

El reto es construir un sistema capaz de predecir de manera precisa y automatizada la probabilidad de incumplimiento de pago de un solicitante, utilizando información financiera y demográfica, para así facilitar decisiones de crédito más justas, rápidas y eficientes.

## Solución Propuesta

Este proyecto implementa una **plataforma integral de análisis de riesgo crediticio** basada en machine learning, que permite:

- **Recolectar y procesar datos** de solicitantes de crédito.
- **Predecir el riesgo crediticio** usando modelos de ML entrenados con datos históricos.
- **Visualizar y consultar resultados** a través de una interfaz web amigable.
- **Orquestar el flujo de datos y predicciones** de manera escalable y reproducible.

### Componentes principales

- **API REST (FastAPI):** expone endpoints para recibir solicitudes de predicción y consultar el historial.
- **Motor de predicción asíncrono (Redis + Worker):** desacopla la recepción de solicitudes y el procesamiento ML para mayor escalabilidad.
- **Base de datos (SQLite):** almacena el historial de predicciones y solicitudes.
- **Interfaz de usuario (Streamlit):** permite a usuarios no técnicos interactuar con el sistema y visualizar resultados.
- **Modelos de ML:** entrenados con técnicas modernas de selección de variables y ensamblado (stacking), serializados para inferencia rápida.

---

## ¿Cómo funciona el flujo?

1. **Ingreso de datos:** El usuario completa un formulario con información relevante (edad, estado civil, ocupación, etc.).
2. **Solicitud de predicción:** Los datos se envían a la API, que los coloca en una cola de trabajo.
3. **Predicción ML:** Un worker procesa la solicitud, aplica el preprocesamiento y el modelo entrenado, y almacena el resultado.
4. **Visualización:** El usuario puede ver el resultado de su predicción y consultar el historial de solicitudes previas.

---

## Ejemplo de uso

### 1. Ejecutar la plataforma

```bash
docker compose up --build
```

### 2. Acceder a la interfaz

- **UI:** [http://localhost:8051](http://localhost:8051)
- **API Docs:** [http://localhost:8051/docs](http://localhost:8051/docs)

### 3. Realizar una predicción vía API

```bash
curl -X POST http://localhost:8000/model/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MARITAL_STATUS": 1,
    "MONTHS_IN_RESIDENCE": 12,
    "AGE": 35,
    "OCCUPATION_TYPE": 1,
    "SEX": "M",
    "FLAG_RESIDENCIAL_PHONE": "Y",
    "STATE_OF_BIRTH": "BA",
    "RESIDENCIAL_STATE": "BA",
    "RESIDENCE_TYPE": 1,
    "PROFESSIONAL_STATE": "BA",
    "PRODUCT": "Personal Loan",
    "RESIDENCIAL_CITY": "Bahía Blanca",
    "RESIDENCIAL_BOROUGH": "Centro",
    "RESIDENCIAL_PHONE_AREA_CODE": "291",
    "RESIDENCIAL_ZIP_3": "800",
    "PROFESSIONAL_ZIP_3": "800"
  }'
```

---

## Estructura del Proyecto

- **`credit_risk_analysis/api/`**: Endpoints REST y lógica de negocio.
- **`credit_risk_analysis/model/`**: Modelos de ML y scripts de predicción.
- **`credit_risk_analysis/db/`**: ORM y gestión de base de datos.
- **`credit_risk_analysis/services/`**: Productor/consumidor de trabajos de predicción.
- **`UI/`**: Interfaz de usuario en Streamlit.
- **`models/`**: Modelos y pipelines serializados.
- **`data/`**: Datos de entrada y procesamiento.
- **`notebooks/`**: Jupyter notebooks para exploración y entrenamiento.

---

## Tecnologías

- Python 3.10+
- FastAPI
- Streamlit
- Scikit-learn
- Redis
- Docker Compose
- SQLite

---

## Estado

✅ Funcional y listo para pruebas  
🔜 Futuras mejoras: integración con PostgreSQL, despliegue cloud, explicabilidad de modelos.

---

## Créditos

Desarrollado como proyecto final para AnyoneAI.

---