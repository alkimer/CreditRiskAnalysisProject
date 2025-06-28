# 🧠 Credit Risk Analysis Project

Este proyecto implementa un sistema completo de **análisis de riesgo crediticio**, combinando:

- 🔌 Una **API REST** construida con **FastAPI**
- 🧵 Un motor de predicción asincrónico usando **Redis**
- 🗃 Una base de datos ligera basada en **SQLite**
- 🖥 Una interfaz de usuario intuitiva desarrollada en **Streamlit**
- 🐳 Todo completamente orquestado mediante **Docker Compose**

---

## 🚀 ¿Cómo ejecutar el proyecto?

Cloná el repositorio y ejecutá:

```bash
docker compose up --build
```

Esto levantará todos los servicios necesarios: API, worker, Redis, base de datos y la UI.

---

## 🌐 Accesos

- **🧑‍💻 Interfaz de Usuario (UI)**:  
  👉 [http://localhost:8051](http://localhost:8051)

- **📚 Documentación Interactiva de la API** (Swagger):  
  👉 [http://localhost:8051/docs](http://localhost:8051/docs)

---

## 📬 Ejemplos de uso vía API

### 🔍 Obtener una predicción

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

### 🕓 Consultar historial de predicciones

```bash
curl -X GET http://localhost:8000/model/predictions
```

---

## 📦 Estructura de servicios (Docker Compose)

| Servicio             | Descripción                                  |
|----------------------|----------------------------------------------|
| `credit-risk-api`    | API REST con FastAPI                         |
| `credit-risk-worker` | Worker asincrónico con Redis                 |
| `credit-risk-redis`  | Servicio de cola de trabajos                 |
| `credit-risk-ui`     | Interfaz de usuario con Streamlit            |
| `sqlite`             | Base de datos ligera embebida                |



## Estructura del Proyecto

### `credit_risk_analysis/db/`
Contiene los scripts y componentes necesarios para la inicialización e interacción con la base de datos. Actualmente se utiliza **SQLite** como motor de almacenamiento relacional, elegido por su simplicidad y compatibilidad con despliegues livianos en AWS (limitación impuesta por la imposibilidad de usar PostgreSQL en este entorno).

- `prediction_orm.py`: Define el modelo ORM para registrar y consultar predicciones realizadas.

---

### `credit_risk_analysis/api/`
Define los **endpoints REST** expuestos por el sistema para interactuar con el motor de predicción. A través de esta capa se reciben solicitudes externas para generar nuevas predicciones y consultar el historial existente.

- `router.py`: Contiene las rutas y controladores principales para los endpoints del sistema.

---

### `credit_risk_analysis/model/`
Incluye la lógica encapsulada del **modelo de machine learning**. Esta capa actúa como fachada hacia el pipeline de inferencia, cargando el modelo serializado y ejecutando la predicción sobre los datos entrantes.

- `predict.py`: Función principal de predicción, incluye validaciones y preprocesamiento.

---

### `models/`
Contiene los **artefactos del modelo serializado**, incluyendo tanto el pipeline de preprocesamiento como el modelo entrenado. Estos objetos son utilizados en tiempo de inferencia.

- `preprocessor.pkl`: Pipeline de preprocesamiento de datos.
- `stacking_model.pkl`: Modelo de machine learning serializado (Stacking Ensemble).

---

### `credit_risk_analysis/services/`
Define los **servicios asincrónicos** encargados de gestionar la cola de trabajos de predicción mediante Redis. Este patrón desacopla el proceso de predicción de la interfaz API, permitiendo un escalamiento más flexible.

- `prediction_job_producer.py`: Encola nuevas tareas de predicción.
- `prediction_job_consumer.py`: Worker que consume tareas de Redis y ejecuta las predicciones.
- `schema.py`: Define esquemas y tipos usados entre componentes del servicio.

---

### `UI/`
Contiene la **interfaz de usuario desarrollada en Streamlit**, pensada para permitir interacción sencilla con el sistema por parte de usuarios no técnicos. Se pueden realizar predicciones manuales y visualizar resultados.

- `app.py`: Archivo principal de la aplicación Streamlit.
- `assets/`: Archivos estáticos (CSS, imágenes, etc.)
- `logo_intro.css`: Estilos personalizados de la interfaz.

---

### `aws/`
Scripts de automatización y configuración para despliegue del sistema en una instancia EC2 de AWS.

- `connect_aws.sh`: Conexión por SSH a la instancia remota.
- `setup_env_aws.sh`: Configura el entorno en la instancia EC2 (Docker, dependencias, etc.).
- `subir_proyecto_aws.sh`: Sincroniza el proyecto local con la instancia EC2.

- `deploy_and_run_aws.sh`: Es el All-in-one, hace todo lo que hacen los anteriores.

---

## Despliegue

Todo el sistema está **dockerizado**. La ejecución completa de los servicios se realiza mediante:

```bash
docker compose up --build -d
```

Este comando levanta los siguientes contenedores:

- API REST
- Redis
- Base de datos SQLite
- Interfaz Streamlit

> ⚠️ Nota: Debido a restricciones de red y almacenamiento en el entorno AWS Free Tier, se utiliza SQLite en lugar de PostgreSQL para simplificar el despliegue.

---

## Tecnologías utilizadas

- Python 3.10+
- FastAPI
- Redis
- Streamlit
- Scikit-learn
- Docker / Docker Compose
- SQLite

---

## Estado del Proyecto

✅ Funcional  
🔜 En futuras versiones se evaluará migrar la base de datos a PostgreSQL y escalar el sistema con Kubernetes para ambientes productivos.

---