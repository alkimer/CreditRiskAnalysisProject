# Usa imagen oficial de Python
FROM python:3.10-slim

# Crea y entra en el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . /app

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto donde corre la app (ajustable)
EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "credit_risk_analysis.modeling.api.router:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
