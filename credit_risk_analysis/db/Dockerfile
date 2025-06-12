FROM postgres:15

# Variables de entorno para crear la base de datos y credenciales
ENV POSTGRES_DB=CREDIT_RISK
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

# Copiar el script SQL que crea la tabla al directorio de inicializacion
COPY init.sql /docker-entrypoint-initdb.d/
