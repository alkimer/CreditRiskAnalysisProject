-- Crear la tabla PREDICTIONS si no existe
CREATE TABLE IF NOT EXISTS public.predictions (
    id SERIAL PRIMARY KEY,
    client_id INTEGER NOT NULL,
    age INTEGER,
    prediction BOOLEAN,
    prediction_date TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);