CREATE TABLE IF NOT EXISTS public.predictions (
    id SERIAL PRIMARY KEY,                        -- 1. ID incremental
    prediction_date TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP, -- 2. Fecha de inserción
    request_json JSONB,                           -- 3. JSON (puede ser TEXT también, pero JSONB es más eficiente para consultas)
    score DECIMAL,                                -- 4. Valor decimal
    model TEXT                                    -- 5. Nombre del modelo
);
