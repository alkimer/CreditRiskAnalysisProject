from sqlalchemy import Column, Integer, Text, DECIMAL, TIMESTAMP, String, func, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_date = Column(TIMESTAMP, server_default=func.now())  # Fecha de inserción
    request_json = Column(JSON)  # o usar Text si preferís
    score = Column(DECIMAL)
    model = Column(String)



def insert_prediction(session: Session, request_data: dict, score: float, model_name: str):
    """
    Inserta un nuevo registro en la tabla predictions.

    Parámetros:
    - session: sesión activa de SQLAlchemy
    - request_data: diccionario con los datos del request (se guarda como JSON)
    - score: valor de la predicción (decimal)
    - model_name: nombre del modelo usado
    """
    nueva_prediccion = Prediction(
        request_json=request_data,
        score=score,
        model=model_name
    )
    session.add(nueva_prediccion)
    session.commit()
