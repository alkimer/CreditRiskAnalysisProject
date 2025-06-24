import logging
import sys
import json
from typing import List
from sqlalchemy import create_engine, Column, Integer, Text, DECIMAL, TIMESTAMP, String, func
from sqlalchemy.orm import sessionmaker, Session, declarative_base

# Configuración de logs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.info("----INIT PREDICTION ORM for SQLITE----")

# Configuración de la base de datos SQLite
DATABASE_URL = "sqlite:///data/predictions.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modelo ORM de la tabla
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_date = Column(TIMESTAMP, server_default=func.now())
    request_json = Column(Text)  # Guardado como string (JSON serializado)
    score = Column(DECIMAL)
    model = Column(String)

# Insertar una nueva predicción
def insert_prediction(session: Session, request_data: dict, score: float, model_name: str):
    logger.info("✅ guardando predicción en SQLITE")

    nueva_prediccion = Prediction(
        request_json=json.dumps(request_data),
        score=score,
        model=model_name
    )
    session.add(nueva_prediccion)
    session.commit()
    logger.info("✅ predicción guardada en SQLITE")

# Obtener todas las predicciones
def get_all_predictions(session: Session) -> List[Prediction]:
    return session.query(Prediction).order_by(Prediction.id.desc()).all()

# Dependencia para inyectar sesión en endpoints FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
