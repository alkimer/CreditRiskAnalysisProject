import os

class Settings:
    REDIS_IP = os.getenv("REDIS_IP", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6000))
    REDIS_DB_ID = int(os.getenv("REDIS_DB_ID", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redis")
    API_SLEEP = 1
    REDIS_PENDING_PREDICTION = "pending_prediction_queue"
    # REDIS_COMPLETED_PREDICTION = "completed_prediction_queue"

    # POSTGRES_CONTAINER= os.getenv("POSTGRES_IP", "credit-risk-postgress")
    # POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
    # POSTGRES_DB_NAME = "CREDIT_RISK"
    # POSTGRES_USER = "postgres"
    # POSTGRES_PASS = "postgres"
    API_TIMEOUT = 10

settings = Settings()
