import os

class Settings:
    REDIS_IP = os.getenv("REDIS_IP", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB_ID = int(os.getenv("REDIS_DB_ID", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    API_SLEEP = 1000
    REDIS_QUEUE = "credit_prediction_queue"


settings = Settings()
