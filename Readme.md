# Credit Risk Analysis Project

### Local Run

For project local running , run in this same path

`docker compose up`

This will start three docker containers:

after successful execution , you should be able to access the API Gui in `localhost:8000/docs`

#### This API will orchestrate request back and forth with the REDIS message broker container and Postgres Container.

ALL prediction requests will be stored in te postgres-db container , db CREDIT_RISK, table PREDICTIONS

## SAMPLE REQUEST TO GET A PREDICTION

### id_client is MANDATORY , all other fields are optional or new ones can be included

curl -X POST "http://localhost:8000/model/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "client_information": {
      "id_client" : 10001,
      "age": 35,
      "income": 42000,
      "marital_status": "single",
      "has_credit_card": true,
      "dependents": 2
    }
}'
