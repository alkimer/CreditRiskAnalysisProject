# Ready for AWS dockerized Credit Risk App

This app is ready for deploy in AWS, check out scripts at /aws folder.

## How to Run the project

```docker compose up --build ```

## UI

http://localhost:8051

## UI Docs

http://localhost:8051/docs

## Samples Api Request

### get a prediction

```
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
### get historic of predictions

```
curl -X GET http://localhost:8000/model/predictions 

```