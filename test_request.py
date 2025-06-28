import requests

data = {
    "AGE": 30,
    "SEX": "M",
    "MARITAL_STATUS": 1,
    "OCCUPATION_TYPE": 3,
    "MONTHS_IN_RESIDENCE": 36,
    "FLAG_RESIDENCIAL_PHONE": "Y",
    "STATE_OF_BIRTH": "NY",
    "RESIDENCIAL_STATE": "NY",
    "RESIDENCE_TYPE": 1,
    "RESIDENCIAL_CITY": "New York",
    "RESIDENCIAL_BOROUGH": "Downtown",
    "RESIDENCIAL_PHONE_AREA_CODE": 212,
    "RESIDENCIAL_ZIP_3": 110,
    "PROFESSIONAL_STATE": "NY",
    "PROFESSIONAL_ZIP_3": 111,
    "PRODUCT": "Mortgage Loan"
}

res = requests.post("http://localhost:8000/predict", json=data)
print(res.json())
