import joblib
model = joblib.load("model.pkl")
print("Clases del modelo:", model.classes_)