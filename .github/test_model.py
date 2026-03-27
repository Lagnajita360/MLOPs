import joblib

# Load model
model = joblib.load("iris_model.pkl")

# Sample test
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("Prediction:", prediction)