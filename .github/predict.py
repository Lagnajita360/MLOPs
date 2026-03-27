import joblib

# Load saved model
model = joblib.load("iris_model.pkl")

# Sample input (sepal length, sepal width, petal length, petal width)
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("Predicted Class:", prediction)