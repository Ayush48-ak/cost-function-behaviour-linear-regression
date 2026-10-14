
import numpy as np

from ml_engine.linear_regression import LinearRegressionGD

X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
y = np.array([35, 40, 50, 55, 65, 70], dtype=float)

model = LinearRegressionGD(
    learning_rate=0.01,
    n_iterations=1000
)

model.fit(X, y)
predictions = model.predict(X)

final_mse = model.calculate_mse(y, predictions)

print("LINEAR REGRESSION RESULTS")
print("-" * 35)
print("Training samples:", len(y))
print("Iterations:", model.n_iterations)
print("Initial cost:", round(model.cost_history[0], 4))
print("Final cost:", round(model.cost_history[-1], 4))
print("Final training MSE:", round(final_mse, 4))

print("\nACTUAL VS PREDICTED")
for actual, predicted in zip(y, predictions):
    print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")

print("\nFirst 5 cost values:", model.cost_history[:5])
print("Last 5 cost values:", model.cost_history[-5:])
