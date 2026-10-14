
import matplotlib.pyplot as plt
import numpy as np

from ml_engine.linear_regression import LinearRegressionGD


# Sample dataset
X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
y = np.array([35, 40, 50, 55, 65, 70], dtype=float)

# Train the model
model = LinearRegressionGD(
    learning_rate=0.01,
    n_iterations=1000
)

model.fit(X, y)

# Get recorded cost values
iterations = np.arange(1, len(model.cost_history) + 1)
cost_values = model.cost_history

# Plot cost versus iterations
plt.figure(figsize=(10, 6))
plt.plot(iterations, cost_values, label="Training MSE")

plt.title("Cost Function Behaviour in Linear Regression")
plt.xlabel("Training Iterations")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the graph for project documentation
plt.savefig("experiments/cost_function_curve.png", dpi=300)

print("Initial cost:", round(cost_values[0], 4))
print("Final cost:", round(cost_values[-1], 4))
print("Graph saved to experiments/cost_function_curve.png")

plt.show()
