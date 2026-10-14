
import matplotlib.pyplot as plt
import numpy as np

from ml_engine.linear_regression import LinearRegressionGD


# Sample dataset
X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
y = np.array([35, 40, 50, 55, 65, 70], dtype=float)

# Compare different learning rates.
learning_rates = [0.001, 0.01, 0.1]
n_iterations = 100

plt.figure(figsize=(10, 6))

for rate in learning_rates:
    model = LinearRegressionGD(
        learning_rate=rate,
        n_iterations=n_iterations
    )

    model.fit(X, y)

    plt.plot(
        range(1, n_iterations + 1),
        model.cost_history,
        label=f"Learning rate = {rate}"
    )

    print(
        f"Learning rate: {rate:<6} | "
        f"Initial cost: {model.cost_history[0]:.4f} | "
        f"Final cost: {model.cost_history[-1]:.4f}"
    )

plt.title("Effect of Learning Rate on Cost Function")
plt.xlabel("Training Iterations")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "experiments/learning_rate_comparison.png",
    dpi=300
)

plt.show()
