
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from ml_engine.linear_regression import LinearRegressionGD


# Sample student-performance dataset.
# These are illustrative values for testing the implementation.
data = pd.DataFrame({
    "study_hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
        6, 8, 3, 5, 7, 9, 1, 3, 6, 9,
        2, 5, 8, 4, 7, 10, 3, 6, 9, 5
    ],
    "attendance": [
        55, 60, 65, 70, 75, 80, 85, 90, 58, 72,
        78, 92, 67, 76, 82, 95, 52, 63, 84, 96,
        61, 73, 91, 69, 86, 98, 64, 81, 94, 77
    ],
    "previous_marks": [
        40, 45, 48, 50, 55, 60, 65, 70, 43, 52,
        58, 74, 47, 57, 63, 78, 38, 46, 62, 80,
        44, 54, 72, 51, 67, 85, 49, 61, 77, 56
    ],
    "final_marks": [
        38, 42, 49, 53, 60, 65, 72, 78, 43, 54,
        64, 80, 47, 59, 69, 84, 34, 45, 66, 87,
        41, 57, 77, 52, 73, 91, 48, 63, 82, 61
    ]
})

feature_sets = {
    "Study Hours": ["study_hours"],
    "Study Hours + Attendance": [
        "study_hours", "attendance"
    ],
    "All Features": [
        "study_hours", "attendance", "previous_marks"
    ]
}

X_all = data.drop(columns=["final_marks"])
y_all = data["final_marks"]

# Use the same split for all experiments.
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.25,
    random_state=42
)

results = []

plt.figure(figsize=(10, 6))

for model_name, features in feature_sets.items():
    model = LinearRegressionGD(
        learning_rate=0.01,
        n_iterations=1000
    )

    model.fit(X_train[features], y_train)
    predictions = model.predict(X_test[features])

    mse = mean_squared_error(y_test, predictions)

    results.append({
        "Model": model_name,
        "Features": ", ".join(features),
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, predictions),
        "Final Training Cost": model.cost_history[-1]
    })

    plt.plot(
        range(1, len(model.cost_history) + 1),
        model.cost_history,
        label=model_name
    )

# Print comparison table.
results_df = pd.DataFrame(results)

print("\nFEATURE SELECTION COMPARISON")
print(results_df.round(4).to_string(index=False))

# Plot training cost for each feature combination.
plt.title("Cost Function Behaviour for Different Feature Sets")
plt.xlabel("Training Iterations")
plt.ylabel("Training MSE")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "experiments/feature_cost_comparison.png",
    dpi=300
)

plt.show()

# Compare test RMSE.
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Test RMSE Comparison by Feature Selection")
plt.xlabel("Model")
plt.ylabel("Test RMSE (lower is better)")
plt.xticks(rotation=10)
plt.tight_layout()

plt.savefig(
    "experiments/feature_rmse_comparison.png",
    dpi=300
)

plt.show()
