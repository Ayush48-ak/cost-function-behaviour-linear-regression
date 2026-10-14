import base64
import warnings
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from django.shortcuts import render
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml_engine.linear_regression import LinearRegressionGD


MAX_UPLOAD_SIZE = 5 * 1024 * 1024
MAX_COLUMNS = 100
MAX_ITERATIONS = 10000

LEARNING_RATE_EXPERIMENTS = (0.001, 0.01, 0.1)


def validate_parameters(request):
    """Validate Gradient Descent configuration."""
    try:
        learning_rate = float(request.POST.get("learning_rate", "0.01"))
        iterations = int(request.POST.get("iterations", "1000"))
    except (TypeError, ValueError):
        raise ValueError(
            "Enter a valid learning rate and iteration count."
        )

    if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError(
            "Learning rate must be greater than 0 and at most 1."
        )

    if not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError(
            f"Iterations must be between 1 and {MAX_ITERATIONS}."
        )

    return learning_rate, iterations


def load_dataset(request):
    """Read and validate the uploaded CSV dataset."""
    uploaded_file = request.FILES.get("dataset")

    if not uploaded_file:
        raise ValueError("Please select your CSV file.")

    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    if uploaded_file.size > MAX_UPLOAD_SIZE:
        raise ValueError("The CSV file must be 5 MB or smaller.")

    try:
        df = pd.read_csv(uploaded_file)
    except (
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Unable to read the CSV file: {exc}"
        ) from exc

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    df.columns = df.columns.map(
        lambda name: str(name).strip()
    )

    if len(df.columns) < 2:
        raise ValueError(
            "The dataset needs at least two columns."
        )

    if len(df.columns) > MAX_COLUMNS:
        raise ValueError(
            f"The dataset cannot contain more than {MAX_COLUMNS} columns."
        )

    if (
        not df.columns.is_unique
        or any(not name for name in df.columns)
    ):
        raise ValueError(
            "Column names must be non-empty and unique."
        )

    numeric_df = df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric_columns = [
        column
        for column in df.columns
        if numeric_df[column].notna().any()
    ]

    if len(numeric_columns) < 2:
        raise ValueError(
            "At least two numeric columns are needed for regression."
        )

    return df, numeric_df, numeric_columns


def add_dataset_context(context, df, numeric_columns):
    """Add dataset summary and preview to the page."""
    context["columns"] = df.columns.tolist()
    context["numeric_columns"] = numeric_columns
    context["row_count"] = len(df)
    context["column_count"] = len(df.columns)

    context["preview"] = df.head(5).to_html(
        classes="table table-striped table-hover",
        index=False,
        border=0,
        escape=True,
    )


def create_cost_plot(
    cost_histories,
    title,
    log_scale=False,
):
    """Generate a cost-versus-iteration graph as a Base64 PNG."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plotted_any = False

    for label, costs in cost_histories:
        values = np.asarray(costs, dtype=float)

        finite_mask = np.isfinite(values)

        if log_scale:
            finite_mask &= values > 0

        finite_indices = np.flatnonzero(finite_mask)

        if finite_indices.size == 0:
            continue

        ax.plot(
            finite_indices + 1,
            values[finite_indices],
            label=label,
            linewidth=2,
        )

        plotted_any = True

    ax.set_title(title)
    ax.set_xlabel("Training Iterations")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.grid(True, alpha=0.3)

    if log_scale and plotted_any:
        ax.set_yscale("log")

    if len(cost_histories) > 1:
        ax.legend()

    if not plotted_any:
        ax.text(
            0.5,
            0.5,
            "No finite cost values were available to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    fig.tight_layout()

    image_buffer = BytesIO()

    fig.savefig(
        image_buffer,
        format="png",
        dpi=120,
    )

    plt.close(fig)

    return base64.b64encode(
        image_buffer.getvalue()
    ).decode("ascii")


def prepare_model_data(features, target, numeric_df):
    """Return complete, finite rows for the selected columns."""
    model_data = numeric_df[
        features + [target]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(model_data) < 4:
        raise ValueError(
            "At least four complete rows are required for the selected features."
        )

    return model_data


def evaluate_model(
    features,
    target,
    numeric_df,
    learning_rate,
    iterations,
):
    """Train and evaluate a model using a reproducible test split."""
    model_data = prepare_model_data(
        features,
        target,
        numeric_df,
    )

    X = model_data[features]
    y = model_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = LinearRegressionGD(
        learning_rate=learning_rate,
        n_iterations=iterations,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        with np.errstate(
            over="ignore",
            invalid="ignore",
            divide="ignore",
        ):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

    costs = list(model.cost_history)

    if (
        not costs
        or not np.isfinite(costs).all()
        or not np.isfinite(predictions).all()
    ):
        raise ValueError(
            "Training diverged for this learning rate. "
            "Try a smaller learning rate."
        )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    r2 = (
        r2_score(y_test, predictions)
        if len(y_test) >= 2
        else None
    )

    results = {
        "features": features,
        "feature_label": " + ".join(features),
        "target": target,
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "mae": float(
            mean_absolute_error(y_test, predictions)
        ),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": (
            float(r2)
            if r2 is not None and np.isfinite(r2)
            else None
        ),
        "initial_cost": float(costs[0]),
        "final_cost": float(costs[-1]),
    }

    return results, costs


def compare_feature_sets(
    numeric_df,
    target,
    learning_rate,
    iterations,
):
    """Compare each feature individually against all features together."""
    available_features = [
        column
        for column in numeric_df.columns
        if column != target
    ]

    if not available_features:
        raise ValueError(
            "At least one input feature is required."
        )

    feature_sets = [
        [feature]
        for feature in available_features
    ]

    if len(available_features) > 1:
        feature_sets.append(available_features)

    results = []
    cost_histories = []
    seen = set()

    for features in feature_sets:
        key = tuple(features)

        if key in seen:
            continue

        seen.add(key)

        try:
            result, costs = evaluate_model(
                features,
                target,
                numeric_df,
                learning_rate,
                iterations,
            )
        except (
            ValueError,
            FloatingPointError,
            OverflowError,
        ):
            continue

        results.append(result)

        cost_histories.append(
            (result["feature_label"], costs)
        )

    if not results:
        raise ValueError(
            "No feature combination trained successfully. "
            "Check the data or use a smaller learning rate."
        )

    results.sort(
        key=lambda result: result["rmse"]
    )

    return results, cost_histories


def compare_learning_rates(
    features,
    target,
    numeric_df,
    iterations,
):
    """Compare fixed learning rates using identical training and test rows."""
    model_data = prepare_model_data(
        features,
        target,
        numeric_df,
    )

    X = model_data[features]
    y = model_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    results = []
    cost_histories = []

    for rate in LEARNING_RATE_EXPERIMENTS:
        row = {
            "learning_rate": rate,
            "initial_cost": None,
            "final_cost": None,
            "cost_reduction": None,
            "cost_reduction_percent": None,
            "test_rmse": None,
            "status": "Failed",
            "status_class": "danger",
            "note": "",
        }

        try:
            model = LinearRegressionGD(
                learning_rate=rate,
                n_iterations=iterations,
            )

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    RuntimeWarning,
                )

                with np.errstate(
                    over="ignore",
                    invalid="ignore",
                    divide="ignore",
                ):
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

            costs = np.asarray(
                model.cost_history,
                dtype=float,
            )

            if not costs.size:
                row["note"] = "No cost values were recorded."
                results.append(row)
                continue

            row["initial_cost"] = (
                float(costs[0])
                if np.isfinite(costs[0])
                else None
            )

            # Keep the curve so the graph can show where divergence begins.
            cost_histories.append(
                (f"Learning rate {rate:g}", costs.tolist())
            )

            if (
                not np.isfinite(costs).all()
                or not np.isfinite(predictions).all()
            ):
                row["status"] = "Diverged"
                row["status_class"] = "danger"
                row["note"] = (
                    "Cost or predictions became non-finite."
                )

                results.append(row)
                continue

            initial_cost = float(costs[0])
            final_cost = float(costs[-1])

            reduction = initial_cost - final_cost

            reduction_percent = (
                (reduction / initial_cost) * 100
                if initial_cost > 0
                else None
            )

            test_rmse = float(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions,
                    )
                )
            )

            row.update({
                "final_cost": final_cost,
                "cost_reduction": float(reduction),
                "cost_reduction_percent": (
                    float(reduction_percent)
                    if (
                        reduction_percent is not None
                        and np.isfinite(reduction_percent)
                    )
                    else None
                ),
                "test_rmse": test_rmse,
            })

            if final_cost < initial_cost:
                row["status"] = "Cost decreased"
                row["status_class"] = "success"
                row["note"] = (
                    "Training cost decreased over the experiment."
                )

            elif np.isclose(final_cost, initial_cost):
                row["status"] = "Little change"
                row["status_class"] = "warning"
                row["note"] = (
                    "Training cost changed very little."
                )

            else:
                row["status"] = "Cost increased"
                row["status_class"] = "warning"
                row["note"] = (
                    "Try a smaller learning rate or inspect target scale."
                )

        except (
            ValueError,
            FloatingPointError,
            OverflowError,
        ) as exc:
            row["note"] = str(exc)

        results.append(row)

    return results, cost_histories


def dashboard(request):
    """Handle dashboard upload and experiment forms."""
    context = {}

    if request.method != "POST":
        return render(
            request,
            "analyzer/dashboard.html",
            context,
        )

    action = request.POST.get("action", "upload")

    try:
        df, numeric_df, numeric_columns = load_dataset(request)

        add_dataset_context(
            context,
            df,
            numeric_columns,
        )

        if action == "upload":
            context["message"] = (
                "Dataset loaded successfully. Configure your experiment below."
            )

        elif action in (
            "train",
            "compare",
            "learning_rate",
        ):
            target = request.POST.get("target")

            if target not in numeric_columns:
                raise ValueError(
                    "Select a valid numeric target column."
                )

            learning_rate, iterations = validate_parameters(
                request
            )

            if action == "train":
                features = request.POST.getlist("features")

                if not features:
                    raise ValueError(
                        "Select at least one input feature."
                    )

                if len(features) != len(set(features)):
                    raise ValueError(
                        "Duplicate features are not allowed."
                    )

                if any(
                    feature not in numeric_columns
                    for feature in features
                ):
                    raise ValueError(
                        "All selected features must be numeric."
                    )

                if target in features:
                    raise ValueError(
                        "The target cannot also be an input feature."
                    )

                results, costs = evaluate_model(
                    features,
                    target,
                    numeric_df,
                    learning_rate,
                    iterations,
                )

                context["results"] = results

                context["cost_plot"] = create_cost_plot(
                    [("Training MSE", costs)],
                    "Cost Function Behaviour During Training",
                )

                context["selected_features"] = features
                context["selected_target"] = target

                context["message"] = (
                    "Model training completed successfully."
                )

            elif action == "compare":
                results, costs = compare_feature_sets(
                    numeric_df,
                    target,
                    learning_rate,
                    iterations,
                )

                context["comparison_results"] = results

                context["comparison_plot"] = create_cost_plot(
                    costs,
                    "Cost Function Behaviour for Different Feature Sets",
                )

                context["selected_target"] = target

                context["message"] = (
                    "Feature combination comparison completed successfully."
                )

            elif action == "learning_rate":
                features = request.POST.getlist("features")

                if not features:
                    raise ValueError(
                        "Select at least one input feature "
                        "for the learning-rate experiment."
                    )

                if len(features) != len(set(features)):
                    raise ValueError(
                        "Duplicate features are not allowed."
                    )

                if any(
                    feature not in numeric_columns
                    for feature in features
                ):
                    raise ValueError(
                        "All selected features must be numeric."
                    )

                if target in features:
                    raise ValueError(
                        "The target cannot also be an input feature."
                    )

                experiment_results, costs = compare_learning_rates(
                    features,
                    target,
                    numeric_df,
                    iterations,
                )

                context["learning_rate_results"] = (
                    experiment_results
                )

                context["learning_rate_plot"] = create_cost_plot(
                    costs,
                    "Cost Function Behaviour at Different Learning Rates",
                    log_scale=True,
                )

                context["learning_rate_features"] = features
                context["selected_features"] = features
                context["selected_target"] = target

                context["message"] = (
                    "Learning-rate experiment completed. "
                    "Review each status before choosing a rate."
                )

        else:
            raise ValueError(
                "Invalid action. Please try again."
            )

    except (
        ValueError,
        FloatingPointError,
        OverflowError,
    ) as exc:
        context["error"] = str(exc)

    return render(
        request,
        "analyzer/dashboard.html",
        context,
    )