
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_5",
    "candle_body",
    "candle_range",
    "upper_wick",
    "lower_wick",
    "sma_5",
    "sma_10",
    "volatility_5"
]


def train_ml_model(data: pd.DataFrame):
    """
    Train and evaluate a Logistic Regression model
    using a chronological train/test split.

    The target column is never used as an input feature.
    """

    required = FEATURE_COLUMNS + ["target", "timestamp"]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    dataset = data.sort_values(
        "timestamp"
    ).reset_index(drop=True).copy()

    dataset = dataset.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    if len(dataset) < 30:
        raise ValueError(
            "At least 30 usable rows are required."
        )

    split_index = int(len(dataset) * 0.80)

    train = dataset.iloc[:split_index - 1]
    test = dataset.iloc[split_index:]

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"].astype(int)

    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data must contain both target classes."
        )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    baseline_prediction = int(y_train.mean() >= 0.5)

    baseline_accuracy = accuracy_score(
        y_test,
        [baseline_prediction] * len(y_test)
    )

    metrics = {
        "training_rows": len(train),
        "testing_rows": len(test),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "baseline_accuracy": baseline_accuracy
    }

    prediction_results = pd.DataFrame({
        "timestamp": test["timestamp"].values,
        "actual": y_test.values,
        "predicted": predictions
    })

    return model, metrics, prediction_results
    
def compare_ml_models(data: pd.DataFrame):
    """
    Compare Logistic Regression and Random Forest
    using the same chronological train/test split.
    """

    required = FEATURE_COLUMNS + ["target", "timestamp"]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    dataset = data.sort_values(
        "timestamp"
    ).reset_index(drop=True).copy()

    dataset = dataset.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    )

    if len(dataset) < 30:
        raise ValueError("At least 30 usable rows are required.")

    split_index = int(len(dataset) * 0.80)

    train = dataset.iloc[:split_index - 1]
    test = dataset.iloc[split_index:]

    if train.empty or test.empty:
        raise ValueError("Not enough data for comparison.")

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target"].astype(int)

    X_test = test[FEATURE_COLUMNS]
    y_test = test["target"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Training data needs both target classes.")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    }

    baseline_class = int(y_train.mean() >= 0.5)

    baseline_accuracy = accuracy_score(
        y_test,
        [baseline_class] * len(y_test)
    )

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_test, predictions
            ),
            "Balanced Accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "Baseline Accuracy": baseline_accuracy
        })

    return pd.DataFrame(results)

