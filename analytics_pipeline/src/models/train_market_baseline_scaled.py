# this file trains a scaled baseline model using market features only
# scaling helps logistic regression converge and treat features fairly
# evaluation is time based to avoid lookahead bias

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import (
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    TRAIN_RATIO,
    MARKET_FEATURE_COLUMNS,
    get_confusion_matrix_path,
    get_evaluation_metrics_path,
    get_labeled_market_path,
)
from src.reproducibility import set_global_seed, sklearn_random_state
from src.models.evaluate import (
    build_confusion_matrix_dataframe,
    build_metrics_dataframe,
    make_time_based_split,
    print_evaluation_summary,
)


SYMBOL = DEFAULT_SYMBOL
TIMEFRAME = DEFAULT_TIMEFRAME
MODEL_NAME = "baseline_scaled"


def train_scaled_baseline_model():
    # deterministic seed so that re-running produces identical evaluation outputs
    set_global_seed()

    # load labeled dataset
    input_file = get_labeled_market_path(SYMBOL, TIMEFRAME)
    df = pd.read_csv(input_file, parse_dates=["open_time"])

    # sort by time to avoid leakage
    df = df.sort_values("open_time").reset_index(drop=True)

    # target labels
    y = df["label"]

    # explicit feature selection keeps the scaled path aligned with other models
    X = df[MARKET_FEATURE_COLUMNS].copy()

    # time based train test split
    X_train, X_test, y_train, y_test = make_time_based_split(X, y, TRAIN_RATIO)

    # build pipeline with scaling + logistic regression
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=sklearn_random_state(),
            ))
        ]
    )

    #train model
    model.fit(X_train, y_train)

    #evaluate model
    y_pred = model.predict(X_test)

    metrics_df = build_metrics_dataframe(y_test, y_pred, MODEL_NAME, SYMBOL, TIMEFRAME)
    confusion_df = build_confusion_matrix_dataframe(y_test, y_pred)

    metrics_df.to_csv(get_evaluation_metrics_path(SYMBOL, TIMEFRAME, MODEL_NAME), index=False)
    confusion_df.to_csv(get_confusion_matrix_path(SYMBOL, TIMEFRAME, MODEL_NAME))

    print_evaluation_summary("scaled baseline market model evaluation", y_test, y_pred)


if __name__ == "__main__":
    train_scaled_baseline_model()
