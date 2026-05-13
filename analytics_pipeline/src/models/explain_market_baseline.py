# this file analyses feature importance for the scaled market baseline model
# it extracts logistic regression coefficients to explain model decisions
# no new training logic is introduced beyond the baseline setup

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import (
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    TRAIN_RATIO,
    MARKET_FEATURE_COLUMNS,
    get_labeled_market_path,
    get_market_coefficients_path,
    get_market_feature_importance_path,
)


SYMBOL = DEFAULT_SYMBOL
TIMEFRAME = DEFAULT_TIMEFRAME


def explain_baseline_model():
    # load labeled dataset
    input_file = get_labeled_market_path(SYMBOL, TIMEFRAME)
    df = pd.read_csv(input_file, parse_dates=["open_time"])

    # sort by time
    df = df.sort_values("open_time").reset_index(drop=True)

    # target labels
    y = df["label"]

    # explicit feature selection keeps explainability aligned with model training
    X = df[MARKET_FEATURE_COLUMNS].copy()

    feature_names = X.columns.tolist()

    # time based split
    split_idx = int(len(df) * TRAIN_RATIO)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    # train scaled logistic regression
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                solver="lbfgs"
            ))
        ]
    )

    model.fit(X_train, y_train)

    # extract coefficients
    clf = model.named_steps["classifier"]
    coef = clf.coef_

    # build explainability dataframe
    coef_df = pd.DataFrame(
        coef,
        columns=feature_names,
        index=clf.classes_
    )

    #absolute importance for ranking
    abs_importance = coef_df.abs().mean(axis=0).sort_values(ascending=False)

    #save outputs
    coef_output = get_market_coefficients_path(SYMBOL, TIMEFRAME)
    importance_output = get_market_feature_importance_path(SYMBOL, TIMEFRAME)

    coef_df.to_csv(coef_output)
    abs_importance.to_csv(importance_output, header=["importance"])

    #print summary
    print("explainability analysis completed")
    print("top 10 most influential features:")
    print(abs_importance.head(10))


if __name__ == "__main__":
    explain_baseline_model()
