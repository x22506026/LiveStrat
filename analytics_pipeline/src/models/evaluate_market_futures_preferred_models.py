"""Evaluate preferred market + futures targets with logistic and LSTM models."""

import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_TIMEFRAME,
    TRAIN_RATIO,
    get_confusion_matrix_path,
    get_evaluation_metrics_path,
    get_market_futures_dataset_path,
    get_market_futures_preferred_model_summary_path,
)
from src.models.evaluate import (
    build_confusion_matrix_dataframe,
    build_metrics_dataframe,
    print_evaluation_summary,
)
from src.models.evaluate_market_futures_strategies import (
    build_combined_feature_matrix,
    train_scaled_logistic_baseline,
)
from src.models.market_futures_targets import (
    build_target_labels,
    get_preferred_market_futures_targets,
)


TIMEFRAME = DEFAULT_TIMEFRAME
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
SEQUENCE_WINDOW = 12
LAG_WINDOW = 12


def set_reproducible_seeds(seed=42):
    """Set seeds for reproducible classical and neural runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def evaluate_prediction_vectors(y_true, y_pred, symbol, model_name, timeframe, start_date, end_date):
    """Persist evaluation outputs and return headline metrics."""
    metrics_df = build_metrics_dataframe(y_true, y_pred, model_name, symbol, timeframe)
    confusion_df = build_confusion_matrix_dataframe(y_true, y_pred)

    metrics_df.to_csv(
        get_evaluation_metrics_path(symbol, timeframe, model_name, start_date, end_date),
        index=False,
    )
    confusion_df.to_csv(
        get_confusion_matrix_path(symbol, timeframe, model_name, start_date, end_date)
    )
    print_evaluation_summary(f"{symbol} {model_name} evaluation", y_true, y_pred)

    accuracy = float(metrics_df.loc[metrics_df["metric"] == "accuracy", "value"].iloc[0])
    macro_f1 = float(metrics_df.loc[metrics_df["metric"] == "macro_f1", "value"].iloc[0])
    balanced_accuracy = float(metrics_df.loc[metrics_df["metric"] == "balanced_accuracy", "value"].iloc[0])
    return accuracy, macro_f1, balanced_accuracy


def build_futures_context_snapshot(df):
    """Capture the latest futures-support state for one evaluation window."""
    if df.empty:
        return {}
    latest_row = df.sort_values("open_time").iloc[-1]
    return {
        "futures_feature_completeness_score": latest_row.get("futures_feature_completeness_score"),
        "futures_completeness_label": latest_row.get("futures_completeness_label"),
        "futures_context_resilience_score": latest_row.get("futures_context_resilience_score"),
        "futures_context_resilience_label": latest_row.get("futures_context_resilience_label"),
        "futures_basis_reliance_score": latest_row.get("futures_basis_reliance_score"),
        "basis_feature_available": latest_row.get("basis_feature_available"),
    }


def build_sequence_arrays(X, y, split_idx, window_size=SEQUENCE_WINDOW):
    """Build train/test rolling windows without leaking future information."""
    X_values = X.to_numpy(dtype=np.float32)
    y_values = y.to_numpy()

    train_sequences = []
    train_labels = []
    test_sequences = []
    test_labels = []

    for idx in range(window_size - 1, len(X_values)):
        sequence = X_values[idx - window_size + 1: idx + 1]
        label = y_values[idx]
        if idx < split_idx:
            train_sequences.append(sequence)
            train_labels.append(label)
        else:
            test_sequences.append(sequence)
            test_labels.append(label)

    return (
        np.asarray(train_sequences, dtype=np.float32),
        np.asarray(test_sequences, dtype=np.float32),
        np.asarray(train_labels),
        np.asarray(test_labels),
    )


def scale_sequence_arrays(X_train_seq, X_test_seq):
    """Standardize sequence features using only the training window statistics."""
    feature_mean = X_train_seq.reshape(-1, X_train_seq.shape[-1]).mean(axis=0)
    feature_std = X_train_seq.reshape(-1, X_train_seq.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    X_train_scaled = (X_train_seq - feature_mean) / feature_std
    X_test_scaled = (X_test_seq - feature_mean) / feature_std
    return X_train_scaled, X_test_scaled


def build_lstm_model(window_size, feature_count, class_count):
    """Create a compact LSTM classifier for the preferred target benchmark."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, feature_count)),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=0.0),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lagged_logistic_baseline(X_train, y_train):
    """Train the preferred balanced logistic baseline on lagged tabular features."""
    return train_scaled_logistic_baseline(X_train, y_train)


def train_lagged_random_forest_baseline(X_train, y_train):
    """Train a nonlinear lagged tabular benchmark."""
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def build_lagged_feature_frames(X, y, split_idx, window_size=LAG_WINDOW):
    """Flatten rolling windows into lagged tabular features."""
    lagged_rows = []
    lagged_labels = []
    lagged_indices = []
    feature_names = list(X.columns)

    for idx in range(window_size - 1, len(X)):
        window = X.iloc[idx - window_size + 1: idx + 1].to_numpy()
        lagged_rows.append(window.reshape(-1))
        lagged_labels.append(y.iloc[idx])
        lagged_indices.append(idx)

    lagged_columns = []
    for lag in range(window_size):
        lag_suffix = f"t_minus_{window_size - lag - 1}"
        for feature_name in feature_names:
            lagged_columns.append(f"{feature_name}_{lag_suffix}")

    lagged_df = pd.DataFrame(lagged_rows, columns=lagged_columns)
    lagged_y = pd.Series(lagged_labels, name="label")
    lagged_indices = pd.Series(lagged_indices, name="source_index")

    train_mask = lagged_indices < split_idx
    X_train_lagged = lagged_df.loc[train_mask].reset_index(drop=True)
    X_test_lagged = lagged_df.loc[~train_mask].reset_index(drop=True)
    y_train_lagged = lagged_y.loc[train_mask].reset_index(drop=True)
    y_test_lagged = lagged_y.loc[~train_mask].reset_index(drop=True)
    return X_train_lagged, X_test_lagged, y_train_lagged, y_test_lagged, lagged_df


def evaluate_preferred_models_for_symbol(symbol, target_config, timeframe, start_date, end_date):
    """Evaluate the preferred target for one symbol using logistic and LSTM models."""
    dataset_path = get_market_futures_dataset_path(symbol, timeframe, start_date, end_date)
    df = pd.read_csv(dataset_path, parse_dates=["open_time", "close_time"])
    df = df[df["futures_data_available"] == True].copy()
    df = df.sort_values("open_time").reset_index(drop=True)
    df = build_target_labels(df, target_config, timeframe=timeframe)
    futures_context = build_futures_context_snapshot(df)

    X, feature_names = build_combined_feature_matrix(df)
    y = df["label"].copy()
    split_idx = int(len(X) * TRAIN_RATIO)
    feature_medians = X.iloc[:split_idx].median(numeric_only=True)
    X = X.fillna(feature_medians)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    summary_rows = []

    logistic_model = train_scaled_logistic_baseline(X_train, y_train)
    logistic_pred = logistic_model.predict(X_test)
    logistic_name = f"market_futures_logistic_preferred_{target_config['target_name']}"
    logistic_accuracy, logistic_macro_f1, logistic_balanced = evaluate_prediction_vectors(
        y_test,
        logistic_pred,
        symbol,
        logistic_name,
        timeframe,
        start_date,
        end_date,
    )
    logistic_latest_signal = logistic_model.predict(X.iloc[[-1]])[0]
    logistic_latest_confidence = float(logistic_model.predict_proba(X.iloc[[-1]]).max())
    summary_rows.append(
        {
            "symbol": symbol,
            "model_name": logistic_name,
            "target_name": target_config["target_name"],
            "label_mode": target_config["label_mode"],
            "horizon_steps": target_config["horizon_steps"],
            "target_horizon_hours": target_config.get("effective_horizon_hours"),
            "target_exact_horizon_match": target_config.get("exact_horizon_match"),
            "target_resolution_note": target_config.get("horizon_resolution_note"),
            "test_accuracy": logistic_accuracy,
            "test_macro_f1": logistic_macro_f1,
            "test_balanced_accuracy": logistic_balanced,
            "latest_signal": logistic_latest_signal,
            "latest_signal_confidence": logistic_latest_confidence,
            "feature_count": len(feature_names),
            "sequence_window": 1,
            **futures_context,
        }
    )

    if len(X_train) >= LAG_WINDOW and len(X_test) >= 1:
        (
            X_train_lagged,
            X_test_lagged,
            y_train_lagged,
            y_test_lagged,
            lagged_df,
        ) = build_lagged_feature_frames(X, y, split_idx, LAG_WINDOW)

        lagged_logistic_model = train_lagged_logistic_baseline(X_train_lagged, y_train_lagged)
        lagged_logistic_pred = lagged_logistic_model.predict(X_test_lagged)
        lagged_logistic_name = f"market_futures_lagged_logistic_preferred_{target_config['target_name']}"
        (
            lagged_logistic_accuracy,
            lagged_logistic_macro_f1,
            lagged_logistic_balanced,
        ) = evaluate_prediction_vectors(
            y_test_lagged,
            lagged_logistic_pred,
            symbol,
            lagged_logistic_name,
            timeframe,
            start_date,
            end_date,
        )
        lagged_logistic_latest_signal = lagged_logistic_model.predict(lagged_df.iloc[[-1]])[0]
        lagged_logistic_latest_confidence = float(
            lagged_logistic_model.predict_proba(lagged_df.iloc[[-1]]).max()
        )
        summary_rows.append(
            {
                "symbol": symbol,
                "model_name": lagged_logistic_name,
                "target_name": target_config["target_name"],
                "label_mode": target_config["label_mode"],
                "horizon_steps": target_config["horizon_steps"],
                "target_horizon_hours": target_config.get("effective_horizon_hours"),
                "target_exact_horizon_match": target_config.get("exact_horizon_match"),
                "target_resolution_note": target_config.get("horizon_resolution_note"),
                "test_accuracy": lagged_logistic_accuracy,
                "test_macro_f1": lagged_logistic_macro_f1,
                "test_balanced_accuracy": lagged_logistic_balanced,
                "latest_signal": lagged_logistic_latest_signal,
                "latest_signal_confidence": lagged_logistic_latest_confidence,
                "feature_count": lagged_df.shape[1],
                "sequence_window": LAG_WINDOW,
                **futures_context,
            }
        )

        lagged_forest_model = train_lagged_random_forest_baseline(X_train_lagged, y_train_lagged)
        lagged_forest_pred = lagged_forest_model.predict(X_test_lagged)
        lagged_forest_name = f"market_futures_lagged_forest_preferred_{target_config['target_name']}"
        (
            lagged_forest_accuracy,
            lagged_forest_macro_f1,
            lagged_forest_balanced,
        ) = evaluate_prediction_vectors(
            y_test_lagged,
            lagged_forest_pred,
            symbol,
            lagged_forest_name,
            timeframe,
            start_date,
            end_date,
        )
        lagged_forest_latest_signal = lagged_forest_model.predict(lagged_df.iloc[[-1]])[0]
        lagged_forest_probabilities = lagged_forest_model.predict_proba(lagged_df.iloc[[-1]])
        lagged_forest_latest_confidence = float(np.max(lagged_forest_probabilities))
        summary_rows.append(
            {
                "symbol": symbol,
                "model_name": lagged_forest_name,
                "target_name": target_config["target_name"],
                "label_mode": target_config["label_mode"],
                "horizon_steps": target_config["horizon_steps"],
                "target_horizon_hours": target_config.get("effective_horizon_hours"),
                "target_exact_horizon_match": target_config.get("exact_horizon_match"),
                "target_resolution_note": target_config.get("horizon_resolution_note"),
                "test_accuracy": lagged_forest_accuracy,
                "test_macro_f1": lagged_forest_macro_f1,
                "test_balanced_accuracy": lagged_forest_balanced,
                "latest_signal": lagged_forest_latest_signal,
                "latest_signal_confidence": lagged_forest_latest_confidence,
                "feature_count": lagged_df.shape[1],
                "sequence_window": LAG_WINDOW,
                **futures_context,
            }
        )

    if len(X_train) < SEQUENCE_WINDOW or len(X_test) < 1:
        summary_rows.append(
            {
                "symbol": symbol,
                "model_name": f"market_futures_lstm_preferred_{target_config['target_name']}",
                "target_name": target_config["target_name"],
                "label_mode": target_config["label_mode"],
                "horizon_steps": target_config["horizon_steps"],
                "target_horizon_hours": target_config.get("effective_horizon_hours"),
                "target_exact_horizon_match": target_config.get("exact_horizon_match"),
                "target_resolution_note": target_config.get("horizon_resolution_note"),
                "status": "skipped_insufficient_sequence_length",
                "feature_count": len(feature_names),
                "sequence_window": SEQUENCE_WINDOW,
                **futures_context,
            }
        )
        return summary_rows

    X_train_seq, X_test_seq, y_train_seq, y_test_seq = build_sequence_arrays(X, y, split_idx, SEQUENCE_WINDOW)
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        summary_rows.append(
            {
                "symbol": symbol,
                "model_name": f"market_futures_lstm_preferred_{target_config['target_name']}",
                "target_name": target_config["target_name"],
                "label_mode": target_config["label_mode"],
                "horizon_steps": target_config["horizon_steps"],
                "target_horizon_hours": target_config.get("effective_horizon_hours"),
                "target_exact_horizon_match": target_config.get("exact_horizon_match"),
                "target_resolution_note": target_config.get("horizon_resolution_note"),
                "status": "skipped_no_sequences",
                "feature_count": len(feature_names),
                "sequence_window": SEQUENCE_WINDOW,
                **futures_context,
            }
        )
        return summary_rows

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_seq)
    y_test_encoded = label_encoder.transform(y_test_seq)
    X_train_scaled, X_test_scaled = scale_sequence_arrays(X_train_seq, X_test_seq)

    set_reproducible_seeds()
    lstm_model = build_lstm_model(SEQUENCE_WINDOW, X_train_scaled.shape[-1], len(label_encoder.classes_))
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )
    lstm_model.fit(
        X_train_scaled,
        y_train_encoded,
        epochs=40,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stopping],
    )

    y_test_probs = lstm_model.predict(X_test_scaled, verbose=0)
    y_test_pred = label_encoder.inverse_transform(np.argmax(y_test_probs, axis=1))
    lstm_name = f"market_futures_lstm_preferred_{target_config['target_name']}"
    lstm_accuracy, lstm_macro_f1, lstm_balanced = evaluate_prediction_vectors(
        y_test_seq,
        y_test_pred,
        symbol,
        lstm_name,
        timeframe,
        start_date,
        end_date,
    )
    latest_sequence = X.iloc[-SEQUENCE_WINDOW:].to_numpy(dtype=np.float32)
    latest_sequence = latest_sequence.reshape(1, SEQUENCE_WINDOW, X.shape[1])
    latest_sequence = (latest_sequence - X_train_seq.reshape(-1, X_train_seq.shape[-1]).mean(axis=0)) / np.where(
        X_train_seq.reshape(-1, X_train_seq.shape[-1]).std(axis=0) == 0,
        1.0,
        X_train_seq.reshape(-1, X_train_seq.shape[-1]).std(axis=0),
    )
    latest_probs = lstm_model.predict(latest_sequence, verbose=0)[0]
    latest_signal = label_encoder.inverse_transform([int(np.argmax(latest_probs))])[0]
    latest_confidence = float(np.max(latest_probs))
    summary_rows.append(
        {
            "symbol": symbol,
            "model_name": lstm_name,
            "target_name": target_config["target_name"],
            "label_mode": target_config["label_mode"],
            "horizon_steps": target_config["horizon_steps"],
            "target_horizon_hours": target_config.get("effective_horizon_hours"),
            "target_exact_horizon_match": target_config.get("exact_horizon_match"),
            "target_resolution_note": target_config.get("horizon_resolution_note"),
            "test_accuracy": lstm_accuracy,
            "test_macro_f1": lstm_macro_f1,
            "test_balanced_accuracy": lstm_balanced,
            "latest_signal": latest_signal,
            "latest_signal_confidence": latest_confidence,
            "feature_count": len(feature_names),
            "sequence_window": SEQUENCE_WINDOW,
            **futures_context,
        }
    )

    return summary_rows


def evaluate_market_futures_preferred_models(timeframe=TIMEFRAME, start_date=START_DATE, end_date=END_DATE):
    """Evaluate preferred targets using the current static and temporal benchmarks."""
    summary_rows = []
    preferred_targets = get_preferred_market_futures_targets(timeframe)

    for symbol, target_config in preferred_targets.items():
        summary_rows.extend(
            evaluate_preferred_models_for_symbol(
                symbol,
                target_config,
                timeframe,
                start_date,
                end_date,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = get_market_futures_preferred_model_summary_path(timeframe, start_date, end_date)
    summary_df.to_csv(summary_path, index=False)

    print("market + futures preferred model summary generated")
    print(f"rows saved: {len(summary_df)}")
    print(f"summary saved to: {summary_path}")
    return summary_df


if __name__ == "__main__":
    evaluate_market_futures_preferred_models()
