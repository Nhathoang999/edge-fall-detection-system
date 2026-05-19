"""LSTM baseline for skeleton sequences."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import config


def create_lstm_classifier(
    input_shape: tuple[int, int] | None = None,
    lstm_units: int = config.LSTM_UNITS,
    dropout_rate: float = config.LSTM_DROPOUT,
    learning_rate: float = config.LEARNING_RATE,
) -> tf.keras.Model:
    if input_shape is None:
        input_shape = (config.INPUT_TIMESTEPS, config.NUM_FEATURES)

    f1_macro = tf.keras.metrics.F1Score(
        average="macro",
        threshold=0.5,
        name="f1_macro",
    )
    model = Sequential(name="lstm_fall_classifier")
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units // 2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", f1_macro],
    )
    return model
