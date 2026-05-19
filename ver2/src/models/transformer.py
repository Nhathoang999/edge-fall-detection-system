"""Transformer classifier for skeleton sequences."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import config


def transformer_encoder_block(
    inputs,
    d_model: int,
    num_heads: int,
    ff_dim: int,
    dropout_rate: float = 0.1,
    name_prefix: str = "",
):
    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        name=f"{name_prefix}_mha",
    )(inputs, inputs, inputs)
    attn = Dropout(dropout_rate, name=f"{name_prefix}_mha_dropout")(attn)
    out1 = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_layernorm1")(inputs + attn)

    ffn = Dense(ff_dim, activation="relu", name=f"{name_prefix}_ffn_dense1")(out1)
    ffn = Dense(d_model, name=f"{name_prefix}_ffn_dense2")(ffn)
    ffn = Dropout(dropout_rate, name=f"{name_prefix}_ffn_dropout")(ffn)
    out2 = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_layernorm2")(out1 + ffn)
    return out2


def positional_embedding(seq_len: int, d_model: int, name_prefix: str = ""):
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_2d = Embedding(
        input_dim=seq_len,
        output_dim=d_model,
        name=f"{name_prefix}_pos_embed",
    )(positions)
    return tf.expand_dims(pos_2d, axis=0)


def create_transformer_classifier(
    input_shape: tuple[int, int] | None = None,
    num_encoder_blocks: int = config.NUM_ENCODER_BLOCKS,
    d_model: int = config.D_MODEL,
    num_heads: int = config.NUM_HEADS,
    ff_dim: int = config.FF_DIM,
    projection_dim: int | None = None,
    final_dense_units: int = config.FINAL_DENSE_UNITS,
    dropout_rate: float = config.DROPOUT_RATE,
    learning_rate: float = config.LEARNING_RATE,
) -> Model:
    if input_shape is None:
        input_shape = (config.INPUT_TIMESTEPS, config.NUM_FEATURES)
    if projection_dim is None:
        projection_dim = d_model

    timesteps, _ = input_shape
    inputs = Input(shape=input_shape, name="input_features")
    x = Dense(projection_dim, name="feature_projection")(inputs)
    pos = positional_embedding(timesteps, projection_dim, name_prefix="pos_enc")
    x = Add(name="add_positional_encoding")([x, pos])
    x = Dropout(dropout_rate, name="input_dropout_after_pos_enc")(x)

    for i in range(num_encoder_blocks):
        x = transformer_encoder_block(
            x,
            projection_dim,
            num_heads,
            ff_dim,
            dropout_rate,
            name_prefix=f"encoder_block_{i + 1}",
        )

    x = GlobalAveragePooling1D(name="global_avg_pooling")(x)
    x = Dropout(dropout_rate, name="dropout_after_pooling")(x)
    x = Dense(final_dense_units, activation="relu", name="final_dense_1")(x)
    x = Dropout(dropout_rate / 2, name="dropout_final_dense")(x)
    outputs = Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    f1_macro = tf.keras.metrics.F1Score(
        average="macro",
        threshold=0.5,
        name="f1_macro",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", f1_macro],
    )
    return model
