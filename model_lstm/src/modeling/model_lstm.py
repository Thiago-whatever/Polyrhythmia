# src/modeling/model_lstm.py
from typing import List
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import Model

def build_lstm_model(
    vocab_size: int,
    steps_per_bar: int = 16,
    num_styles: int = 6,           # Jazz, Choro, Bossa, Samba, HipHop, AfroCuban (one-hot)
    embedding_dim: int = 64,
    lstm_units: List[int] = (128, 128),
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
    inject_mode: str = "concat",   # "concat" estilo+embedding
):
    """
    Modelo LSTM autoregresivo por compás:
    Inputs:
      - X_tokens: (batch, steps) ids de tokens (0..vocab_size-1)
      - X_style:  (batch, num_styles) vector binario de estilos (one-hot/multi-hot)
    Output:
      - Y_logits_softmax: (batch, steps, vocab_size) distrib. p(token_t | historial)
    """
    # Entradas
    inp_tokens = L.Input(shape=(steps_per_bar,), dtype="int32", name="X_tokens")
    inp_style  = L.Input(shape=(num_styles,),   dtype="float32", name="X_style")

    # Embedding de tokens (no usamos mask_zero para mantener estabilidad simple)
    x_tok = L.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embed")(inp_tokens)

    # Inyección de estilo: repetir estilo a lo largo del tiempo y concatenar
    s_rep = L.RepeatVector(steps_per_bar, name="style_repeat")(inp_style)  # (batch, steps, num_styles)
    if inject_mode == "concat":
        x = L.Concatenate(axis=-1, name="concat_tok_style")([x_tok, s_rep])
    else:
        x = x_tok  # (permite experimentar otros modos más adelante)

    # Pila LSTM(s)
    for i, units in enumerate(lstm_units):
        x = L.LSTM(
            units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name=f"lstm_{i+1}",
        )(x)

    # Capa de salida por paso temporal
    y = L.TimeDistributed(L.Dense(vocab_size, activation="softmax"), name="time_dense")(x)

    model = Model(inputs=[inp_tokens, inp_style], outputs=y, name="bar_lstm")
    return model


@tf.function
def perplexity(y_true, y_pred):
    """
    Perplejidad = exp(CE media). Se calcula por-batch.
    y_true: (batch, steps) int32
    y_pred: (batch, steps, vocab) float32 (probabilidades)
    """
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    # ce shape: (batch, steps) -> media
    ce_mean = tf.reduce_mean(ce)
    return tf.exp(ce_mean)
