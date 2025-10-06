# src/modeling/model_lstm_2.py
import tensorflow as tf
from tensorflow.keras import layers as L, regularizers as R, losses as KLoss

import keras

@keras.saving.register_keras_serializable(package="Custom")
class SparseCELS(tf.keras.losses.Loss):
    """Sparse Categorical Cross-Entropy con label smoothing (serializable)."""
    def __init__(self, ls=0.05, reduction=tf.keras.losses.Reduction.AUTO, name="sparse_ce_ls"):
        super().__init__(reduction=reduction, name=name)
        self.ls = ls

    def call(self, y_true, y_pred):
        V = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=V)
        y_true_ls = (1.0 - self.ls) * y_true_oh + self.ls / tf.cast(V, tf.float32)
        ce = -tf.reduce_sum(
            y_true_ls * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)),
            axis=-1
        )
        return tf.reduce_mean(ce)

    def get_config(self):
        config = super().get_config()
        config.update({"ls": self.ls})
        return config

def perplexity(y_true, y_pred):
    # y_pred ya softmax; ppx = exp(loss)
    loss = KLoss.sparse_categorical_crossentropy(y_true, y_pred)
    return tf.exp(tf.reduce_mean(loss))

def build_lstm_model_2(
    vocab_size: int,
    styles: int = 6,
    style_dim: int = 8,      # tamaño del embedding de estilo
    token_emb: int = 128,
    pos_emb: int = 16,
    lstm_units1: int = 384,
    lstm_units2: int = 384,
    l2_dense: float = 1e-5,
    dropout: float = 0.3,
    recurrent_dropout: float = 0.1,
    clipnorm: float = 1.0,
    label_smoothing: float = 0.05,
    seq_len: int = 16,
):
    """
    Inputs:
      - tokens (B, T)   : ids de eventos
      - pos    (B, T)   : 1..16 posiciones de compás
      - style  (B, S)   : vector binario de estilos; se embebe y se repite a T pasos
    """
    tokens_in = L.Input(shape=(seq_len,), name="tokens")
    pos_in    = L.Input(shape=(seq_len,), name="pos")               # enteros 1..16
    style_in  = L.Input(shape=(styles,), name="style")                 # S (6)

    tokE = L.Embedding(input_dim=vocab_size, output_dim=token_emb, name="tok_emb")(tokens_in)
    posE = L.Embedding(input_dim=seq_len + 1, output_dim=pos_emb, name="pos_emb")(pos_in)  # 1..16

    # Proyección lineal del vector de estilos y repetición a T pasos
    style_proj = L.Dense(style_dim, activation="linear", name="style_proj")(style_in)     # (B, style_dim)
    style_rep  = L.RepeatVector(seq_len, name="style_tiled")(style_proj)  

    x = L.Concatenate(name="concat_feats")([tokE, posE, style_rep])

    x = L.LSTM(lstm_units1, return_sequences=True, dropout=dropout,
               recurrent_dropout=recurrent_dropout, name="lstm1")(x)
    x = L.LSTM(lstm_units2, return_sequences=True, dropout=dropout,
               recurrent_dropout=recurrent_dropout, name="lstm2")(x)

    out = L.TimeDistributed(
        L.Dense(vocab_size, activation="softmax",
                kernel_regularizer=R.l2(l2_dense)),
        name="logits_softmax"
    )(x)

    model = tf.keras.Model(inputs=[tokens_in, pos_in, style_in], outputs=out, name="lstm_improved")

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=clipnorm)
    model.compile(
        optimizer=opt,
        loss=SparseCELS(ls=label_smoothing), 
        metrics=["sparse_categorical_accuracy", perplexity],
    )


    return model
