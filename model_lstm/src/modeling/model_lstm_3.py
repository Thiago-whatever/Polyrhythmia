# src/modeling/model_lstm_3.py
import tensorflow as tf
from tensorflow.keras import layers as L, regularizers as R
from src.modeling.model_lstm_2 import SparseCELS  # reutiliza tu loss con LS

def build_lstm_model_3(
    vocab_size: int,
    seq_len: int = 16,
    styles: int = 6,
    token_emb: int = 128,
    pos_emb: int = 16,
    style_dim: int = 32,
    lstm_units=(384,384),
    dropout=0.35,
    recurrent_dropout=0.1,
    l2_dense=2e-5,
    lr=1e-3,
    label_smoothing=0.12,
):
    tokens_in = L.Input(shape=(seq_len,), name="tokens")
    pos_in    = L.Input(shape=(seq_len,), name="pos")
    style_in  = L.Input(shape=(styles,), name="style")

    tokE = L.Embedding(vocab_size, token_emb, name="tok_emb")(tokens_in)
    posE = L.Embedding(seq_len + 1, pos_emb, name="pos_emb")(pos_in)

    style_proj = L.Dense(style_dim, name="style_proj")(style_in)
    style_rep  = L.RepeatVector(seq_len, name="style_tiled")(style_proj)

    x = L.Concatenate(name="feat_concat")([tokE, posE, style_rep])
    for i, u in enumerate(lstm_units):
        x = L.LSTM(u, return_sequences=True, dropout=dropout,
                   recurrent_dropout=recurrent_dropout, name=f"lstm{i+1}")(x)

    out = L.TimeDistributed(
        L.Dense(vocab_size, activation="softmax",
                kernel_regularizer=R.l2(l2_dense)),
        name="logits_softmax")(x)

    model = tf.keras.Model(inputs=[tokens_in, pos_in, style_in], outputs=out, name="lstm_v3")
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    model.compile(optimizer=opt, loss=SparseCELS(ls=label_smoothing),
                  metrics=["sparse_categorical_accuracy"])
    return model
