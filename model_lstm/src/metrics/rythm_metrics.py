import numpy as np
import tensorflow as tf

def tokens_to_multihot(seq, vocab_map, n_instruments=9):
    """
    seq: (T,) tokens; vocab_map: dict[token_id] -> list(indices_de_instrumentos)
    Devuelve (T, 9) binario
    """
    T = len(seq)
    M = np.zeros((T, n_instruments), dtype=np.int32)
    for t, tok in enumerate(seq):
        for inst in vocab_map.get(int(tok), []):
            M[t, inst] = 1
    return M

def note_density(M):
    # % de steps con al menos un golpe
    hits = (M.sum(axis=1) > 0).mean()
    return float(hits)

def f1_per_instrument(M_pred, M_true, eps=1e-8):
    # F1 promedio
    TP = (M_pred & M_true).sum(axis=0).astype(np.float32)
    FP = (M_pred & (1 - M_true)).sum(axis=0).astype(np.float32)
    FN = ((1 - M_pred) & M_true).sum(axis=0).astype(np.float32)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(np.mean(f1))

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

def density(M):   # M: (T,9)
    return float((M.sum(axis=1) > 0).mean())

def syncopation(M):
    T = M.shape[0]
    on  = M[::2].sum() / max(1, M[::2].size)
    off = M[1::2].sum() / max(1, M[1::2].size)
    return float(off / (on + 1e-6))

def cooccurrence(M):
    C = (M.T @ M) / M.shape[0]  # (9,9)
    return C

