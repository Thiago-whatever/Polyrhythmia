# src/train/train_lstm_3.py
import os, json, time, argparse, datetime, pathlib as P
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, ModelCheckpoint

from src.modeling.model_lstm_3 import build_lstm_model_3
from src.modeling.model_lstm_2 import SparseCELS  # para cargar checkpoints con loss
from src.metrics.rythm_metrics import perplexity    # ya lo tienes
# ─────────────────────────────────────────────────────────────────

def load_npz_full(path):
    d = np.load(path, allow_pickle=True)
    # X/Y
    if "X_tokens" in d and "Y_tokens" in d:
        X, Y = d["X_tokens"], d["Y_tokens"]
    elif "X" in d and "Y" in d:
        X, Y = d["X"], d["Y"]
    else:
        raise KeyError(f"{path} sin X/Y esperados")
    # G (estilos)
    G = d["X_style"] if "X_style" in d.files else None
    # vocab_size (opcional)
    vs = None
    for k in ["vocab_size", "num_classes"]:
        if k in d.files: vs = int(d[k])
    return X, Y, G, vs

def make_positional(N, T):
    return np.tile(np.arange(1, T+1, dtype=np.int32), (N, 1))

def build_class_weights(y, vocab_size, max_weight=5.0):
    counts = np.bincount(y.ravel(), minlength=vocab_size).astype(np.float64)
    probs = counts / (counts.sum() + 1e-8)
    inv = 1.0 / np.maximum(probs, 1e-8)
    inv = inv / inv.mean()
    inv = np.minimum(inv, max_weight)
    return inv  # vector [V]

def epsilon(epoch, s, e, m):
    if epoch <= s: return 0.0
    if epoch >= e: return m
    return m * (epoch - s) / (e - s)

def argmax_sample(probs):
    return np.argmax(probs, axis=-1).astype(np.int32)

# ─────────────────────────────────────────────────────────────────

def main(
    train="data/processed/dataset_3/train.npz",
    val="data/processed/dataset_3/validation.npz",
    test="data/processed/dataset_3/test.npz",
    runs_dir="runs/genre_sched",
    ckpt_dir="models/checkpoints_improved",
    final_path="models/final/best_3.h5",
    vocab_cap=128,
    use_class_weights=True,
    focal_enabled=False,
    focal_gamma=1.5,
    ss_start=1, ss_end=10, ss_max=0.20,
    batch_size=192, epochs=100, patience_es=12, patience_rlr=4,
):
    P.Path(runs_dir).mkdir(parents=True, exist_ok=True)
    P.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    P.Path(P.Path(final_path).parent).mkdir(parents=True, exist_ok=True)

    X_tr, Y_tr, G_tr, vs_tr = load_npz_full(train)
    X_va, Y_va, G_va, vs_va = load_npz_full(val)
    X_te, Y_te, G_te, vs_te = load_npz_full(test)

    T = X_tr.shape[1]
    pos_tr = make_positional(len(X_tr), T)
    pos_va = make_positional(len(X_va), T)
    pos_te = make_positional(len(X_te), T)

    # Estilos (si algún split no lo trae, usa ceros)
    S = (G_tr.shape[1] if G_tr is not None else 6)
    Z_tr = G_tr if G_tr is not None else np.zeros((len(X_tr), S), dtype=np.float32)
    Z_va = G_va if G_va is not None else np.zeros((len(X_va), S), dtype=np.float32)
    Z_te = G_te if G_te is not None else np.zeros((len(X_te), S), dtype=np.float32)

    # vocab size
    vmax = max(int(X_tr.max()), int(Y_tr.max()), int(X_va.max()), int(Y_va.max()), int(X_te.max()), int(Y_te.max()))
    vocab_size = (vs_tr or vmax + 1)

    # Vocab cap → UNK = vocab_cap-1
    if vocab_cap and vocab_cap < vocab_size:
        UNK = vocab_cap - 1
        def cap(arr): a = arr.copy(); a[a >= UNK] = UNK; return a
        X_tr, Y_tr = cap(X_tr), cap(Y_tr)
        X_va, Y_va = cap(X_va), cap(Y_va)
        X_te, Y_te = cap(X_te), cap(Y_te)
        vocab_size = vocab_cap
        print(f"[INFO] vocab_cap={vocab_cap} (UNK={UNK})")

    # Modelo
    model = build_lstm_model_3(vocab_size=vocab_size, seq_len=T, styles=S)

    # Callbacks/logs
    exp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = P.Path(runs_dir) / exp_id
    log_dir.mkdir(parents=True, exist_ok=True)
    cbs = [
        EarlyStopping(patience=patience_es, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=patience_rlr, min_lr=1e-5, monitor="val_loss"),
        TensorBoard(log_dir=str(log_dir)),
        CSVLogger(str(log_dir / "training_log.csv")),
        ModelCheckpoint(filepath=str(P.Path(ckpt_dir) / "best_3.h5"),
                        monitor="val_loss", save_best_only=True)
    ]

    # Class weights vector [V] → expandiremos por token
    cw_vec = build_class_weights(Y_tr, vocab_size) if use_class_weights else None

    # Datasets tf para batching
    tr = tf.data.Dataset.from_tensor_slices(((X_tr, pos_tr, Z_tr), Y_tr)).shuffle(len(X_tr)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    va = tf.data.Dataset.from_tensor_slices(((X_va, pos_va, Z_va), Y_va)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Entrenamiento con Scheduled Sampling (iterativo T veces, T=16)
    optimizer = model.optimizer
    loss_fn = model.loss  # SparseCELS con LS

    @tf.function
    def step_val(x_tokens, x_pos, x_style, y_true):
        y_pred = model([x_tokens, x_pos, x_style], training=False)
        loss = tf.reduce_mean(loss_fn(y_true, y_pred))
        acc  = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred))
        return loss, acc

    def train_epoch(epoch):
        eps = epsilon(epoch, ss_start, ss_end, ss_max)
        tr_loss, tr_acc, n_batches = 0.0, 0.0, 0

        for (x_tokens, x_pos, x_style), y_true in tr:
            # 1) Construir inputs con scheduled sampling (sin gradientes)
            #    T = longitud de secuencia (16)
            x_tokens_ss = tf.identity(x_tokens)  # (B,T)
            T_local = tf.shape(x_tokens_ss)[1]

            # loop por paso temporal, SOLO para ir reemplazando la entrada de ese paso
            # usando el token predicho con prob eps
            t0 = tf.constant(0, dtype=tf.int32)
            cond = lambda t, x_cur: tf.less(t, T_local)

            def body(t, x_cur):
                # forward sin gradiente para obtener pred en paso t
                y_step = model([x_cur, x_pos, x_style], training=False)  # (B,T,V)
                probs_t = y_step[:, t, :]                                 # (B,V)
                pred_t  = tf.argmax(probs_t, axis=-1, output_type=tf.int32)   # (B,)

                # Bernoulli(eps): reemplaza input[t] por predicho con prob eps
                m = tf.cast(tf.random.uniform(tf.shape(pred_t)) < eps, tf.int32)
                xt = tf.cast(x_cur[:, t], tf.int32)
                new_t = m * pred_t + (1 - m) * xt

                idx = tf.stack([
                    tf.range(tf.shape(new_t)[0], dtype=tf.int32),
                    tf.fill([tf.shape(new_t)[0]], t)
                ], axis=1)

                x_next = tf.tensor_scatter_nd_update(
                    x_cur,
                    idx,
                    tf.cast(new_t, x_cur.dtype)
                )
                return t + 1, x_next

            _, x_tokens_ss = tf.while_loop(cond, body, [t0, x_tokens_ss], parallel_iterations=1)

            # 2) Una sola pasada con gradientes para calcular loss y actualizar
            with tf.GradientTape() as tape:
                y_pred = model([x_tokens_ss, x_pos, x_style], training=True)  # (B,T,V)

                if cw_vec is not None:
                    V = tf.shape(y_pred)[-1]
                    y_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=V)     # (B,T,V)
                    p = tf.clip_by_value(y_pred, 1e-7, 1.0)
                    ce = -tf.reduce_sum(y_oh * tf.math.log(p), axis=-1)       # (B,T)
                    w = tf.gather(tf.constant(cw_vec, dtype=tf.float32), y_true)  # (B,T)
                    loss = tf.reduce_mean(ce * w)
                else:
                    loss = tf.reduce_mean(loss_fn(y_true, y_pred))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred))
            tr_loss += float(loss); tr_acc += float(acc); n_batches += 1

        return tr_loss / n_batches, tr_acc / n_batches


    best_val = 1e9
    patience = patience_es
    for epoch in range(1, epochs+1):
        tl, ta = train_epoch(epoch)
        vl, va_acc = 0.0, 0.0
        n = 0
        for (x_tokens, x_pos, x_style), y_true in va:
            l, a = step_val(x_tokens, x_pos, x_style, y_true)
            vl += float(l); va_acc += float(a); n += 1
        vl /= n; va_acc /= n
        print(f"[E{epoch:03d}] train_loss={tl:.4f} train_acc={ta:.4f} | val_loss={vl:.4f} val_acc={va_acc:.4f}")

        # callbacks “manuales” básicos
        # (mantengo ReduceLROnPlateau solo con val_loss)
        if vl < best_val:
            best_val = vl; patience = patience_es
            model.save(P.Path(ckpt_dir) / "best_3.h5")
        else:
            patience -= 1
            if patience == patience_es - patience_rlr:  # activa ReduceLROnPlateau simple
                old = float(tf.keras.backend.get_value(model.optimizer.lr))
                tf.keras.backend.set_value(model.optimizer.lr, max(old * 0.5, 1e-5))
            if patience <= 0:
                print("[INFO] EarlyStopping."); break

    # Carga best y eval test
    best = tf.keras.models.load_model(P.Path(ckpt_dir) / "best_3.h5",
                                      custom_objects={"SparseCELS": SparseCELS})
    pos_te_tf = tf.convert_to_tensor(pos_te, dtype=tf.int32)
    Z_te_tf   = tf.convert_to_tensor(Z_te, dtype=tf.float32)
    test_metrics = best.evaluate([X_te, pos_te_tf, Z_te_tf], Y_te, verbose=1)
    print("[INFO] Test:", dict(zip(best.metrics_names, test_metrics)))

    best.save(final_path)
    with open(P.Path(runs_dir) / exp_id / "hparams.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": int(vocab_size),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "seq_len": int(T),
            "use_class_weights": bool(use_class_weights),
            "vocab_cap": int(vocab_cap),
            "scheduled_sampling": {"start": ss_start, "end": ss_end, "max": ss_max},
        }, f, indent=2)
    print(f"[OK] Guardado: {final_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/dataset_3/train.npz")
    ap.add_argument("--val",   default="data/processed/dataset_3/validation.npz")
    ap.add_argument("--test",  default="data/processed/dataset_3/test.npz")
    ap.add_argument("--runs_dir", default="runs/genre_sched")
    ap.add_argument("--ckpt_dir", default="models/checkpoints_improved")
    ap.add_argument("--final_path", default="models/final/best_3.h5")
    ap.add_argument("--vocab_cap", type=int, default=128)
    ap.add_argument("--use_class_weights", action="store_true")
    ap.add_argument("--focal_enabled", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=1.5)
    ap.add_argument("--ss_start", type=int, default=1)
    ap.add_argument("--ss_end", type=int, default=10)
    ap.add_argument("--ss_max", type=float, default=0.20)
    ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    main(**vars(args))
