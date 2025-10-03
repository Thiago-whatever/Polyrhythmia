# src/train/train_lstm_2.py
import os, json, time, argparse, datetime, pathlib as P
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, ModelCheckpoint

from src.modeling.model_lstm_2 import build_lstm_model_2 as build_lstm_model
from src.metrics.rythm_metrics import perplexity


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)

    # Detectar nombres de claves más comunes
    if "X" in keys and "Y" in keys:
        X, Y = d["X"], d["Y"]
    elif "X_tokens" in keys and "Y_tokens" in keys:
        X, Y = d["X_tokens"], d["Y_tokens"]
    elif "X_ids" in keys and "Y_ids" in keys:
        X, Y = d["X_ids"], d["Y_ids"]
    else:
        raise KeyError(f"No encuentro X/Y en {path}. Claves disponibles: {sorted(keys)}")

    # vocab_size si viene dentro del npz (opcional)
    vocab_size = None
    for k in ["vocab_size", "num_classes"]:
        if k in keys:
            vocab_size = int(d[k])
            break

    return {"X": X, "Y": Y, "vocab_size": vocab_size}


def build_class_weights(y, vocab_size, max_weight=5.0):
    # y: (N, T) ints; frecuencia → peso inverso
    counts = np.bincount(y.ravel(), minlength=vocab_size).astype(np.float64)
    probs = counts / (counts.sum() + 1e-8)
    inv = 1.0 / np.maximum(probs, 1e-8)
    inv = inv / inv.mean()
    inv = np.minimum(inv, max_weight)
    return {i: inv[i] for i in range(vocab_size)}

def make_positional(N, T):
    # matriz (N, T) con 1..T
    pos = np.tile(np.arange(1, T+1, dtype=np.int32), (N, 1))
    return pos

def get_style_vec(N, S, style_idx_list=None):
    # style_idx_list: lista de índices activos (si no, cero → “sin condición”)
    g = np.zeros((N, S), dtype=np.float32)
    if style_idx_list:
        g[:, style_idx_list] = 1.0
    return g

def main(
    train_path="data/processed/train.npz",
    val_path="data/processed/val.npz",
    test_path="data/processed/test.npz",
    runs_dir="runs/improved",
    ckpt_dir="models/checkpoints_improved",
    final_path="models/final/best_2.h5",
    seq_len=16,
    vocab_cap=None,          # e.g., 512; map resto a UNK
    use_class_weights=False,
    batch_size=128,
    epochs=80,
    patience_es=10,
    patience_rlr=4,
    styles=6,
):
    P.Path(runs_dir).mkdir(parents=True, exist_ok=True)
    P.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    P.Path(P.Path(final_path).parent).mkdir(parents=True, exist_ok=True)

    ds_tr = load_npz(train_path)
    ds_va = load_npz(val_path)
    ds_te = load_npz(test_path)

    X_tr, Y_tr = ds_tr["X"], ds_tr["Y"]
    X_va, Y_va = ds_va["X"], ds_va["Y"]
    X_te, Y_te = ds_te["X"], ds_te["Y"]

    # Longitud de secuencia desde los datos
    seq_len = int(X_tr.shape[1])

    # vocab_size: usa el del npz si existe; si no, calcula por máximos observados
    vs = ds_tr["vocab_size"]
    if vs is None:
        vmax = max(int(X_tr.max()), int(Y_tr.max()),
                   int(X_va.max()), int(Y_va.max()),
                   int(X_te.max()), int(Y_te.max()))
        vocab_size = vmax + 1
        print(f"[INFO] vocab_size no venía en NPZ; calculado desde datos: {vocab_size}")
    else:
        vocab_size = int(vs)
        print(f"[INFO] vocab_size leído del NPZ: {vocab_size}")

    # --- cap de vocabulario a top-N ---
    if vocab_cap and vocab_cap < vocab_size:
        UNK = vocab_cap - 1  # último id como UNK
        def cap_tokens(arr):
            arr2 = arr.copy()
            arr2[arr2 >= UNK] = UNK
            return arr2
        X_tr, Y_tr = cap_tokens(X_tr), cap_tokens(Y_tr)
        X_va, Y_va = cap_tokens(X_va), cap_tokens(Y_va)
        X_te, Y_te = cap_tokens(X_te), cap_tokens(Y_te)

        vocab_size = int(vocab_cap)
        print(f"[INFO] Vocab cap aplicado: vocab_size={vocab_size} (UNK={UNK})")
        

    # --- posicional 1..16 ---
    pos_tr = make_positional(len(X_tr), seq_len)
    pos_va = make_positional(len(X_va), seq_len)
    pos_te = make_positional(len(X_te), seq_len)

    # --- estilo (por ahora sin condición => vector cero) ---
    style_tr = get_style_vec(len(X_tr), S=styles)
    style_va = get_style_vec(len(X_va), S=styles)
    style_te = get_style_vec(len(X_te), S=styles)

    # --- modelo ---
    model = build_lstm_model(vocab_size=vocab_size, seq_len=seq_len)

    # --- class weights opcional (ojo: aplica a sparse CE por token; Keras no lo toma
    # directamente para secuencias; se aplica por batch via sample_weight si lo necesitas)
    cw = None
    if use_class_weights:
        cw = build_class_weights(Y_tr, vocab_size)
        print("[INFO] Class weights activados")

    # --- logging / experiment id ---
    exp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = P.Path(runs_dir) / exp_id
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(patience=patience_es, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=patience_rlr, min_lr=1e-5, monitor="val_loss"),
        TensorBoard(log_dir=str(log_dir)),
        CSVLogger(str(log_dir / "training_log.csv")),
        ModelCheckpoint(filepath=str(P.Path(ckpt_dir) / "best_2.h5"),
                        monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        x={"tokens": X_tr, "pos": pos_tr, "style": style_tr},
        y=Y_tr,
        validation_data=({"tokens": X_va, "pos": pos_va, "style": style_va}, Y_va),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # eval
    best = tf.keras.models.load_model(
        P.Path(ckpt_dir) / "best_2.h5",
        custom_objects={"perplexity": perplexity}
    )
    test_metrics = best.evaluate(
        {"tokens": X_te, "pos": pos_te, "style": style_te}, Y_te, verbose=1
    )
    print("[INFO] Test metrics:", dict(zip(best.metrics_names, test_metrics)))

    # guardar final + config usada
    best.save(final_path)
    with open(log_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "epochs": epochs,
            "seq_len": seq_len,
            "use_class_weights": use_class_weights,
            "vocab_cap": vocab_cap,
        }, f, indent=2)
    print(f"[OK] Modelo mejorado guardado en: {final_path}\n[RUN] Logs: {log_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="data/processed/dataset_2/train.npz")
    parser.add_argument("--val_path",   default="data/processed/dataset_2/validation.npz")
    parser.add_argument("--test_path",  default="data/processed/dataset_2/test.npz")
    parser.add_argument("--runs_dir", default="runs/improved")
    parser.add_argument("--ckpt_dir", default="models/checkpoints_improved")
    parser.add_argument("--final_path", default="models/final/best_2.h5")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--vocab_cap", type=int, default=512)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience_es", type=int, default=10)
    parser.add_argument("--patience_rlr", type=int, default=4)
    parser.add_argument("--styles", type=int, default=6)
    args = parser.parse_args()
    main(**vars(args))
