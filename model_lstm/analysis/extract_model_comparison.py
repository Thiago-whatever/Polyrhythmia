# analysis/extract_model_comparison.py
import math
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# ---- Custom objects (según tus scripts) ----
from src.modeling.model_lstm import perplexity as perplexity_m1
from src.metrics.rythm_metrics import perplexity as perplexity_m2m3
from src.modeling.model_lstm_2 import SparseCELS

ROOT = Path(".")
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

def load_npz_model1(npz_path: Path):
    d = np.load(npz_path, allow_pickle=False)
    Xt = d["X_tokens"].astype("int32")
    Yt = d["Y_tokens"].astype("int32")
    Xs = d["X_style"].astype("float32")
    return (Xt, Xs), Yt

def load_npz_model2(npz_path: Path, styles=6, vocab_cap=512):
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    if "X" in keys and "Y" in keys:
        X, Y = d["X"], d["Y"]
    elif "X_tokens" in keys and "Y_tokens" in keys:
        X, Y = d["X_tokens"], d["Y_tokens"]
    elif "X_ids" in keys and "Y_ids" in keys:
        X, Y = d["X_ids"], d["Y_ids"]
    else:
        raise KeyError(f"No encuentro X/Y en {npz_path}. Claves: {sorted(keys)}")

    X = X.astype("int32")
    Y = Y.astype("int32")

    # ✅ replicate vocab cap del train_lstm_2.py
    if vocab_cap:
        UNK = vocab_cap - 1
        X = X.copy()
        Y = Y.copy()
        X[X >= UNK] = UNK
        Y[Y >= UNK] = UNK

    N, T = X.shape
    pos = np.tile(np.arange(1, T + 1, dtype=np.int32), (N, 1))
    style = np.zeros((N, styles), dtype=np.float32)

    return {"tokens": X, "pos": pos, "style": style}, Y

def load_npz_model3(npz_path: Path, styles=6, vocab_cap=128):
    d = np.load(npz_path, allow_pickle=True)
    if "X_tokens" in d and "Y_tokens" in d:
        X, Y = d["X_tokens"], d["Y_tokens"]
    elif "X" in d and "Y" in d:
        X, Y = d["X"], d["Y"]
    else:
        raise KeyError(f"{npz_path} sin X/Y esperados")

    X = X.astype("int32")
    Y = Y.astype("int32")

    # ✅ replicate vocab cap del train_lstm_3.py
    if vocab_cap:
        UNK = vocab_cap - 1
        X = X.copy()
        Y = Y.copy()
        X[X >= UNK] = UNK
        Y[Y >= UNK] = UNK

    N, T = X.shape
    pos = np.tile(np.arange(1, T + 1, dtype=np.int32), (N, 1))

    if "X_style" in d.files:
        Z = d["X_style"].astype("float32")
        S = Z.shape[1]
    else:
        S = styles
        Z = np.zeros((N, S), dtype=np.float32)

    return [X, pos, Z], Y

def safe_perplexity_from_loss(loss_val: float) -> float:
    # Perplexity = exp(cross-entropy)
    return float(math.exp(loss_val))

def evaluate_model(model, x, y, batch_size=256):
    out = model.evaluate(x, y, verbose=0, batch_size=batch_size)
    metrics = dict(zip(model.metrics_names, out))
    return metrics

def pick_ppl(metrics: dict):
    # Prioriza si existe "perplexity" en metrics; si no, usa exp(loss)
    if "perplexity" in metrics:
        return float(metrics["perplexity"])
    if "loss" in metrics:
        return safe_perplexity_from_loss(float(metrics["loss"]))
    return None

def main():
    results = []

    # ---------------- Model 1 ----------------
    m1_path = MODELS / "final" / "best.h5"  # tú confirmaste que existe
    m1 = tf.keras.models.load_model(
        m1_path,
        custom_objects={"perplexity": perplexity_m1},
        compile=True
    )

    x_tr, y_tr = load_npz_model1(DATA / "train.npz")
    x_va, y_va = load_npz_model1(DATA / "validation.npz")

    tr_m = evaluate_model(m1, x_tr, y_tr)
    va_m = evaluate_model(m1, x_va, y_va)

    results.append({
        "modelo": "LSTM-1 (baseline)",
        "train_loss": tr_m.get("loss"),
        "val_loss": va_m.get("loss"),
        "train_acc": tr_m.get("sparse_categorical_accuracy"),
        "val_acc": va_m.get("sparse_categorical_accuracy"),
        "val_perplexity": pick_ppl(va_m),
        "params": int(m1.count_params()),
        "early_stopping": "val_loss, patience=7 (epoch exacto: N/A, logs no persistidos)",
    })

    # ---------------- Model 2 ----------------
    m2_path = MODELS / "final" / "best_2.h5"
    m2 = tf.keras.models.load_model(
        m2_path,
        custom_objects={"perplexity": perplexity_m2m3, "SparseCELS": SparseCELS},
        compile=True
    )

    x_tr2, y_tr2 = load_npz_model2(DATA / "dataset_2" / "train.npz", vocab_cap=512)
    x_va2, y_va2 = load_npz_model2(DATA / "dataset_2" / "validation.npz", vocab_cap=512)

    tr2_m = evaluate_model(m2, x_tr2, y_tr2)
    va2_m = evaluate_model(m2, x_va2, y_va2)

    results.append({
        "modelo": "LSTM-2 (improved)",
        "train_loss": tr2_m.get("loss"),
        "val_loss": va2_m.get("loss"),
        "train_acc": tr2_m.get("sparse_categorical_accuracy"),
        "val_acc": va2_m.get("sparse_categorical_accuracy"),
        "val_perplexity": pick_ppl(va2_m),
        "params": int(m2.count_params()),
        "early_stopping": "val_loss, patience=12 (default) (epoch exacto: N/A, logs no persistidos)",
    })

    # ---------------- Model 3 ----------------
    m3_path = MODELS / "final" / "best_3.h5"
    m3 = tf.keras.models.load_model(
        m3_path,
        custom_objects={"SparseCELS": SparseCELS, "perplexity": perplexity_m2m3},
        compile=True
    )

    x_tr3, y_tr3 = load_npz_model3(DATA / "dataset_3" / "train.npz", vocab_cap=128)
    x_va3, y_va3 = load_npz_model3(DATA / "dataset_3" / "validation.npz", vocab_cap=128)

    tr3_m = evaluate_model(m3, x_tr3, y_tr3)
    va3_m = evaluate_model(m3, x_va3, y_va3)

    results.append({
        "modelo": "LSTM-3 (genre_sched)",
        "train_loss": tr3_m.get("loss"),
        "val_loss": va3_m.get("loss"),
        "train_acc": tr3_m.get("sparse_categorical_accuracy"),
        "val_acc": va3_m.get("sparse_categorical_accuracy"),
        "val_perplexity": pick_ppl(va3_m),
        "params": int(m3.count_params()),
        "early_stopping": "manual val_loss, patience=12 (default) (epoch exacto: N/A, logs no persistidos)",
    })

    # pretty print + guardar json
    out_path = ROOT / "analysis" / "model_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Imprime tabla markdown para pegar
    headers = ["modelo", "loss train/val", "acc train/val", "perplexity (val)", "params", "early stopping"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in results:
        loss_tv = f"{r['train_loss']:.4f} / {r['val_loss']:.4f}"
        acc_tv  = f"{r['train_acc']:.4f} / {r['val_acc']:.4f}"
        ppl     = f"{r['val_perplexity']:.2f}" if r["val_perplexity"] is not None else "N/A"
        print("| " + " | ".join([
            r["modelo"],
            loss_tv,
            acc_tv,
            ppl,
            f"{r['params']:,}",
            r["early_stopping"]
        ]) + " |")

    print(f"\n[OK] Guardado: {out_path}")

if __name__ == "__main__":
    main()