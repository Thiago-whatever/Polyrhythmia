from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT = Path(".")
ANALYSIS = ROOT / "analysis"
ANALYSIS.mkdir(exist_ok=True)

# ========= CONFIG =========
# Modelo 2 (improved)
CSV_M2 = ROOT / "runs" / "improved" / "20251005_221709" / "training_log.csv"

# Modelo 3 
M3_RUN_DIR = ROOT / "runs" / "genre_sched_relogged" / "20260301_235908"
CSV_M3 = M3_RUN_DIR / "training_log.csv"

# Modelo 1 (baseline) TensorBoard
TB_TRAIN_DIR = ROOT / "logs" / "tensorboard" / "train"
TB_VAL_DIR   = ROOT / "logs" / "tensorboard" / "validation"

# ========= HELPERS =========
def newest_event_file(folder: Path) -> Path:
    files = sorted(folder.glob("events.out.tfevents.*"))
    if not files:
        files = sorted(folder.glob("*tfevents*"))
    if not files:
        raise RuntimeError(f"No encontré tfevents en {folder}")
    return max(files, key=lambda p: p.stat().st_mtime)

import numpy as np

def _scalar_from_value(v):
    # Caso 1: TF1-style scalar
    if hasattr(v, "simple_value") and v.simple_value is not None and v.simple_value != 0.0:
        return float(v.simple_value)

    # Caso 2: TF2-style scalar stored as tensor
    if hasattr(v, "tensor") and v.tensor is not None:
        try:
            arr = tf.make_ndarray(v.tensor)
            # arr puede ser 0-d (scalar) o 1-d de longitud 1
            return float(np.array(arr).reshape(-1)[0])
        except Exception:
            pass

    return 0.0

def read_scalar_series(event_path: Path, tag: str):
    series = []
    for e in tf.compat.v1.train.summary_iterator(str(event_path)):
        if e.summary is None:
            continue
        for v in e.summary.value:
            if v.tag == tag:
                series.append((e.step, _scalar_from_value(v)))
    series.sort(key=lambda x: x[0])
    return [val for _, val in series]

def read_csv_curves(csv_path: Path):
    df = pd.read_csv(csv_path)
    epochs = (df["epoch"] + 1).tolist()

    train_loss = df["loss"].tolist()
    val_loss   = df["val_loss"].tolist()

    acc_col = "sparse_categorical_accuracy" if "sparse_categorical_accuracy" in df.columns else "accuracy"
    val_acc_col = "val_sparse_categorical_accuracy" if "val_sparse_categorical_accuracy" in df.columns else "val_accuracy"

    train_acc = df[acc_col].tolist()
    val_acc   = df[val_acc_col].tolist()

    return epochs, train_loss, val_loss, train_acc, val_acc

# ========= LOAD M2 + M3 (CSV) =========
e2, m2_tr_loss, m2_va_loss, m2_tr_acc, m2_va_acc = read_csv_curves(CSV_M2)
e3, m3_tr_loss, m3_va_loss, m3_tr_acc, m3_va_acc = read_csv_curves(CSV_M3)

# ========= LOAD M1 (TensorBoard) =========
train_event = newest_event_file(TB_TRAIN_DIR)
val_event   = newest_event_file(TB_VAL_DIR)

# Tags verificados por tu debug
TAG_LOSS = "epoch_loss"
TAG_ACC  = "epoch_sparse_categorical_accuracy"

m1_tr_loss = read_scalar_series(train_event, TAG_LOSS)
m1_tr_acc  = read_scalar_series(train_event, TAG_ACC)
m1_va_loss = read_scalar_series(val_event, TAG_LOSS)
m1_va_acc  = read_scalar_series(val_event, TAG_ACC)

print("M1 loss head:", m1_tr_loss[:5], m1_va_loss[:5])
print("M1 acc  head:", m1_tr_acc[:5],  m1_va_acc[:5])

# Eje de épocas para M1
max_len_m1 = max(len(m1_tr_loss), len(m1_va_loss), len(m1_tr_acc), len(m1_va_acc))
e1 = list(range(1, max_len_m1 + 1))
m1_tr_loss = m1_tr_loss[:max_len_m1]
m1_va_loss = m1_va_loss[:max_len_m1]
m1_tr_acc  = m1_tr_acc[:max_len_m1]
m1_va_acc  = m1_va_acc[:max_len_m1]

# ========= PLOT (paper + B/N friendly) =========
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=False)

# Estilos: diferenciación por patrón + marcador (sirve B/N)
sty1 = dict(linestyle="-",  marker="o", markersize=3, markevery=max(1, len(e1)//12), linewidth=1.2)
sty2 = dict(linestyle="--", marker="s", markersize=3, markevery=max(1, len(e2)//12), linewidth=1.2)
sty3 = dict(linestyle="-.", marker="^", markersize=3, markevery=max(1, len(e3)//12), linewidth=1.2)

# --- LOSS ---
ax1.plot(e1, m1_tr_loss, label="M1 Train", **sty1)
ax1.plot(e1, m1_va_loss, label="M1 Val",   linestyle="-", linewidth=1.8)

ax1.plot(e2, m2_tr_loss, label="M2 Train", **sty2)
ax1.plot(e2, m2_va_loss, label="M2 Val",   linestyle="--", linewidth=1.8)

ax1.plot(e3, m3_tr_loss, label="M3 Train", **sty3)
ax1.plot(e3, m3_va_loss, label="M3 Val",   linestyle="-.", linewidth=1.8)

ax1.set_ylabel("Loss")
ax1.set_title("Curvas de aprendizaje – Comparación de modelos")
ax1.grid(True, linewidth=0.5, alpha=0.6)

# --- ACCURACY ---
ax2.plot(e1, m1_tr_acc, label="M1 Train", **sty1)
ax2.plot(e1, m1_va_acc, label="M1 Val",   linestyle="-", linewidth=1.8)

ax2.plot(e2, m2_tr_acc, label="M2 Train", **sty2)
ax2.plot(e2, m2_va_acc, label="M2 Val",   linestyle="--", linewidth=1.8)

ax2.plot(e3, m3_tr_acc, label="M3 Train", **sty3)
ax2.plot(e3, m3_va_acc, label="M3 Val",   linestyle="-.", linewidth=1.8)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.grid(True, linewidth=0.5, alpha=0.6)

# Leyenda única compacta
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
seen = set()
handles, labels = [], []
for h, l in list(zip(handles1, labels1)) + list(zip(handles2, labels2)):
    if l not in seen:
        handles.append(h); labels.append(l); seen.add(l)

fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.05, 1, 1])

png_out = ANALYSIS / "learning_curves_paper.png"
pdf_out = ANALYSIS / "learning_curves_paper.pdf"
fig.savefig(png_out, dpi=300)
fig.savefig(pdf_out)
plt.show()

print("[OK] Guardado:")
print(" -", png_out)
print(" -", pdf_out)
print("\n[INFO] M1 tfevents:")
print(" Train:", train_event)
print(" Val  :", val_event)
print("[INFO] M3 CSV:", CSV_M3)