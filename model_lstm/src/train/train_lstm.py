# src/train/train_lstm.py
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks as KCB
from tensorflow.keras.optimizers import Adam

from modeling.model_lstm import build_lstm_model, perplexity

def set_seeds(seed=42):
    import os, random
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_npz(npz_path: Path):
    pack = np.load(npz_path, allow_pickle=False)
    X_tokens = pack["X_tokens"].astype("int32")  # (N, 16)
    Y_tokens = pack["Y_tokens"].astype("int32")  # (N, 16) shifted targets
    X_style  = pack["X_style"].astype("float32") # (N, 6)
    return X_tokens, X_style, Y_tokens

def make_ds(Xt, Xs, Yt, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(((Xt, Xs), Yt))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(Xt), 10000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main(
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    vocab_json: Path,
    steps_per_bar: int = 16,
    num_styles: int = 6,
    embedding_dim: int = 64,
    lstm_units=(128, 128),
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 50,
    ckpt_dir: Path = Path("models/checkpoints"),
    final_path: Path = Path("models/final/best.keras"),
):
    set_seeds(42)

    # 1) Carga datos
    Xtr, Gtr, Ytr = load_npz(train_npz)
    Xva, Gva, Yva = load_npz(val_npz)
    Xte, Gte, Yte = load_npz(test_npz)

    # 2) Carga vocab_size
    with open(vocab_json, "r") as f:
        vocab = json.load(f)
    vocab_size = int(max(vocab.values())) + 1  # id más alto + 1

    print(f"[INFO] Shapes -> Train X:{Xtr.shape}  G:{Gtr.shape}  Y:{Ytr.shape}")
    print(f"[INFO] Vocab size: {vocab_size}")

    # 3) Datasets tf.data
    ds_train = make_ds(Xtr, Gtr, Ytr, batch_size=batch_size, shuffle=True)
    ds_val   = make_ds(Xva, Gva, Yva, batch_size=batch_size, shuffle=False)
    ds_test  = make_ds(Xte, Gte, Yte, batch_size=batch_size, shuffle=False)

    # 4) Modelo
    model = build_lstm_model(
        vocab_size=vocab_size,
        steps_per_bar=steps_per_bar,
        num_styles=num_styles,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        inject_mode="concat",
    )
    model.summary()

    # 5) Compilar
    opt = Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",           # CE categórica (targets enteros)
        metrics=["sparse_categorical_accuracy", perplexity],
    )

    # 6) Callbacks
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_cb = KCB.ModelCheckpoint(
        filepath=str(ckpt_dir / "best.keras"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    )
    es_cb = KCB.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
    )
    tb_cb = KCB.TensorBoard(log_dir="logs/tensorboard", update_freq="epoch")

    # 7) Entrenar
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=[ckpt_cb, es_cb, tb_cb],
        verbose=1,
    )

    # 8) Evaluar en test
    print("\n[INFO] Evaluación en TEST:")
    eval_out = model.evaluate(ds_test, verbose=1)
    print(dict(zip(model.metrics_names, eval_out)))

    # 9) Guardar mejor modelo en models/final/
    # Copiamos/borramos y guardamos final
    best_model = tf.keras.models.load_model(ckpt_dir / "best.keras", custom_objects={"perplexity": perplexity})
    best_model.save(final_path)
    print(f"[OK] Modelo final guardado en: {final_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", default="data/processed/train.npz")
    ap.add_argument("--val_npz",   default="data/processed/validation.npz")
    ap.add_argument("--test_npz",  default="data/processed/test.npz")
    ap.add_argument("--vocab_json", default="data/processed/vocab.json")
    ap.add_argument("--steps_per_bar", type=int, default=16)
    ap.add_argument("--num_styles", type=int, default=6)
    ap.add_argument("--embedding_dim", type=int, default=64)
    ap.add_argument("--lstm_units", nargs="+", type=int, default=[128, 128])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--recurrent_dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ckpt_dir", default="models/checkpoints")
    ap.add_argument("--final_path", default="models/final/best.keras")
    args = ap.parse_args()

    main(
        train_npz=Path(args.train_npz),
        val_npz=Path(args.val_npz),
        test_npz=Path(args.test_npz),
        vocab_json=Path(args.vocab_json),
        steps_per_bar=args.steps_per_bar,
        num_styles=args.num_styles,
        embedding_dim=args.embedding_dim,
        lstm_units=tuple(args.lstm_units),
        dropout=args.dropout,
        recurrent_dropout=args.recurrent_dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        ckpt_dir=Path(args.ckpt_dir),
        final_path=Path(args.final_path),
    )
