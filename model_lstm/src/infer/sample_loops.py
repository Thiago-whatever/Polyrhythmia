# src/infer/sample_loops.py
import argparse, json, pathlib as P, numpy as np, tensorflow as tf
import json

def load_vocab(vocab_path):
    """Soporta dos formatos:
       (A) nuevo: {"bitmask2id": {...}, "id2bitmask": {...}, "unk_id": N}
       (B) legado: {bitmask: id}
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    if isinstance(vocab, dict) and "bitmask2id" in vocab:
        # Formato nuevo
        b2i_raw = vocab["bitmask2id"]
        i2b_raw = vocab.get("id2bitmask", {})
        unk_id  = int(vocab.get("unk_id", max(int(k) for k in i2b_raw.keys())))
        bitmask2id = {int(k): int(v) for k, v in b2i_raw.items()}
        # Asegura id2bitmask completo; si no viene, invierte b2i
        if i2b_raw:
            id2bitmask = {int(k): int(v) for k, v in i2b_raw.items()}
        else:
            id2bitmask = {int(v): int(k) for k, v in bitmask2id.items()}
        return bitmask2id, id2bitmask, unk_id

    else:
        # Formato legado: diccionario plano {bitmask: id}
        bitmask2id = {int(k): int(v) for k, v in vocab.items()}
        id2bitmask = {int(v): int(k) for k, v in bitmask2id.items()}
        unk_id = max(id2bitmask.keys())
        return bitmask2id, id2bitmask, unk_id


def top_k_sample(probs, k=10, temperature=0.9):
    # estabiliza
    logits = np.log(np.clip(probs, 1e-9, 1.0))
    logits = logits / max(temperature, 1e-6)
    # top-k
    k = int(min(k, len(logits)))
    top_idx = np.argpartition(-logits, k-1)[:k]
    sub = logits[top_idx]
    sub = np.exp(sub - sub.max())
    sub = sub / sub.sum()
    return int(np.random.choice(top_idx, p=sub))

def generate_bars(model, num_bars=16, seq_len=16, styles=6,
                  temperature=0.9, top_k=10, start_token=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    bars = []
    style_vec = np.zeros((1, styles), dtype=np.float32)  # sin condición por ahora
    for _ in range(num_bars):
        tokens = np.zeros((1, seq_len), dtype=np.int32)
        tokens[0, 0] = start_token
        pos = np.tile(np.arange(1, seq_len+1, dtype=np.int32), (1,1))
        # auto-regresivo
        for t in range(seq_len):
            out = model.predict({"tokens": tokens, "pos": pos, "style": style_vec}, verbose=0)
            probs_t = out[0, t]  # (V,)
            nxt = top_k_sample(probs_t, k=top_k, temperature=temperature)
            tokens[0, t] = nxt
        bars.append(tokens[0].tolist())
    return bars

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Ruta al .h5 del modelo")
    ap.add_argument("--vocab_path", required=True, help="Ruta a vocab_topN.json (bitmask->id)")
    ap.add_argument("--num_bars", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--styles", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--start_token", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_json", default="samples_bars.jsonl", help="Salida JSONL (1 línea = lista de 16 ids)")
    ap.add_argument("--styles_on", default="", help="CSV de estilos activos por nombre o idx, e.g. jazz,afrocubano")
    args = ap.parse_args()

    style_vec = np.zeros((1, args.styles), dtype=np.float32)
    if args.styles_on:
        names = [s.strip() for s in args.styles_on.split(",") if s.strip()]
        # si pasas nombres, mapea a índices; o permite pasar "0,3"
        for n in names:
            try:
                idx = int(n)
            except:
                # mapea por nombre si tienes lista ordenada en la app
                name2idx = {"jazz":0,"bossa":1,"samba":2,"hiphop":3,"afrocubano":4,"choro":5}
                idx = name2idx.get(n.lower(), None)
            if idx is not None and 0 <= idx < args.styles: style_vec[0, idx] = 1.0

    # Cargar modelo (sin recompilar)
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # Vocab por si quieres post-procesar o exportar luego a MIDI
    bitmask2id, id2bitmask, unk_id = load_vocab(args.vocab_path)
    vocab_size = max(id2bitmask.keys()) + 1

    # Generar
    bars = generate_bars(
        model=model,
        num_bars=args.num_bars,
        seq_len=args.seq_len,
        styles=args.styles,
        temperature=args.temperature,
        top_k=args.top_k,
        start_token=args.start_token,
        seed=args.seed
    )

    # Guardar JSONL
    out_path = P.Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for bar in bars:
            f.write(json.dumps({"tokens": bar}) + "\n")

    print(f"[OK] Generados {len(bars)} compases → {out_path}")
    # Nota: el export a MIDI se hace con export_to_midi.py (id → bitmask → notas)

if __name__ == "__main__":
    main()
