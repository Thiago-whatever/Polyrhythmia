# src/utils/build_vocab_topN.py
import argparse, json, numpy as np

def main(in_npz: str, out_vocab: str, topN: int):
    data = np.load(in_npz)
    # Detecta claves típicas
    if "X" in data:
        X = data["X"]
    elif "X_tokens" in data:
        X = data["X_tokens"]
    elif "X_ids" in data:
        X = data["X_ids"]
    else:
        raise KeyError(f"No se encontró X en {in_npz}; claves={data.files}")

    # Aplana y cuenta ocurrencias
    flat = X.ravel()
    uniq, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)  # top por frecuencia
    top_ids = uniq[order[:topN]]

    # Mapea bitmask→id topN y el resto a UNK
    bitmask2id = {}
    id2bitmask = {}
    unk_id = topN - 1

    for new_id, bitmask in enumerate(top_ids):
        bitmask2id[int(bitmask)] = int(new_id)
        id2bitmask[int(new_id)] = int(bitmask)

    # resto → UNK
    vocab = {
        "bitmask2id": bitmask2id,
        "id2bitmask": id2bitmask,
        "unk_id": unk_id
    }

    with open(out_vocab, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    print(f"[OK] Vocab Top-{topN} guardado en {out_vocab}")
    print(f"[INFO] {len(bitmask2id)} tokens mapeados; unk_id={unk_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_npz", required=True)
    parser.add_argument("--out_vocab", required=True)
    parser.add_argument("--topN", type=int, required=True)
    args = parser.parse_args()
    main(args.in_npz, args.out_vocab, args.topN)
