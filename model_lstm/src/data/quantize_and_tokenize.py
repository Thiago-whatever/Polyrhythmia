# src/data/quantize_and_tokenize.py
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pretty_midi as pm

# ---------- 9-PIECE MAP (similar a Magenta DrumRNN) ----------
# Orden de clases: [KICK, SNARE, HH_CLOSED, HH_OPEN, TOM_LOW, TOM_MID, TOM_HIGH, CRASH, RIDE]
CLASS_NAMES = ["KICK","SNARE","HH_CLOSED","HH_OPEN","TOM_LOW","TOM_MID","TOM_HIGH","CRASH","RIDE"]

PITCH_CLASSSETS = {
    "KICK":       {35,36},
    "SNARE":      {38,40,37,39},               # snare, electric snare, side stick, hand clap -> snare
    "HH_CLOSED":  {42,44},                     # closed + pedal
    "HH_OPEN":    {46},
    "TOM_LOW":    {41,43},                     # low floor / hi floor
    "TOM_MID":    {45,47},                     # low tom / low-mid
    "TOM_HIGH":   {48,50},                     # hi-mid / high
    "CRASH":      {49,57,55},                  # crash1/2 + splash
    "RIDE":       {51,59,53},                  # ride1/2 + ride bell
}

# Construye un dict pitch->idx_clase (0..8)
PITCH2CLASS = {}
for idx, name in enumerate(CLASS_NAMES):
    for p in PITCH_CLASSSETS[name]:
        PITCH2CLASS[p] = idx

def drums_of(pm_obj: pm.PrettyMIDI):
    """Devuelve lista de instrumentos marcados como is_drum."""
    return [inst for inst in pm_obj.instruments if inst.is_drum]

def bitmask_from_pitches(pitches):
    """pitches: iterable de ints (clases 0..8). Devuelve string binaria de 9 bits."""
    bits = [0]*9
    for cls in pitches:
        bits[cls] = 1
    return "".join(str(b) for b in bits)

def quantize_bar(pm_obj: pm.PrettyMIDI, bar_start: float, bar_end: float, steps_per_bar: int = 16):
    """Cuantiza un compás [bar_start, bar_end) en 'steps_per_bar' pasos.
       Devuelve lista de 16 sets con clases activas en cada subdivisión."""
    step_edges = np.linspace(bar_start, bar_end, steps_per_bar+1)
    step_sets = [set() for _ in range(steps_per_bar)]

    for inst in drums_of(pm_obj):
        for note in inst.notes:
            # Ignora notas fuera del compás
            if note.start < bar_start or note.start >= bar_end:
                continue
            # Mapea pitch -> clase (si no está mapeado, se ignora)
            if note.pitch not in PITCH2CLASS:
                continue
            # Encuentra subdivisión más cercana
            idx = np.argmin(np.abs(step_edges[:-1] - note.start))
            step_sets[idx].add(PITCH2CLASS[note.pitch])

    return step_sets

def extract_bars(pm_obj: pm.PrettyMIDI, steps_per_bar=16, max_bars=None):
    """
    Segmenta por downbeats (inicio de compás) y cuantiza cada compás a 16 pasos.
    Devuelve lista de tokens (cada compás = lista de 16 tokens bitmask-string).
    """
    downbeats = pm_obj.get_downbeats()
    if len(downbeats) < 2:
        return []

    bars_tokens = []

    for i in range(len(downbeats)-1):
        b_start = float(downbeats[i])
        b_end   = float(downbeats[i+1])

        # cuantiza compás
        step_sets = quantize_bar(pm_obj, b_start, b_end, steps_per_bar)
        # convierte sets a bitmask-string
        tokens = [bitmask_from_pitches(s) if len(s)>0 else "000000000" for s in step_sets]
        bars_tokens.append(tokens)

        if max_bars is not None and len(bars_tokens) >= max_bars:
            break
    return bars_tokens

def build_vocab(observed_bitmasks):
    """Crea vocab: bitmask string -> id (0 reservado a silencio '000000000')."""
    unique = sorted(set(observed_bitmasks))
    # aseguramos que el silencio sea 0
    vocab = {"000000000": 0}
    next_id = 1
    for bm in unique:
        if bm == "000000000":
            continue
        vocab[bm] = next_id
        next_id += 1
    return vocab

def encode_tokens(tokens_16, vocab):
    """tokens_16: lista de 16 bitmasks. Devuelve np.array shape (16,) con ids."""
    return np.array([vocab[bm] for bm in tokens_16], dtype=np.int32)

def make_XY_from_bars(bars_encoded):
    """
    bars_encoded: lista de np.arrays shape (16,), enteros ids.
    X = secuencia 16; Y = next-token por paso (shiftado).
    """
    X = []
    Y = []
    for seq in bars_encoded:
        X.append(seq.copy())
        # objetivo autoregresivo: predicción del siguiente evento
        # aquí creamos Y como la misma secuencia desplazada a la izquierda,
        # y el último target será el propio último token
        Y.append(np.concatenate([seq[1:], seq[-1:]]))
    return np.stack(X), np.stack(Y)

def load_split_from_info(info_csv: Path, split: str):
    """Lee info.csv y devuelve lista de paths relativos 'midi/....mid' del split."""
    df = pd.read_csv(info_csv)
    # Columnas comunes: 'midi_filename', 'split' o 'subset' (train/validation/test)
    split_col = 'split' if 'split' in df.columns else 'subset'
    rows = df[df[split_col] == split]
    return rows['midi_filename'].tolist()

def main(groove_dir: Path, out_dir: Path, steps_per_bar=16, save_vocab=True):
    info_csv = groove_dir / "info.csv"
    midi_root = groove_dir / "midi"
    assert info_csv.exists(), f"Falta info.csv en {groove_dir}"
    assert midi_root.exists(), f"Falta carpeta midi/ en {groove_dir}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Recolecta bitmasks para construir vocab
    observed = []

    # 2) Procesa por split
    splits = ["train", "validation", "test"]
    for split in splits:
        print(f"[INFO] Procesando split = {split}")
        file_list = load_split_from_info(info_csv, split)
        all_bars_bitmasks = []

        for rel_path in file_list:
            midi_path = midi_root / rel_path
            if not midi_path.exists():
                # algunos paths en info.csv empiezan sin 'midi/'; esto asegura robustez
                midi_path = groove_dir / rel_path
            if not midi_path.exists():
                print(f"[WARN] No existe: {rel_path}")
                continue

            try:
                pm_obj = pm.PrettyMIDI(str(midi_path))
            except Exception as e:
                print(f"[WARN] Error leyendo {rel_path}: {e}")
                continue

            bars_tokens = extract_bars(pm_obj, steps_per_bar=steps_per_bar)
            for tokens_16 in bars_tokens:
                all_bars_bitmasks.append(tokens_16)
                observed.extend(tokens_16)

        # cache por split (bitmasks crudos, por si quieres inspeccionar)
        np.savez_compressed(out_dir / f"{split}_bitmasks.npz",
                            bars=np.array(all_bars_bitmasks, dtype=object))

    # 3) Construye vocabulario a partir de TODOS los bitmasks observados
    vocab = build_vocab(observed)
    print(f"[OK] Vocab construido. Tamaño: {len(vocab)} (incluye silencio=0)")
    if save_vocab:
        with open(out_dir / "vocab.json", "w") as f:
            json.dump(vocab, f, indent=2)

    # 4) Re-encodea y guarda X/Y por split
    for split in ["train", "validation", "test"]:
        pack = np.load(out_dir / f"{split}_bitmasks.npz", allow_pickle=True)
        bars_bitmasks = pack["bars"]
        bars_encoded = [encode_tokens(tokens_16, vocab) for tokens_16 in bars_bitmasks]
        X, Y = make_XY_from_bars(bars_encoded)

        # Vector de estilos g: si no hay mapeo en info.csv, guarda dummy ceros
        # shape (N, 6) -> Jazz, Choro, Bossa, Samba, HipHop, AfroCuban
        G = np.zeros((X.shape[0], 6), dtype=np.float32)

        np.savez_compressed(out_dir / f"{split}.npz",
                            X_tokens=X, Y_tokens=Y, X_style=G)

        print(f"[OK] Guardado {split}.npz  X:{X.shape}  Y:{Y.shape}  G:{G.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groove_dir", default="data/raw/groove")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--steps_per_bar", type=int, default=16)
    args = ap.parse_args()

    main(Path(args.groove_dir), Path(args.out_dir), steps_per_bar=args.steps_per_bar)
