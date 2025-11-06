# src/utils/export_to_midi.py
import argparse, json, pathlib as P, numpy as np, pretty_midi as pm

# Orden de clases: [KICK, SNARE, HH_CLOSED, HH_OPEN, TOM_LOW, TOM_MID, TOM_HIGH, CRASH, RIDE]
CLASS_PITCHES = [36, 38, 42, 46, 41, 45, 48, 49, 51]  # un pitch representativo por clase
SILENCE_MASK = "000000000"

# src/utils/export_to_midi.py (reemplaza load_vocab)

def load_vocab(vocab_path):
    import json
    with open(vocab_path, "r", encoding="utf-8") as f:
        voc = json.load(f)

    # Formato nuevo (recomendado):
    # {"bitmask2id": {"000000000":0, ...}, "id2bitmask": {"0":"000000000", ...}, "unk_id": 127}
    if isinstance(voc, dict) and "bitmask2id" in voc:
        id2b_raw = voc.get("id2bitmask", {})
        # clave id → int; valor bitmask → str
        id2bitmask = {int(k): str(v) for k, v in id2b_raw.items()}
        unk_id = voc.get("unk_id", None)
        return None, id2bitmask, unk_id

    # Formato legado: {bitmask_str: id_int}
    bitmask2id = {str(k): int(v) for k, v in voc.items()}
    id2bitmask = {v: k for k, v in bitmask2id.items()}
    unk_id = bitmask2id.get("UNK", None)  # puede no existir
    return bitmask2id, id2bitmask, unk_id


def bar_ids_to_bitmasks(bar_ids, id2bitmask, unk_id=None):
    SILENCE_MASK = "000000000"
    bitmasks = []
    for tid in bar_ids:
        if tid in id2bitmask:
            bm = id2bitmask[tid]
        elif (unk_id is not None) and (tid == unk_id):
            bm = SILENCE_MASK
        else:
            bm = SILENCE_MASK

        # Normaliza: si viene int → binario de 9 bits; si viene str → asegura 9 chars
        if isinstance(bm, int):
            bm = format(bm, "09b")
        else:
            bm = str(bm).zfill(9)
        bitmasks.append(bm)
    return bitmasks

def export_bars_to_midi(bars_ids, id2bitmask, out_midi, bpm=100, velocity=90, note_len=0.05):
    """
    bars_ids: list[list[int]]  (cada sublista debe ser de longitud 16)
    """
    midi = pm.PrettyMIDI(initial_tempo=bpm)
    drum = pm.Instrument(program=0, is_drum=True, name="Drums")

    seconds_per_beat = 60.0 / float(bpm)
    bar_duration = seconds_per_beat * 4.0   # 4/4
    step_duration = bar_duration / 16.0     # 16 subdivisiones

    t0 = 0.0
    for bar in bars_ids:
        # sanity
        if len(bar) != 16:
            raise ValueError("Cada compás debe tener 16 tokens.")
        bitmasks = bar_ids_to_bitmasks(bar, id2bitmask)
        for i, bm in enumerate(bitmasks):
            # bm = '010000100' (9 chars)
            for cls_idx, ch in enumerate(bm):
                if ch == '1':
                    pitch = CLASS_PITCHES[cls_idx]
                    start = t0 + i * step_duration
                    end = start + note_len
                    drum.notes.append(pm.Note(velocity=velocity, pitch=pitch, start=start, end=end))
        t0 += bar_duration  # avanza al siguiente compás

    midi.instruments.append(drum)
    P.Path(out_midi).parent.mkdir(parents=True, exist_ok=True)
    midi.write(out_midi)
    print(f"[OK] MIDI escrito en: {out_midi}")

def read_jsonl_bars(jsonl_path):
    bars = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            bars.append(obj["tokens"])
    return bars

def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars_jsonl", required=True, help="Salida de sample_loops.jsonl")
    ap.add_argument("--vocab_path", required=True, help="vocab_topN.json (bitmask->id)")
    ap.add_argument("--out_midi", required=True)
    ap.add_argument("--bpm", type=int, default=100)
    ap.add_argument("--velocity", type=int, default=90)
    ap.add_argument("--note_len", type=float, default=0.05)
    args = ap.parse_args()

    _, id2bitmask, unk_id = load_vocab(args.vocab_path)
    bars = read_jsonl_bars(args.bars_jsonl)

    export_bars_to_midi(
        bars_ids=bars,
        id2bitmask=id2bitmask,
        out_midi=args.out_midi,
        bpm=args.bpm,
        velocity=args.velocity,
        note_len=args.note_len
    )

if __name__ == "__main__":
    main()
