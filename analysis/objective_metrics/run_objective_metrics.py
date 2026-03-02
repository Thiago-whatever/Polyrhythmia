from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.objective_metrics.bitmask_io import load_bitmasks_npz
from analysis.objective_metrics.metrics import compute_objective_metrics

from analysis.objective_metrics.generate import (
    load_model_no_compile, infer_signature,
    generate_bar_tokens, build_style_vec
)

# ---- stats (scipy) ----
try:
    from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


import json
import numpy as np
from pathlib import Path

import json
import numpy as np

def int_to_bitvec(x: int, I: int) -> np.ndarray:
    v = np.zeros(I, dtype=np.int8)
    for i in range(I):
        v[i] = (x >> i) & 1
    return v

def token_to_bitvec(token, I: int) -> np.ndarray:
    """
    Intenta convertir representaciones comunes de vocab a bitvec (I,):
    - int / str-int: bitmask comprimido
    - list[int]: índices de instrumentos activos
    - str con separadores: "0,3,5" o "0_3_5" o "0 3 5"
    """
    # Caso 1: token es int o str que representa int
    if isinstance(token, (int, np.integer)):
        return int_to_bitvec(int(token), I)

    if isinstance(token, str):
        s = token.strip()
        # ¿es número puro?
        if s.isdigit():
            return int_to_bitvec(int(s), I)

        # separadores típicos
        for sep in [",", "_", " "]:
            if sep in s:
                parts = [p for p in s.split(sep) if p != ""]
                if all(p.strip().isdigit() for p in parts):
                    idxs = [int(p) for p in parts]
                    v = np.zeros(I, dtype=np.int8)
                    for j in idxs:
                        if 0 <= j < I:
                            v[j] = 1
                    return v

        # si no pudimos, fallamos explícito
        raise ValueError(f"No pude parsear token string a bitmask: {token!r}")

    # Caso 2: lista/tupla de índices
    if isinstance(token, (list, tuple)):
        v = np.zeros(I, dtype=np.int8)
        for j in token:
            if isinstance(j, (int, np.integer)) and 0 <= int(j) < I:
                v[int(j)] = 1
        return v

    raise ValueError(f"Tipo de token no soportado para bitmask: {type(token).__name__}")

import json
import numpy as np

def load_vocab_id2bitmask_from_vocab(vocab_path: str, I: int) -> dict:
    """
    Construye id2bitmask desde vocab.json soportando:
    A) Formato con campos: {'id2bitmask': {...}} o {'id2token': ...}
    B) Formato plano: {'010010100': 51, '000000001': 1, ...}  (bitmask_str -> id)
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # --------- A) Formato con campos ----------
    if isinstance(vocab, dict) and ("id2bitmask" in vocab or "id2token" in vocab):
        if "id2bitmask" in vocab:
            return {int(k): np.asarray(v, dtype=np.int8) for k, v in vocab["id2bitmask"].items()}

        # Derivar desde id2token
        it = vocab["id2token"]
        if isinstance(it, list):
            out = {}
            for i, tok in enumerate(it):
                out[int(i)] = token_to_bitvec(tok, I)  # usa tu helper existente
            return out
        if isinstance(it, dict):
            return {int(k): token_to_bitvec(it[k], I) for k in it.keys()}

    # --------- B) Formato plano bitmask_str -> id ----------
    # Detectamos: keys son strings binarias de longitud I y values numéricos
    if isinstance(vocab, dict):
        sample_k = next(iter(vocab.keys()))
        sample_v = vocab[sample_k]

        is_binary_key = isinstance(sample_k, str) and len(sample_k) == I and set(sample_k) <= {"0", "1"}
        is_numeric_val = isinstance(sample_v, (int, float, np.integer, np.floating))

        if is_binary_key and is_numeric_val:
            id2bitmask = {}
            for bm_str, idx in vocab.items():
                idx_int = int(idx)
                vec = np.array([1 if c == "1" else 0 for c in bm_str], dtype=np.int8)
                if vec.size != I:
                    raise ValueError(f"Bitmask string con longitud inesperada: {bm_str} (len={len(bm_str)})")
                id2bitmask[idx_int] = vec
            return id2bitmask

    raise ValueError(
        f"No pude construir id2bitmask desde vocab.json (formato desconocido). "
        f"Path: {vocab_path}"
    )

import numpy as np

def int_to_bitvec(x: int, I: int) -> np.ndarray:
    """
    Convierte un entero 'x' que representa un bitmask a un vector binario (I,).
    bit 0 -> posición 0 (LSB).
    """
    v = np.zeros(I, dtype=np.int8)
    for i in range(I):
        v[i] = (x >> i) & 1
    return v

def tokens_to_multihot(tokens: np.ndarray, id2bitmask: dict, I: int) -> np.ndarray:
    """
    tokens: (T,)
    Devuelve M: (T, I) multi-hot por paso.
    Soporta id2bitmask[id] como:
      - vector binario shape (I,)
      - entero (bitmask comprimido)
      - array shape (1,) con entero dentro
    """
    T = int(tokens.shape[0])
    M = np.zeros((T, I), dtype=np.int8)

    for t, tok in enumerate(tokens.tolist()):
        bm = id2bitmask.get(int(tok), None)
        if bm is None:
            # token desconocido (UNK): dejamos todo en 0 (silencio)
            continue

        bm_arr = np.asarray(bm)

        # Caso A: ya es vector binario (I,)
        if bm_arr.ndim == 1 and bm_arr.size == I:
            M[t] = bm_arr.astype(np.int8)
            continue

        # Caso B: es escalar o array tamaño 1 (bitmask comprimido)
        if bm_arr.ndim == 0:
            x = int(bm_arr)
            M[t] = int_to_bitvec(x, I)
            continue

        if bm_arr.ndim == 1 and bm_arr.size == 1:
            x = int(bm_arr[0])
            M[t] = int_to_bitvec(x, I)
            continue

        raise ValueError(f"Formato de bitmask inesperado para token={tok}: shape={bm_arr.shape}, dtype={bm_arr.dtype}")

    return M


def sample_human_by_genre(M: np.ndarray, genres: Optional[np.ndarray], genre_index: int, n: int, seed: int) -> np.ndarray:
    """
    M humano (N,T,I). Si genres existe como multi-hot (N,G), filtramos por género.
    Si no existe, sampleamos global.
    """
    rng = np.random.default_rng(seed)
    N = M.shape[0]

    if genres is None:
        idx = rng.choice(N, size=min(n, N), replace=False)
        return M[idx]

    # multi-hot (N,G)
    if genres.ndim == 2:
        mask = genres[:, genre_index] > 0.5
        pool = np.where(mask)[0]
        if pool.size == 0:
            raise ValueError(f"No hay barras humanas para genre_index={genre_index} en test set.")
        idx = rng.choice(pool, size=min(n, pool.size), replace=False)
        return M[idx]

    # labels (N,)
    pool = np.where(genres == genre_index)[0]
    if pool.size == 0:
        raise ValueError(f"No hay barras humanas para genre_index={genre_index} en test set.")
    idx = rng.choice(pool, size=min(n, pool.size), replace=False)
    return M[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset3_test_bitmasks", default="model_lstm/data/processed/dataset_3/test_bitmasks.npz")
    p.add_argument("--genres", nargs="+", default=["jazz","bossa","samba","hiphop","afrocubano","choro"])
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--I", type=int, default=9, help="Número de instrumentos (si tu kit reducido usa 9).")
    p.add_argument("--n_per_genre", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)

    # modelos
    p.add_argument("--model1", default="model_lstm/models/final/best.h5")
    p.add_argument("--model2", default="model_lstm/models/final/best_2.h5")
    p.add_argument("--model3", default="model_lstm/models/final/best_3.h5")

    # vocabs (para convertir tokens -> bitmask)
    p.add_argument("--vocab1", default="model_lstm/data/processed/vocab.json")
    p.add_argument("--vocab2", default="model_lstm/data/processed/dataset_2/vocab.json")
    p.add_argument("--vocab3", default="model_lstm/data/processed/dataset_3/vocab.json")

    # params de sampling
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--rest_ids", nargs="*", type=int, default=[0], help="IDs que representan silencio")
    args = p.parse_args()

    if not SCIPY_OK:
        raise RuntimeError("Necesitas scipy instalado para las pruebas estadísticas (mannwhitneyu/chi2).")

    genres = args.genres
    G = len(genres)
    rest_ids = set(args.rest_ids)

    # --- humanos ---
    human = load_bitmasks_npz(args.dataset3_test_bitmasks)
    M_h_all = human.M  # (N,T,I)
    genres_h = human.genres  # opcional

    # --- modelos IA ---
    model_specs = [
        ("model1", Path(args.model1), Path(args.vocab1)),
        ("model2", Path(args.model2), Path(args.vocab2)),
        ("model3", Path(args.model3), Path(args.vocab3)),
    ]

    rows_summary = []
    rows_inst = []

    for model_name, model_path, vocab_path in model_specs:
        print(f"\n====================== {model_name.upper()} ======================")
        print("model:", model_path)
        print("vocab:", vocab_path)

        model = load_model_no_compile(model_path)
        sig = infer_signature(model)
        # fallback: bitmasks del mismo dataset que el vocab del modelo actual
        fallback_npz = str(Path(vocab_path).parent / "train_bitmasks.npz")
        id2bitmask = load_vocab_id2bitmask_from_vocab(vocab_path, I=args.I)

        # --- por género ---
        for g_idx, g_name in enumerate(genres):
            # humanos del test por género
            M_h = sample_human_by_genre(M_h_all, genres_h, g_idx, n=args.n_per_genre, seed=args.seed + g_idx)

            # IA: si tiene style -> condicionar; si no -> generar igual y etiquetar como "uncond"
            M_ia_list = []
            for k in range(args.n_per_genre):
                style_vec = None
                if sig.get("style") is not None:
                    style_input = model.inputs[sig["style"]]
                    style_rank = len(style_input.shape)  # 2 o 3
                    style_vec = build_style_vec(g_idx, G, args.T, style_rank=style_rank)
                toks = generate_bar_tokens(
                    model=model, T=args.T, sig=sig,
                    top_k=args.top_k, temperature=args.temperature,
                    style_vec=style_vec, rest_ids=rest_ids
                )
                M_ia = tokens_to_multihot(toks, id2bitmask=id2bitmask, I=args.I)
                M_ia_list.append(M_ia)

            M_ia = np.stack(M_ia_list, axis=0).astype(np.float32)  # (n,T,I)

            met_h = compute_objective_metrics(M_h)
            met_ia = compute_objective_metrics(M_ia)

            # --- stats densidad / entropía ---
            u_den = mannwhitneyu(met_ia.density, met_h.density, alternative="two-sided")
            u_ent = mannwhitneyu(met_ia.entropy, met_h.entropy, alternative="two-sided")

            # --- instrumentos: chi² homogeneidad (IA vs humano) ---
            # agregamos conteos globales por instrumento (I,)
            c_h = met_h.inst_counts.sum(axis=0)
            c_ia = met_ia.inst_counts.sum(axis=0)
            table = np.vstack([c_h, c_ia])
            chi2, pval, dof, expected = chi2_contingency(table)

            rows_summary.append({
                "model": model_name,
                "genre": g_name,
                "n_h": int(M_h.shape[0]),
                "n_ia": int(M_ia.shape[0]),
                "density_h_median": float(np.median(met_h.density)),
                "density_ia_median": float(np.median(met_ia.density)),
                "mw_density_U": float(u_den.statistic),
                "mw_density_p": float(u_den.pvalue),
                "entropy_h_median": float(np.median(met_h.entropy)),
                "entropy_ia_median": float(np.median(met_ia.entropy)),
                "mw_entropy_U": float(u_ent.statistic),
                "mw_entropy_p": float(u_ent.pvalue),
                "chi2_inst": float(chi2),
                "chi2_inst_dof": int(dof),
                "chi2_inst_p": float(pval),
            })

            rows_inst.append({
                "model": model_name,
                "genre": g_name,
                **{f"human_inst_{i}": float(c_h[i]) for i in range(len(c_h))},
                **{f"ia_inst_{i}": float(c_ia[i]) for i in range(len(c_ia))},
            })

            print(f"\n[{model_name} | {g_name}]")
            print(f" density median (H vs IA): {np.median(met_h.density):.4f} vs {np.median(met_ia.density):.4f} | p={u_den.pvalue:.4g}")
            print(f" entropy  median (H vs IA): {np.median(met_h.entropy):.4f} vs {np.median(met_ia.entropy):.4f} | p={u_ent.pvalue:.4g}")
            print(f" inst chi2: {chi2:.4f} (dof={dof}) p={pval:.4g}")

    df_sum = pd.DataFrame(rows_summary)
    df_inst = pd.DataFrame(rows_inst)

    print("\n====================== RESUMEN (primeras filas) ======================")
    print(df_sum.head(12).to_string(index=False))

    # Comparación global entre modelos (Kruskal–Wallis) sobre densidad/entropía agregadas por género
    # (opcional, pero útil si quieres “modelo 3 es mejor” en métricas objetivas)
    print("\n====================== KRUSKAL–WALLIS (por modelo, sobre medianas por género) ======================")
    for metric in ["density_ia_median", "entropy_ia_median"]:
        groups = [df_sum[df_sum["model"] == m][metric].values for m in ["model1","model2","model3"]]
        h = kruskal(*groups)
        print(f"{metric}: H={h.statistic:.6f} p={h.pvalue:.6g}")

    # Guardar CSVs
    out_dir = Path("analysis/objective_metrics/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(out_dir / "objective_metrics_summary.csv", index=False, encoding="utf-8")
    df_inst.to_csv(out_dir / "objective_metrics_instruments.csv", index=False, encoding="utf-8")
    print("\nGuardado:")
    print(" -", out_dir / "objective_metrics_summary.csv")
    print(" -", out_dir / "objective_metrics_instruments.csv")


if __name__ == "__main__":
    main()