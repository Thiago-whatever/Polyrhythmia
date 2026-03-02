from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import tensorflow as tf


@dataclass
class ModelSpec:
    name: str
    model_path: Path
    vocab_path: Optional[Path] = None
    genres: Optional[List[str]] = None  # para construir style vec si aplica


def infer_signature(model) -> Dict[str, Optional[int]]:
    """
    Infere índices de inputs: tokens / pos / style (si existen).
    Compatible con modelos 1/2/3.
    """
    sig = {"tokens": None, "pos": None, "style": None}
    for i, inp in enumerate(model.inputs):
        name = getattr(inp, "name", f"in_{i}").split(":")[0].lower()
        if "token" in name:
            sig["tokens"] = i
        elif "pos" in name:
            sig["pos"] = i
        elif "style" in name:
            sig["style"] = i
    if sig["tokens"] is None:
        sig["tokens"] = 0
    return sig


def top_k_sample(probs: np.ndarray, k: int = 8, temperature: float = 1.0, rest_ids: Optional[set[int]] = None) -> int:
    p = np.asarray(probs, dtype=np.float64)
    p = np.maximum(p, 1e-12)

    if temperature != 1.0:
        p = np.power(p, 1.0 / temperature)

    k = min(k, p.size)
    idx = np.argpartition(p, -k)[-k:]
    sub = p[idx].copy()
    sub = sub / sub.sum()

    if rest_ids:
        mask = ~np.isin(idx, list(rest_ids))
        if mask.any():
            idx = idx[mask]
            sub = sub[mask]
            sub = sub / sub.sum()

    return int(np.random.choice(idx, p=sub))


def generate_bar_tokens(
    model,
    T: int,
    sig: Dict[str, Optional[int]],
    top_k: int = 8,
    temperature: float = 1.0,
    style_vec: Optional[np.ndarray] = None,
    rest_ids: Optional[set[int]] = None,
) -> np.ndarray:
    """
    Genera una barra de T pasos como secuencia de tokens (T,).

    Maneja modelos donde:
      - tokens: (B,T)
      - pos:   (B,T)  (si existe)
      - style: (B,G) o (B,T,G) (según el modelo)
    """
    X = np.zeros((1, T), dtype=np.int32)
    pos = np.arange(1, T + 1, dtype=np.int32)[None, :]  # (1,T)

    feeds = [None] * len(model.inputs)

    # tokens
    feeds[sig["tokens"]] = X

    # pos (si aplica)
    if sig.get("pos") is not None:
        feeds[sig["pos"]] = pos

    # style (si aplica)
    if sig.get("style") is not None:
        if style_vec is None:
            raise ValueError("El modelo requiere style_vec pero no se proporcionó.")

        # Verificar compatibilidad con la forma esperada del input style
        style_input = model.inputs[sig["style"]]
        # KerasTensor.shape suele ser (None, G) o (None, T, G)
        rank = len(style_input.shape)
        if rank == 2:
            # espera (B,G)
            if style_vec.ndim == 3:
                # convertir (1,T,G) -> (1,G) (toma el primer paso; todos iguales)
                style_vec_use = style_vec[:, 0, :]
            else:
                style_vec_use = style_vec
        elif rank == 3:
            # espera (B,T,G)
            if style_vec.ndim == 2:
                # expandir (1,G) -> (1,T,G)
                style_vec_use = np.repeat(style_vec[:, None, :], repeats=T, axis=1)
            else:
                style_vec_use = style_vec
        else:
            raise ValueError(f"Input style con rank inesperado: {rank}")

        feeds[sig["style"]] = style_vec_use

    # Autoregresivo dentro de la barra
    for t in range(T):
        y = model(feeds, training=False).numpy()  # (1,T,V)
        probs_t = y[0, t]
        tok = top_k_sample(probs_t, k=top_k, temperature=temperature, rest_ids=rest_ids)
        X[0, t] = tok
        feeds[sig["tokens"]] = X

    return X[0].astype(np.int32)


def build_style_vec(genre_index: int, G: int, T: int, style_rank: int) -> np.ndarray:
    """
    Construye style vector compatible con el modelo.

    style_rank:
      - 2  => (1, G)   condición global por barra
      - 3  => (1, T, G) condición repetida por paso
    """
    if style_rank == 2:
        v = np.zeros((1, G), dtype=np.float32)
        v[0, genre_index] = 1.0
        return v
    elif style_rank == 3:
        v = np.zeros((1, T, G), dtype=np.float32)
        v[0, :, genre_index] = 1.0
        return v
    else:
        raise ValueError(f"style_rank no soportado: {style_rank} (esperado 2 o 3)")


def load_model_no_compile(model_path: Path):
    """
    Carga el modelo sin compilar (como haces en notebook).
    Si usas un Loss custom serializable, tendrás que registrar el scope aquí también,
    pero para inferencia normalmente compile=False es suficiente.
    """
    return tf.keras.models.load_model(str(model_path), compile=False)