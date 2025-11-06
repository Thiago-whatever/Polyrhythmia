import os
import yaml
import json
import numpy as np
import tensorflow as tf

class DataLoader:
    def __init__(self, config_path: str):
        # Leer YAML
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        dataset_cfg = self.config["dataset"]
        self.train_path = dataset_cfg["train"]
        self.val_path = dataset_cfg["val"]
        self.test_path = dataset_cfg["test"]
        self.vocab_path = dataset_cfg["vocab"]
        self.max_seq_len = dataset_cfg["max_seq_len"]
        #self.num_classes = dataset_cfg["num_classes"]
        self.augmentation = dataset_cfg.get("augmentation", False)

        # Cargar vocabulario
        with open(self.vocab_path, "r") as f:
            self.vocab = json.load(f)

        self.num_classes = len(self.vocab)

        print(f"[INFO] Dataset configurado desde {config_path}")
        print(f"[INFO] train: {self.train_path}, val: {self.val_path}, test: {self.test_path}")
        print(f"[INFO] vocab size: {len(self.vocab)} (num_classes esperado: {self.num_classes})")

    def load_split(self, split_path: str):
        data = np.load(split_path, allow_pickle=True)
        if "X" in data.files and "Y" in data.files:
            X, Y = data["X"], data["Y"]
            G = data["G"] if "G" in data.files else None
        elif "X_tokens" in data.files and "Y_tokens" in data.files:
            X, Y = data["X_tokens"], data["Y_tokens"]
            G = data["X_style"] if "X_style" in data.files else None
        else:
            raise KeyError(
                f"Archivo {split_path} sin claves esperadas. "
                f"Se esperaban ('X','Y') o ('X_tokens','Y_tokens')."
            )
        return X, Y, G

    def get_tf_dataset(self, split="train", batch_size=64, shuffle=True):
        if split == "train":
            X, Y, G = self.load_split(self.train_path)
        elif split == "val":
            X, Y, G = self.load_split(self.val_path)
        elif split == "test":
            X, Y, G = self.load_split(self.test_path)
        else:
             raise ValueError(f"Split desconocido: {split}")

        # Si hay estilos, el input será (X, G)
        if G is not None:
            ds = tf.data.Dataset.from_tensor_slices(((X, G), Y))
        else:
            ds = tf.data.Dataset.from_tensor_slices((X, Y))
        if shuffle:
             ds = ds.shuffle(buffer_size=len(X))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="DataLoader demo (no es un CLI de build_vocab).")
    ap.add_argument("--config", required=True, help="Ruta al YAML de dataset (p. ej. configs/dataset_2.yaml)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    loader = DataLoader(args.config)
    train_ds = loader.get_tf_dataset(split="train", batch_size=64)
    
    for batch in train_ds.take(1):
        if isinstance(batch[0], tuple):  # (X, G), Y
            (X, G), Y = batch
            print("[DEBUG] X:", X.shape, "G:", G.shape, "Y:", Y.shape)
        else:                            # X, Y
            X, Y = batch
            print("[DEBUG] X:", X.shape, "Y:", Y.shape)
