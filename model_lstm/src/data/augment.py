import numpy as np

def rotate_bars(X, Y, n_rot=3):
    # X,Y : (N, T)
    outs = [ (X, Y) ]
    T = X.shape[1]
    for k in range(1, n_rot+1):
        rX = np.roll(X, shift=k, axis=1)
        rY = np.roll(Y, shift=k, axis=1)
        outs.append((rX, rY))
    Xs = np.concatenate([o[0] for o in outs], axis=0)
    Ys = np.concatenate([o[1] for o in outs], axis=0)
    return Xs, Ys

def swap_tokens_mask(X, mapping_dict):
    """
    mapping_dict: {old_id: new_id, ...} aplica swaps esporádicos
    """
    X2 = X.copy()
    for src, dst in mapping_dict.items():
        X2[X == src] = dst
    return X2
