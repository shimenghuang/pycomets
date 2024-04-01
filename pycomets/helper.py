import numpy as np


def _reshape_to_vec(x):
    x_copy = x.copy()
    if x.ndim > 1 and x.shape[1] == 1:
        x_copy = x_copy.reshape(-1)
    return x_copy


def _data_check(Y, X, Z):
    Y_new, X_new, Z_new = Y.copy(), X.copy(), Z.copy()
    if Y.ndim > 1:
        Y_new = _reshape_to_vec(Y_new)
    if X.ndim == 1:
        X_new = X_new[:, np.newaxis]
    if Z.ndim == 1:
        Z_new = Z_new[:, np.newaxis]
    return Y_new, X_new, Z_new
