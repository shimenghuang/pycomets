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


def _split_sample(Y, X, Z, test_split=0.5, rng=np.random.default_rng()):
    nn = Y.shape[0]
    idx_tr = rng.choice(np.arange(nn), replace=False,
                        size=int(np.ceil(nn * (1-test_split))))
    idx_te = np.setdiff1d(np.arange(nn), idx_tr, assume_unique=True)
    return Y[idx_tr], X[idx_tr, :], Z[idx_tr, :], Y[idx_te], X[idx_te, :], Z[idx_te, :]


def _get_valid_args(func, args_dict):
    '''
    Return dictionary without invalid function arguments.
    Modified from https://stackoverflow.com/a/196978
    '''
    validArgs = func.__code__.co_varnames[:func.__code__.co_argcount]
    return dict((key, value) for key, value in args_dict.items()
                if key in validArgs)


def _cov_to_cor(sig):
    stds = np.sqrt(np.diag(sig))
    stds_inv = 1/stds
    cor = stds_inv * sig * stds_inv
    return cor
