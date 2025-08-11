import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import numbers


# def _reshape_to_vec(x):
#     x_copy = x.copy()
#     if x.ndim > 1 and x.shape[1] == 1:
#         x_copy = x_copy.reshape(-1)
#     return x_copy

def _safe_squeeze(arr, axis=1):
    """
    Squeeze out axis if it is 1. 
    e.g., if arr is of shape (n, 1) and axis=1, returns a view of shape (n,)
    """

    # Nothing to squeeze if only one dimensional, axis ignored
    if arr.ndim == 1:
        return arr
    
    # Otherwise check dimension vs axis
    if axis >= arr.ndim:
        raise IndexError(f"axis {axis} exceeds the dimension of arr {arr.ndim}")
    if (axis < 0) or (isinstance(axis, (float, np.floating))):
        raise ValueError(f"axis can only be nonnegative integers, got value {axis}")
    
    # In case the axis is only length 1, squeeze it
    if arr.shape[axis] == 1:
        return np.squeeze(arr.copy(), axis=axis)
    else:
        return arr

def _safe_atleast_2d(arr):
    """
    Ensures a column vector of shape (n, 1) in case arr is of shape (n,)
    """
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    else:
        return np.atleast_2d(arr)


def _data_check(Y, X, Z):
    """
    Check data dimensions and reshape if needed.
    Note: Z is the one being conditioned on and thus always should have 
        a second dimension. X in GCM and PCM is never conditioned on by 
        itself, so it can be one dimensional or multi-dimensional. 
        Y is always the response, which can be one or multi-dimensional. 
    """
    Y_new, X_new, Z_new = Y.copy(), X.copy(), Z.copy()
    if Y.ndim > 1:
        Y_new = _safe_squeeze(Y_new, axis=1)
    if X.ndim > 1:
        X_new = _safe_squeeze(X_new, axis=1)
    if Z.ndim == 1:
        Z_new = Z_new[:, np.newaxis]
    return Y_new, X_new, Z_new


def _split_sample(Y, X, Z, test_split=0.5, rng=np.random.default_rng()):
    nn = Y.shape[0]
    idx_tr = rng.choice(np.arange(nn), replace=False,
                        size=int(np.ceil(nn * (1-test_split))))
    idx_te = np.setdiff1d(np.arange(nn), idx_tr, assume_unique=True)
    return (Y[idx_tr], 
            X[idx_tr], 
            Z[idx_tr], 
            Y[idx_te], 
            X[idx_te], 
            Z[idx_te])


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

def _is_numeric(x):
    return isinstance(x, numbers.Number)

def _compute_median_heuristic(z):
    """
    Computes the median pairwise distance over z.
    """
    dists = pairwise_distances(z, metric='euclidean')
    dists_no_diag = dists[np.triu_indices_from(dists, k=1)]
    median = np.median(dists_no_diag)
    return median.item()