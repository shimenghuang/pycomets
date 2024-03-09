def _reshape_to_vec(x):
    x_copy = x.copy()
    if x.ndim > 1 and x.shape[1] == 1:
        x_copy = x_copy.reshape(-1)
    return x_copy
