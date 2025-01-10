import numpy as np
from scipy.stats import chi2


def _coin_test(rY, rX, type="max"):
    n = rY.shape[0]
    dim_rX = 1 if rX.ndim == 1 else rX.shape[1]
    dim_rY = 1 if rY.ndim == 1 else rY.shape[1]
    rY_cm = np.mean(rY, axis=0)
    rX_cs = np.sum(rX, axis=0)
    mu = np.outer(rX_cs, rY_cm).reshape(-1, order='F')
    rY_center = rY - rY_cm
    rY_cv = rY_center.T @ rY_center

    rX_sp = rX.T @ rX
    rX_ps = np.outer(rX_cs, rX_cs)
    sig = n/(n-1) * np.kron(rY_cv, rX_sp) - 1 / (n-1) * np.kron(rY_cv, rX_ps)

    t_obs = (rX.T @ rY).reshape(-1)
    if type == "max":
        # vals, vcts = np.linalg.eig(sig)
        # sig_neg_half = vcts * np.sqrt(1/vals) @ np.linalg.inv(vcts)
        # stat = np.abs(sig_neg_half @ (t_obs - mu))
        stat = np.abs((t_obs - mu)/np.sqrt(np.diag(sig)))
        pval = np.nan  # TODO!!
    elif type == "quad":
        sig_pinv = np.linalg.pinv(sig)
        stat = (t_obs - mu).T @ sig_pinv @ (t_obs - mu)
        pval = 1 - chi2(dim_rX * dim_rY).cdf(stat)

    return pval, stat
