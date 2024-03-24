import numpy as np
from scipy.stats import chi2
# from comet import Comet
from pycomets.helper import _reshape_to_vec

# class GCM(Comet):

#     def __init__(self, Y, X, Z):
#         print("hello")


def _gcm_test(rY, rX):
    """
    TODO
    """
    nn = rY.shape[0]
    rY = _reshape_to_vec(rY)
    rX = _reshape_to_vec(rX)
    dim_rX = 1 if rX.ndim == 1 else rX.shape[1]
    if dim_rX > 1:
        rmat = rX * rY[:, np.newaxis]
        rmat_cm = rmat.mean(axis=0)[:, np.newaxis]
        sig = rmat.T.dot(rmat)/nn - rmat_cm.dot(rmat_cm.T)
        eig_val, eig_vec = np.linalg.eig(sig)
        sig_inv_half = eig_vec @ np.diag(eig_val**(-1/2)) @ eig_vec.T
        tstat = sig_inv_half @ rmat.sum(axis=0) / np.sqrt(nn)
    else:
        rvec = rY * rX
        rvec_m = rvec.mean()
        tstat = np.sqrt(nn) * rvec_m / \
            np.sqrt((rvec ** 2).mean() - rvec_m ** 2)
    stat = np.sum(tstat ** 2)
    pval = 1 - chi2(dim_rX).cdf(stat)
    return pval, stat, dim_rX
