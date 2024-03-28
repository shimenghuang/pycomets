import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from comet import Comet
from helper import _reshape_to_vec
from regression import RegressionMethod, RF


class GCM(Comet):

    def __init__(self):
        self.pval = None
        self.stat = None
        self.df = None
        self.rY = None
        self.rX = None

    def test(self, Y, X, Z,
             reg_yz: RegressionMethod = RF(),
             reg_xz: RegressionMethod = RF()):
        """
        TODO
        """
        reg_yz.fit(Y=Y, X=Z)
        self.rY = reg_yz.residuals(Y=Y, X=Z)
        if X.ndim == 1:
            reg_xz.fit(Y=X, X=Z)
            self.rX = reg_xz.residuals(Y=X, X=Z)
        else:
            def _comp_resid(x):
                reg_xz.fit(Y=x, X=Z)
                return reg_xz.residuals(Y=x, X=Z)
            self.rX = np.apply_along_axis(_comp_resid, axis=0, arr=X)
        self.pval, self.stat, self.df = _gcm_test(self.rY, self.rX)

    def summary(self, digits=3):
        print("\tGeneralized covariance measure test")
        print(
            f'X-squared = {self.stat:.{digits}f}, df = {self.df}, p-value = {self.pval:.{digits}f}')
        print(
            "alternative hypothesis: true E[cov(Y, X | Z)] is not equal to 0")

    def plot(self):
        fig, ax = plt.subplots(1, 1)

        def _scatter(rx):
            ax.scatter(x=rx, y=self.rY)
        np.apply_along_axis(_scatter, axis=0, arr=self.rX)
        ax.set_xlabel("Residuals X | Z")
        ax.set_ylabel("Residuals Y | Z")
        return fig, ax


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
