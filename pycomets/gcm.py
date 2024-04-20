import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from comet import Comet
from helper import _reshape_to_vec, _data_check, _split_sample
from regression import RegressionMethod, RF, DefaultMultiRegression, KRR


class GCM(Comet):

    def __init__(self):
        self.pval = None
        self.stat = None
        self.df = None
        self.rY = None
        self.rX = None
        self.summary_title = "Generalized covariance measure test"
        self.hypothesis = "E[cov(Y, X | Z)]"

    def test(self, Y, X, Z,
             reg_yz: RegressionMethod = RF(),
             reg_xz: RegressionMethod = RF(),
             mreg_xz=None,
             summary_digits=3):
        """
        TODO
        """
        Y, X, Z = _data_check(Y, X, Z)
        reg_yz.fit(Y=Y, X=Z)
        self.rY = reg_yz.residuals(Y=Y, X=Z)
        if reg_yz.resid_type == "score":
            self.hypothesis = "E[cov(rY, X | Z)]"
            self.summary_title = "TRAM-Generalized covariance measure test"
        if mreg_xz is None:
            mreg_xz = DefaultMultiRegression(reg_xz, X.shape[1])
        mreg_xz.fit(Y=X, X=Z)
        self.rX = mreg_xz.residuals(Y=X, X=Z)
        self.pval, self.stat, self.df = _gcm_test(self.rY, self.rX)
        self.summary(digits=summary_digits)

    def summary(self, digits=3):
        print(f"\t{self.summary_title}")
        print(
            f'X-squared = {self.stat:.{digits}f}, df = {self.df}, p-value = {self.pval:.{digits}f}')
        print(
            f"alternative hypothesis: true {self.hypothesis} is not equal to 0")

    def plot(self):
        fig, ax = plt.subplots(1, 1)

        def _scatter(rx):
            ax.scatter(x=rx, y=self.rY)
        np.apply_along_axis(_scatter, axis=0, arr=self.rX)
        ax.set_xlabel("Residuals X | Z")
        ax.set_ylabel("Residuals Y | Z")
        return fig, ax


class WGCM(Comet):

    def __init__(self):
        self.pval = None
        self.stat = None
        self.df = None
        self.rY = None
        self.rX = None
        self.W = None
        self.summary_title = "Generalized covariance measure test"
        self.hypothesis = "E[w(Z) cov(Y, X | Z)]"

    def test(self, Y, X, Z,
             reg_yz: RegressionMethod = RF(),
             reg_xz: RegressionMethod = RF(),
             reg_wz: RegressionMethod = KRR(kernel="rbf", param_grid={'alpha': [0.1, 1]}),
             mreg_xz=None,
             mreg_wz=None,
             test_split=0.5,
             rng=np.random.default_rng(),
             summary_digits=3):
        """
        TODO
        """
        Y, X, Z = _data_check(Y, X, Z)
        # estimate weight function
        Ytr, Xtr, Ztr, Yte, Xte, Zte = _split_sample(
            Y, X, Z, test_split, rng)
        reg_yz.fit(Y=Ytr, X=Ztr)
        rYw = reg_yz.residuals(Y=Ytr, X=Ztr)

        if mreg_xz is None:
            mreg_xz = DefaultMultiRegression(reg_xz, dim=X.shape[1])
        if mreg_wz is None:
            mreg_wz = DefaultMultiRegression(reg_wz, dim=X.shape[1])

        mreg_xz.fit(Y=Xtr, X=Ztr)
        rXw = mreg_xz.residuals(Y=Xtr, X=Ztr)
        if rYw.ndim == 1:
            rYw = rYw[:, np.newaxis]
        res_prod = rXw * rYw
        mreg_wz.fit(Y=res_prod, X=Ztr)
        self.W = np.sign(mreg_wz.predict(X=Zte))
        reg_yz.fit(Y=Yte, X=Zte)
        self.rY = reg_yz.residuals(Y=Yte, X=Zte)
        if reg_yz.resid_type == "score":
            self.hypothesis = "E[w(Z) cov(rY, X | Z)]"
            self.summary_title = "Weighted TRAM-generalized covariance measure test"
        mreg_xz.fit(Y=Xte, X=Zte)
        self.rX = mreg_xz.residuals(Y=Xte, X=Zte)
        self.pval, self.stat, self.df = _gcm_test(self.rY, self.rX * self.W)
        self.summary(digits=summary_digits)

    def summary(self, digits=3):
        print(f"\t{self.summary_title}")
        print(
            f'X-squared = {self.stat:.{digits}f}, df = {self.df}, p-value = {self.pval:.{digits}f}')
        print(
            f"alternative hypothesis: true {self.hypothesis} is not equal to 0")

    def plot(self):
        fig, ax = plt.subplots(1, 1)

        def _scatter(rx):
            ax.scatter(x=rx, y=self.rY)
        np.apply_along_axis(_scatter, axis=0, arr=self.rX * self.W)
        ax.set_xlabel("Weighted Residuals X | Z")
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
