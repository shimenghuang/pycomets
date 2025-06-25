import itertools
import warnings
import numpy as np
from scipy.stats import chi2, norm, spearmanr, pearsonr
from .comet import Comet
from .utils import _safe_atleast_2d, _safe_squeeze, _data_check, _split_sample
from .regression import RegressionMethod, RF, DefaultMultiRegression, KRR


class GCM(Comet):
    """
    Generalised covariance measure (GCM).

    The generalised covariance measure test tests whether the conditional
    covariance of Y and X given Z is zero.

    Parameters:
    ----------

    Attributes:
    ----------
    pval : float
        The p-value of the `hypothesis`.

    stat : float
        The value of the test statistic.

    df : int
        The degree of freedom.

    rY : array of shape (n_sample,)
        Residuals of the Y on Z regression.

    rX : array of shape (n_sample, n_feature_x)
        Residuals of the X on Z regression.

    summary_title : string
        The string "Generalized covariance measure test".

    hypothesis : string
        String specifying the null hypothesis.

    References:
    ----------
    Rajen D. Shah, Jonas Peters "The hardness of conditional independence
    testing and the generalised covariance measure," The Annals of Statistics,
    48(3) 1514-1538.
    [doi:10.1214/19-aos1857](https://doi.org/10.1214/19-aos1857)
    """

    def __init__(self):
        self.pval = None
        self.stat = None
        self.df = None
        self.rY = None
        self.rX = None
        self.summary_title = "Generalized covariance measure test"
        self.hypothesis = "E[cov(Y, X | Z)]"

    def test(
        self,
        Y,
        X,
        Z,
        reg_yz: RegressionMethod = RF(),
        reg_xz: RegressionMethod = RF(),
        mreg_xz=None,
        alternative="two.sided",
        test_type="quadratic",
        B=499,
        show_summary=True,
        summary_digits=3,
    ):
        """
        Computation of the GCM test.

        Parameters
        ----------
        Y : array of shape (n_sample,) or (n_sample, 1)
            Response values.

        X : array of shape (n_sample, n_feature_x)
            Values of the first set of covariates.

        Z : array of shape (n_sample, n_feature_z)
            Values of the second set of covariates.

        Returns
        -------
        """
        Y, X, Z = _data_check(Y, X, Z)
        reg_yz.fit(Y=Y, X=Z)
        self.rY = reg_yz.residuals(Y=Y, X=Z)
        if reg_yz.resid_type == "score":
            self.hypothesis = "E[cov(rY, X | Z)]"
            self.summary_title = "TRAM-Generalized covariance measure test"
        if mreg_xz is None:
            dim_X = 1 if X.ndim == 1 else X.shape[1]
            mreg_xz = DefaultMultiRegression(model=reg_xz, dim=dim_X)
        mreg_xz.fit(Y=X, X=Z)
        # Note: here assumes that mreg_xz
        self.rX = mreg_xz.residuals(Y=X, X=Z)
        self.pval, self.stat, self.df = _gcm_test(rY=self.rY, 
                                                  rX=self.rX, 
                                                  alternative=alternative, 
                                                  test_type=test_type, 
                                                  B=B)
        if show_summary:
            self.summary(digits=summary_digits)

    def summary(self, digits=3):
        """
        Print the test results.
        """
        print(f"\t{self.summary_title}")
        print(
            f"X-squared = {self.stat:.{digits}f}, df = {self.df}, p-value = {self.pval:.{digits}f}"
        )
        print(f"alternative hypothesis: true {self.hypothesis} is not equal to 0")

    def plot(self, colors=None, **kwargs):
        """
        Plot the residuals of X on Z regression versus Y on Z regression.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        except ImportError as e:
            raise ImportError("To use `GCM.plot`, please install the 'plot' extra, e.g., pip install pycomets[plot]") from e
        
        fig, ax = plt.subplots(1, 1)
        if colors is None:
            colors = cm.tab20.colors  # or cm.get_cmap("tab10")(i)
        s = kwargs.pop("s", 1.0)
        rX = _safe_atleast_2d(self.rX)
        rY = _safe_atleast_2d(self.rY)
        for k, (i, j) in enumerate(itertools.product(range(rX.shape[1]), 
                                                     range(rY.shape[1]))):
            ax.scatter(rX[:, i], 
                       rY[:, j], 
                       label = rf"$X^{{{i}}}$ and $Y^{{{j}}}$",
                       color=colors[k % len(colors)],
                       s = s,
                       **kwargs)
        ax.set_xlabel("Residuals X | Z")
        ax.set_ylabel("Residuals Y | Z")
        ax.legend()
        return fig, ax

    def get_resids(self):
        return self.rX, self.rY

    def get_cor(self, type="pearson"):
        """
        type : str
            one of "pearson" or "spearman"
        """
        dim_rX = 1 if self.rX.ndim == 1 else self.rX.shape[1]
        cor = np.zeros(dim_rX)
        if type == "pearson":
            for ii in np.arange(dim_rX):
                cor[ii] = pearsonr(self.rX[:, ii], self.rY).statistic
        elif type == "spearman":
            for ii in np.arange(dim_rX):
                cor[ii] = spearmanr(self.rX[:, ii], self.rY).statistic
        return cor


class WGCM(Comet):
    """
    Weighted generalised covariance measure (WGCM).

    The weighted generalised covariance measure test tests whether a weighted
    version of the conditional covariance of Y and X given Z is zero.

    Parameters:
    ----------

    Attributes:
    ----------
    pval : float
        The p-value of the `hypothesis`.

    stat : float
        The value of the test statistic.

    df : int
        The degree of freedom.

    rY : array of shape (n_sample,)
        Residuals of the Y on Z regression.

    rX : array of shape (n_sample, n_feature_x)
        Residuals of the X on Z regression.

    W : array of shape (n_sample, n_feature_x)
        Estimated weight matrix.

    summary_title : string
        The string "Weighted generalized covariance measure test".

    hypothesis : string
        String specifying the null hypothesis.

    References:
    ----------
    Scheidegger, C., Hörrmann, J., & Bühlmann, P. (2022). The weighted
    generalised covariance measure. Journal of Machine Learning Research,
    23(273), 1-68.
    """

    def __init__(self):
        self.pval = None
        self.stat = None
        self.df = None
        self.rY = None
        self.rX = None
        self.W = None
        self.summary_title = "Weighted generalized covariance measure test"
        self.hypothesis = "E[w(Z) cov(Y, X | Z)]"

    def test(
        self,
        Y,
        X,
        Z,
        reg_yz: RegressionMethod = RF(),
        reg_xz: RegressionMethod = RF(),
        reg_wz: RegressionMethod = KRR(kernel="rbf", param_grid={"alpha": [0.1, 1]}),
        mreg_xz=None,
        mreg_wz=None,
        test_split=0.5,
        rng=np.random.default_rng(),
        alternative="two.sided",
        test_type="quadratic",
        B=499,
        show_summary=True, 
        summary_digits=3,
    ):
        """
        Computation of the WGCM test.

        Parameters
        ----------
        Y : array of shape (n_sample,) or (n_sample, 1)
            Response values.

        X : array of shape (n_sample, n_feature_x)
            Values of the first set of covariates.

        Z : array of shape (n_sample, n_feature_z)
            Values of the second set of covariates.

        reg_yz : RegressionMethod
            Regression method for Y on Z regression. Default to be `RF()`.

        reg_xz : RegressionMethod
            Regression method for X on Z regression. Default to be `RF()`.

        reg_wz : RegressionMethod
            Regression method for the residual product on Z regression. Default
            to be `KRR(kernel="rbf", param_grid={'alpha': [0.1, 1]})`.

        mreg_xz : RegressionMethod
            Multivarite regression method for X on Z regression. If `None`,
            `DefaultMultiRegression` is used as a wrapper for the given
            `reg_xz`.

        mreg_wz : RegressionMethod
            Multivarite regression method for the residual product on Z Z
            regression. If `None`, `DefaultMultiRegression` is used as a wrapper
            for the given `reg_xz`.

        test_split : float
            Relative size of test split.

        rng : numpy.random._generator.Generator
            Random number generator.

        summary_digits : int
            Number of digits to display in the printed summary.

        Returns
        -------
        """
        Y, X, Z = _data_check(Y, X, Z)
        # estimate weight function
        Ytr, Xtr, Ztr, Yte, Xte, Zte = _split_sample(Y, X, Z, test_split, rng)
        reg_yz.fit(Y=Ytr, X=Ztr)
        rYw = reg_yz.residuals(Y=Ytr, X=Ztr)

        dim_X = 1 if X.ndim == 1 else X.shape[1]
        if mreg_xz is None:
            mreg_xz = DefaultMultiRegression(reg_xz, dim=dim_X)
        if mreg_wz is None:
            mreg_wz = DefaultMultiRegression(reg_wz, dim=dim_X)

        mreg_xz.fit(Y=Xtr, X=Ztr)
        rXw = mreg_xz.residuals(Y=Xtr, X=Ztr)
        # if rYw.ndim == 1:
        #     rYw = rYw[:, np.newaxis]
        rYw = _safe_atleast_2d(rYw)
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
        self.pval, self.stat, self.df = _gcm_test(rY=self.rY, 
                                                  rX=self.rX * self.W, 
                                                  alternative=alternative, 
                                                  test_type=test_type, 
                                                  B=B)
        # self.pval, self.stat, self.df = _gcm_test(self.rY, self.rX * self.W)
        if show_summary:
            self.summary(digits=summary_digits)

    def summary(self, digits=3):
        """
        Print the test results.
        """
        print(f"\t{self.summary_title}")
        print(
            f"X-squared = {self.stat:.{digits}f}, df = {self.df}, p-value = {self.pval:.{digits}f}"
        )
        print(f"alternative hypothesis: true {self.hypothesis} is not equal to 0")

    # def plot(self):
    #     """
    #     Plot the residuals of X on Z regression versus Y on Z regression.
    #     """
    #     fig, ax = plt.subplots(1, 1)

    #     def _scatter(rx):
    #         ax.scatter(x=rx, y=self.rY)

    #     np.apply_along_axis(_scatter, axis=0, arr=self.rX * self.W)
    #     ax.set_xlabel("Weighted Residuals X | Z")
    #     ax.set_ylabel("Residuals Y | Z")
    #     return fig, ax

    def plot(self, colors=None, **kwargs):
        """
        Plot the residuals of X on Z regression versus Y on Z regression.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        except ImportError as e:
            raise ImportError("To use `WGCM.plot`, please install the 'plot' extra, e.g., pip install pycomets[plot]") from e
        
        fig, ax = plt.subplots(1, 1)
        if colors is None:
            colors = cm.tab20.colors  # or cm.get_cmap("tab10")(i)
        s = kwargs.pop("s", 1.0)
        rXW = _safe_atleast_2d(self.rX * self.W)
        rY = _safe_atleast_2d(self.rY)
        for k, (i, j) in enumerate(itertools.product(range(rXW.shape[1]), 
                                                     range(rY.shape[1]))):
            ax.scatter(rXW[:, i], 
                       rY[:, j], 
                       label = rf"$X^{{{i}}}$ and $Y^{{{j}}}$",
                       color=colors[k % len(colors)],
                       s = s,
                       **kwargs)
        ax.set_xlabel("Residuals X | Z")
        ax.set_ylabel("Residuals Y | Z")
        ax.legend()
        return fig, ax


# def _gcm_test(rY, rX):
#     """
#     Computation of the GCM test based on residuals.
#     Note: currently only support rY of shape (nsample,)
#     """
#     nn = rY.shape[0]
#     rY = _safe_squeeze(rY)
#     rX = _safe_squeeze(rX)
#     dim_rX = 1 if rX.ndim == 1 else rX.shape[1]
#     if dim_rX > 1:
#         rmat = rX * rY[:, np.newaxis]
#         rmat_cm = rmat.mean(axis=0)[:, np.newaxis]
#         sig = rmat.T.dot(rmat) / nn - rmat_cm.dot(rmat_cm.T)
#         eig_val, eig_vec = np.linalg.eig(sig)
#         sig_inv_half = eig_vec @ np.diag(eig_val ** (-1 / 2)) @ eig_vec.T
#         tstat = sig_inv_half @ rmat.sum(axis=0) / np.sqrt(nn)
#     else:
#         rvec = rY * rX
#         rvec_m = rvec.mean()
#         tstat = np.sqrt(nn) * rvec_m / np.sqrt((rvec**2).mean() - rvec_m**2)
#     stat = np.sum(tstat**2)
#     pval = 1 - chi2(dim_rX).cdf(stat)
#     return pval, stat, dim_rX

def _gcm_test(rY, rX, alternative="two.sided", test_type="quadratic", B=499):

    # Check inputs
    alternatives = ["two.sided", "greater", "less"]
    if alternative not in alternatives:
        raise ValueError(f"Invalid alternative '{alternative}'. Must be one of {alternatives}.")
    test_types = ["quadratic", "max"]
    if test_type not in test_types:
        raise ValueError(f"Invalid type '{test_type}'. Must be one of {test_types}.")
    if rY.shape[0] != rX.shape[0]:
        raise ValueError("rY and rX must have the same number of rows")

    nn = rY.shape[0]
    # squeeze into vectors if 1d
    rY = _safe_squeeze(rY, axis=1)
    rX = _safe_squeeze(rX, axis=1)
    dim_rX = 1 if rX.ndim == 1 else rX.shape[1]
    dim_rY = 1 if rY.ndim == 1 else rY.shape[1]
    if dim_rX == 1 and dim_rY == 1:
        rvec = rY * rX
        rvec_m = rvec.mean()
        stat = np.sqrt(nn) * rvec_m / np.sqrt((rvec**2).mean() - rvec_m**2)
        if alternative == "two.sided":
            pval = 2 * norm.cdf(-np.abs(stat))
        elif alternative == "greater":
            pval = 1 - norm.cdf(stat)
        elif alternative == "less":
            pval = norm.cdf(stat)
    else:
        rY = _safe_atleast_2d(rY)
        rX = _safe_atleast_2d(rX)
        tiled_rY = rY[:, np.repeat(np.arange(dim_rY), dim_rX)]
        tiled_rX = rX[:, np.tile(np.arange(dim_rX), dim_rY)]
        RR = tiled_rY * tiled_rX
        if test_type == "quadratic":
            Sig = (RR.T @ RR) / nn - np.outer(RR.mean(axis=0), RR.mean(axis=0))
            eigvals, eigvecs = np.linalg.eigh(Sig)
            if np.min(eigvals) < np.finfo(float).eps:
                warnings.warn("`vcov` of test statistic is not invertible")
            Sig_inv_half = eigvecs @ np.diag(eigvals ** (-0.5)) @ eigvecs.T
            tstat = Sig_inv_half @ RR.sum(axis=0) / np.sqrt(nn)
            stat = np.sum(tstat ** 2)
            pval = 1 - chi2(dim_rX * dim_rY).cdf(stat)
        else:
            tRR = RR.T
            mRR = tRR.mean(axis=1)
            vRR = np.mean(tRR**2, axis=1) - mRR**2
            tRR_normalized = tRR / np.sqrt(vRR[:, np.newaxis])
            stat = np.max(np.abs(mRR)) * np.sqrt(nn)
            rand_mat = np.random.randn(nn, B)
            sim = np.abs(tRR_normalized @ rand_mat)
            sim_max = sim.max(axis=0) / np.sqrt(nn)
            pval = (np.sum(sim_max >= stat) + 1) / (B + 1)

    return pval, stat, dim_rX * dim_rY

