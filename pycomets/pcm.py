import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import norm

from .comet import Comet
from .utils import _data_check, _split_sample
from .regression import RF, RegressionMethod


class PCM(Comet):
    """
    Projected covariance measure test for conditional mean independence.

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

    rT : array of shape (n_sample,)
        Residuals of the transformation of Y and Z on Z regression.


    References:
    ----------
    Lundborg, A. R., Kim, I., Shah, R. D., & Samworth, R. J. (2022). The
    Projected Covariance Measure for assumption-lean variable significance
    testing. arXiv preprint.
    [10.48550/arXiv.2211.02039](https://arxiv.org/abs/2211.02039)

    """

    def __init__(self):
        self.pval = None
        self.stat = None
        self.pvals = None
        self.stats = None
        self.df = None
        self.rY = None
        self.rT = None

    def test(
        self,
        Y,
        X,
        Z,
        rep=1,
        reg_yonxz: RegressionMethod = RF(),
        reg_yonz: RegressionMethod = RF(),
        reg_yhatonz: RegressionMethod = RF(),
        reg_vonxz: RegressionMethod = RF(),
        reg_ronz: RegressionMethod = RF(),
        estimate_variance=True,
        test_split=0.5,
        max_exp=5,
        rng=np.random.default_rng(),
        summary_digits=3,
    ):
        """
        Computation of the PCM test.

        Parameters
        ----------
        Y : array of shape (n_sample,) or (n_sample, 1)
            Response values.

        X : array of shape (n_sample, n_feature_x)
            Values of the first set of covariates.

        Z : array of shape (n_sample, n_feature_z)
            Values of the second set of covariates.

        rep : int
            Number of repetitions with which to repeat the PCM test.

        reg_yonxz : RegressionMethod
            TODO

        reg_yonz : RegressionMethod
            TODO

        reg_yhatonz : RegressionMethod
            TODO

        reg_vonxz : RegressionMethod
            TODO

        reg_ronz : RegressionMethod
            TODO

        estimate_variance : bool

        test_split : float
            Relative size of test split.

        rng : numpy.random._generator.Generator
            Random number generator.

        summary_digits : int
            Number of digits to display in the printed summary.

        Returns
        -------
        """
        self.pvals = np.empty(rep)
        self.stats = np.empty(rep)
        self.rY = np.empty((int(np.floor(Y.shape[0] * test_split)), rep))
        self.rT = np.empty((int(np.floor(Y.shape[0] * test_split)), rep))
        for ii in range(rep):
            self.pvals[ii], self.stats[ii], self.rY[:, ii], self.rT[:, ii] = (
                _pcm_test(
                    Y,
                    X,
                    Z,
                    reg_yonxz,
                    reg_yonz,
                    reg_yhatonz,
                    reg_vonxz,
                    reg_ronz,
                    estimate_variance,
                    test_split,
                    max_exp,
                    rng,
                )
            )
        self.stat = np.mean(self.stats)
        self.pval = 1 - norm().cdf(self.stat)
        self.summary(digits=summary_digits)

    def summary(self, digits=3):
        print("\tProjected covariance measure test")
        print(f"Z = {self.stat:.{digits}f}, p-value = {self.pval:.{digits}f}")
        print(
            "alternative hypothesis: true E[Y | X, Z] is not equal to E[Y | Z]"
        )

    # def plot(self):
    #     fig, ax = plt.subplots(1, 1)
    #     for idx in range(self.rT.shape[1]):
    #         ax.scatter(x=self.rT[:, idx], y=self.rY[:, idx])
    #     ax.set_xlabel("Residuals f(X, Z) | Z")
    #     ax.set_ylabel("Residuals Y | Z")
    #     return fig, ax

    def plot(self, colors=None, **kwargs):
        """
        Plot the residuals of X on Z regression versus Y on Z regression.
        """
        fig, ax = plt.subplots(1, 1)
        if colors is None:
            colors = cm.tab20.colors  # or cm.get_cmap("tab10")(i)
        s = kwargs.pop("s", 1.0)
        for idx in range(self.rT.shape[1]):
            ax.scatter(x=self.rT[:, idx], 
                       y=self.rY[:, idx],
                       label = rf"$f(X,Z)^{{{idx}}}$",
                       color=colors[idx % len(colors)],
                       s = s,
                       **kwargs)
        ax.set_xlabel("Residuals f(X, Z) | Z")
        ax.set_ylabel("Residuals Y | Z")
        ax.legend()
        return fig, ax


def _pcm_test(
    Y,
    X,
    Z,
    reg_yonxz: RegressionMethod = RF(),
    reg_yonz: RegressionMethod = RF(),
    reg_yhatonz: RegressionMethod = RF(),
    reg_vonxz: RegressionMethod = RF(),
    reg_ronz: RegressionMethod = RF(),
    estimate_variance=True,
    test_split=0.5,
    max_exp=5,
    rng=np.random.default_rng(),
):
    """
    Computation of the PCM test with data splitting.
    """

    # sample splitting
    Y, X, Z = _data_check(Y, X, Z)
    if Y.ndim > 1:
        raise ValueError(f'PCM does not support multi-dimensional Y.')
    Ytr, Xtr, Ztr, Yte, Xte, Zte = _split_sample(Y, X, Z, test_split, rng)

    # regression on the training data (direction estimate)
    XZtr = np.column_stack([Xtr, Ztr])
    reg_yonxz.fit(Y=Ytr, X=XZtr)
    yhat = reg_yonxz.predict(X=XZtr)
    reg_yhatonz.fit(Y=yhat, X=Ztr)
    rho = np.mean((Ytr - reg_yhatonz.predict(X=Ztr)) * yhat)

    def hhat(X, Z):
        htilde = reg_yonxz.predict(
            np.column_stack([X, Z])
        ) - reg_yhatonz.predict(Z)
        return np.sign(rho) * htilde

    # estimate variance
    if estimate_variance:
        sqr = (Ytr - yhat) ** 2
        reg_vonxz.fit(Y=sqr, X=XZtr)

        def a(c):
            den = np.column_stack(
                [reg_vonxz.predict(XZtr), np.repeat(0, XZtr.shape[0])]
            )
            return np.mean(sqr / (np.max(den, axis=1) + c)) - 1

        if a(0) < 0:
            chat = 0
        else:
            lwr, upr = 0, 10
            counter = 0
            while np.sign(a(lwr)) * np.sign(a(upr)) == 1:
                upr += 5
                counter += 1
                if counter > max_exp:
                    raise ValueError(
                        "Cannot compute variance estimate, try rerunning with `estimate_variance=False`."
                    )
            chat = root_scalar(a, method="brentq", bracket=[lwr, upr]).root

        def vhat(X, Z):
            XZ = np.column_stack([X, Z])
            vtemp = np.max(
                np.column_stack(
                    [reg_vonxz.predict(XZ), np.repeat(0, XZ.shape[0])]
                ),
                axis=1,
            )
            return vtemp + chat

    else:

        def vhat(X, Z):
            return 1

    # regression on the test data
    def fhat(X, Z):
        return hhat(X, Z) / vhat(X, Z)

    fhats = fhat(Xte, Zte)
    reg_ronz.fit(Y=fhats, X=Zte)
    reg_yonz.fit(Y=Yte, X=Zte)

    # test
    rY = Yte - reg_yonz.predict(X=Zte)
    rT = fhats - reg_ronz.predict(X=Zte)
    L = rY * rT
    stat = (
        np.sqrt(Yte.shape[0])
        * np.mean(L)
        / np.sqrt(np.mean(L**2) - np.mean(L) ** 2)
    )
    if np.isnan(stat):
        stat = -np.Inf
    pval = 1 - norm().cdf(stat)

    return pval, stat, rY, rT
