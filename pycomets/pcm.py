import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from comet import Comet
from regression import RegressionMethod, RF
from helper import _data_check, _split_sample


class PCM(Comet):

    def __init__(self):
        self.pval = None
        self.stat = None
        self.pvals = None
        self.stats = None
        self.df = None
        self.rY = None
        self.rT = None

    def test(self, Y, X, Z, rep=1,
             reg_yonxz: RegressionMethod = RF(),
             reg_yonz: RegressionMethod = RF(),
             reg_yhatonz: RegressionMethod = RF(),
             reg_vonxz: RegressionMethod = RF(),
             reg_ronz: RegressionMethod = RF(),
             estimate_variance=True,
             test_split=0.5,
             max_exp=5,
             rng=np.random.default_rng(),
             summary_digits=3):
        """
        TODO
        """
        self.pvals = np.empty(rep)
        self.stats = np.empty(rep)
        self.rY = np.empty((int(np.floor(Y.shape[0] * test_split)), rep))
        self.rT = np.empty((int(np.floor(Y.shape[0] * test_split)), rep))
        for ii in range(rep):
            self.pvals[ii], self.stats[ii], self.rY[:, ii], self.rT[:, ii] = \
                _pcm_test(Y, X, Z, reg_yonxz, reg_yonz, reg_yhatonz, reg_vonxz,
                          reg_ronz, estimate_variance, test_split, max_exp, rng)
        self.stat = np.mean(self.stats)
        self.pval = 1 - norm().cdf(self.stat)
        self.summary(digits=summary_digits)

    def summary(self, digits=3):
        print("\tProjected covariance measure test")
        print(
            f'Z = {self.stat:.{digits}f}, p-value = {self.pval:.{digits}f}')
        print(
            "alternative hypothesis: true E[Y | X, Z] is not equal to E[Y | Z]")

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        for idx in range(self.rT.shape[1]):
            ax.scatter(x=self.rT[:, idx], y=self.rY[:, idx])
        ax.set_xlabel("Residuals f(X, Z) | Z")
        ax.set_ylabel("Residuals Y | Z")
        return fig, ax


def _pcm_test(Y, X, Z,
              reg_yonxz: RegressionMethod = RF(),
              reg_yonz: RegressionMethod = RF(),
              reg_yhatonz: RegressionMethod = RF(),
              reg_vonxz: RegressionMethod = RF(),
              reg_ronz: RegressionMethod = RF(),
              estimate_variance=True,
              test_split=0.5,
              max_exp=5,
              rng=np.random.default_rng()):
    """
    TODO
    """

    # sample splitting
    Y, X, Z = _data_check(Y, X, Z)
    Ytr, Xtr, Ztr, Yte, Xte, Zte = _split_sample(Y, X, Z, test_split, rng)

    # regression on the training data (direction estimate)
    XZtr = np.column_stack([Xtr, Ztr])
    reg_yonxz.fit(Y=Ytr, X=XZtr)
    yhat = reg_yonxz.predict(X=XZtr)
    reg_yhatonz.fit(Y=yhat, X=Ztr)
    rho = np.mean((Ytr - reg_yhatonz.predict(X=Ztr)) * yhat)

    def hhat(X, Z):
        htilde = reg_yonxz.predict(
            np.column_stack([X, Z])) - reg_yhatonz.predict(Z)
        return np.sign(rho) * htilde

    # estimate variance
    if estimate_variance:
        sqr = (Ytr - yhat)**2
        reg_vonxz.fit(Y=sqr, X=XZtr)

        def a(c):
            den = np.column_stack([reg_vonxz.predict(XZtr),
                                   np.repeat(0, XZtr.shape[0])])
            return np.mean(sqr / (np.max(den, axis=1) + c)) - 1

        if (a(0) < 0):
            chat = 0
        else:
            lwr, upr = 0, 10
            counter = 0
            while np.sign(a(lwr)) * np.sign(a(upr)) == 1:
                upr += 5
                counter += 1
                if counter > max_exp:
                    raise ValueError(
                        "Cannot compute variance estimate, try rerunning with `estimate_variance=False`.")
            chat = root_scalar(a, method="brentq", bracket=[lwr, upr])

        def vhat(X, Z):
            XZ = np.column_stack([X, Z])
            vtemp = np.max(np.column_stack([reg_vonxz.predict(XZ),
                                            np.repeat(0, XZ.shape[0])]), axis=1)
            return vtemp + chat
    else:
        def vhat(X, Z): return 1

    # regression on the test data
    def fhat(X, Z): return hhat(X, Z) / vhat(X, Z)
    fhats = fhat(Xte, Zte)
    reg_ronz.fit(Y=fhats, X=Zte)
    reg_yonz.fit(Y=Yte, X=Zte)

    # test
    rY = Yte - reg_yonz.predict(X=Zte)
    rT = fhats - reg_ronz.predict(X=Zte)
    L = rY * rT
    stat = np.sqrt(Yte.shape[0]) * np.mean(L) / \
        np.sqrt(np.mean(L**2) - np.mean(L)**2)
    if np.isnan(stat):
        stat = -np.Inf
    pval = 1 - norm().cdf(stat)

    return pval, stat, rY, rT
