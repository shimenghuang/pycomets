import numpy as np
import scipy.stats as sp
from .utils import _cov_to_cor


def _coin_test(
    rY,
    rX,
    alternative="two.sided",
    type="max",
    distribution="asymptotic",
    rng=np.random.default_rng(),
    B=499,
):
    """
    Computation of the permutation test.

    Parameters
    ----------
    rY : array of shape (n_sample, n_feature_y)
        Residuals of the Y on Z regression.

    rX : array of shape (n_sample, n_feature_x)
        Residuals of the X on Z regression.

    alternative: str
        One of "two.sided", "less", or "greater".

    type : str
        One of "quadratic" or "max".

    distribution : str
        One of "asymptotic" or "approximate". Only applies if `type = "max"` is used.

    B : int
        Number of bootstrap samples. Only applies if `type = "max"` is used.
    """

    if alternative not in ["two.sided", "less", "greater"]:
        raise Warning('alternative needs to be one of "quadratic" or "max".')

    if type not in ["quadratic", "max"]:
        raise Warning('type needs to be one of "quadratic" or "max".')

    if distribution not in ["asymptotic", "approximate"]:
        raise Warning('distribution needs to be one of "asymptotic" or "approximate".')

    n = rY.shape[0]
    dim_rX = 1 if rX.ndim == 1 else rX.shape[1]
    dim_rY = 1 if rY.ndim == 1 else rY.shape[1]
    rY_cm = np.mean(rY, axis=0)
    rX_cs = np.sum(rX, axis=0)
    mu = np.outer(rX_cs, rY_cm).reshape(-1, order="F")
    rY_center = rY - rY_cm
    rY_cv = 1 / n * rY_center.T @ rY_center

    rX_sp = rX.T @ rX
    rX_ps = np.outer(rX_cs, rX_cs)
    sig = n / (n - 1) * np.kron(rY_cv, rX_sp) - 1 / (n - 1) * np.kron(rY_cv, rX_ps)

    t_obs = (rX.T @ rY).reshape(-1)
    if type == "max":

        def _cal_std_ts(tt):
            # vals, vcts = np.linalg.eig(sig)
            # sig_neg_half = vcts * np.sqrt(1/vals) @ np.linalg.inv(vcts)
            # stat = np.max(np.abs(sig_neg_half @ (tt - mu)))
            stat = (tt - mu) / np.sqrt(np.diag(sig))
            return stat

        std_t_obs = _cal_std_ts(t_obs)
        if distribution == "asymptotic":
            # sim = rng.multivariate_normal(size=B, mean=mu, cov=sig)
            # print(sim.shape)
            # stat_sim = np.apply_along_axis(_cal_stat, axis=1, arr=sim)
            # pval = (np.sum(stat >= stat_sim) + 1) / (B + 1)
            cor = _cov_to_cor(sig)
            if alternative == "two.sided":
                stat = np.max(np.abs(std_t_obs))
                pval = 1 - sp.multivariate_normal(cov=cor).cdf(
                    np.repeat(np.abs(stat), dim_rX * dim_rY),
                    lower_limit=np.repeat(-np.abs(stat), dim_rX * dim_rY),
                )
            elif alternative == "greater":
                stat = np.max(std_t_obs)
                pval = 1 - sp.multivariate_normal(cov=cor).cdf(
                    np.repeat(stat, dim_rX * dim_rY)
                )
            else:
                stat = np.min(std_t_obs)
                pval = 1 - sp.multivariate_normal(cov=cor).cdf(
                    np.repeat(np.inf, dim_rX * dim_rY),
                    lower_limit=np.repeat(stat, dim_rX * dim_rY),
                )

        else:
            pval = np.nan  # TODO (code in C may be avaiable)!!
    elif type == "quadratic":
        sig_pinv = np.linalg.pinv(sig)
        stat = (t_obs - mu).T @ sig_pinv @ (t_obs - mu)
        pval = 1 - sp.chi2(dim_rX * dim_rY).cdf(stat)

    return pval, stat, mu, sig
