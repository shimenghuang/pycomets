# %%
# libs
# ##
import numpy as np
from sklearn.linear_model import LinearRegression
from pycomets.gcm import _gcm_test
from pycomets.regression import LM


# %%
# DGP
# ##

n = 300
rng = np.random.default_rng()
X = rng.normal(0, 1, (n, 2))
Z = rng.normal(0, 1, (n, 2))
Y = X[:, 0]**2 + Z[:, 1] + rng.normal(0, 1, n)

# %%
# Y indep X | Z with linear regression
# ##

reg_YonZ = LinearRegression().fit(X=Z, y=Y)
rY = Y - reg_YonZ.predict(X=Z)

reg_XonZ = LinearRegression().fit(X=Z, y=X)
rX = X - reg_XonZ.predict(X=Z)

pval, stat, df = _gcm_test(rY, rX)

# %%
mod = LM(fit_intercept=True, copy_X=True)
mod.fit(Y=Y, X=X)
mod.residuals(Y, X)

# %%
