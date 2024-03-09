# %%
# libs
# ##
import numpy as np
from gcm import compute_stat
from sklearn.linear_model import LinearRegression

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

stat, df = compute_stat(rY, rX)
