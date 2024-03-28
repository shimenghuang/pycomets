# %%
# libs
# ##
import numpy as np
from gcm import GCM
from regression import LM, RF, RFC

# %%
# DGP: regression
# ##

n = 300
rng = np.random.default_rng()
X = rng.normal(0, 1, (n, 2))
Z = rng.normal(0, 1, (n, 2))
Y = X[:, 0]**2 + Z[:, 1] + rng.normal(0, 1, n)

# %%
# Y indep X | Z with linear regression
# ##

gcm = GCM()
gcm.test(Y, X, Z, LM(), LM())
gcm.summary(digits=2)
fig, ax = gcm.plot()

# %%
# DGP: classification
# ##

n = 300
rng = np.random.default_rng()
X = rng.binomial(1, 0.5, n * 2).reshape(n, 2)
Z = rng.normal(0, 1, (n, 2))
Y = X[:, 0]**2 + Z[:, 1] + rng.normal(0, 1, n)

# %%
# Y indep X | Z with linear regression
# ##

gcm = GCM()
gcm.test(Y, X, Z, RF(), RFC())
gcm.summary(digits=2)
fig, ax = gcm.plot()

# %%
