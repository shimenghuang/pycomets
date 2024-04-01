# %%
# libs
# ##
import numpy as np
from gcm import GCM
from pcm import PCM
from regression import LM, RF, RFC, CoxPH
from sksurv.datasets import load_whas500

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
fig, ax = gcm.plot()

pcm = PCM()
pcm.test(Y, X, Z, rep=3)
fig, ax = pcm.plot()

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
# Surv example
# ##

X, y = load_whas500()
X = X.astype(float)
gcm = GCM()
gcm.test(y, X.iloc[:, 0].to_numpy(), X.iloc[:, 1].to_numpy(), CoxPH(), LM())

# %%
