# %%
# libs
# ##
import numpy as np
from gcm import GCM, WGCM
from pcm import PCM
from regression import DefaultMultiRegression, LM, RF, RFC, CoxPH, KRR
from sksurv.datasets import load_whas500

# %%
# DGP: regression
# ##

n = 300
rng = np.random.default_rng(1)
X = rng.normal(0, 1, (n, 2))
Z = rng.normal(0, 1, (n, 2))
Y = X[:, 0]**2 + Z[:, 1] + rng.normal(0, 1, n)

# %%
# Y indep X | Z
# ##

gcm = GCM()
gcm.test(Y, X, Z, LM(), LM())
fig, ax = gcm.plot()

gcm = GCM()
gcm.test(Y, X[:, 0], Z[:, 0], LM(), LM())
fig, ax = gcm.plot()

gcm = GCM()
gcm.test(Y, X, Z, LM(), None, RF(random_state=1))
fig, ax = gcm.plot()

gcm = GCM()
gcm.test(Y, X, Z,
         KRR(kernel="rbf", param_grid={'alpha': [0.1, 1]}),
         KRR(param_grid={'kernel': ('linear', 'rbf'), 'alpha': [0.1, 1]}))

pcm = PCM()
pcm.test(Y, X, Z, rep=3, rng=rng)
fig, ax = pcm.plot()

pcm = PCM()
pcm.test(Y, X[:, 0], Z[:, 0], rep=3, rng=rng)
fig, ax = pcm.plot()

# %%
# Y indep X | Z
# ##

wgcm = WGCM()
wgcm.test(Y, X, Z, RF(random_state=1), RF(random_state=1), rng=rng)
fig, ax = wgcm.plot()

wgcm = WGCM()
wgcm.test(Y, X[:, 0], Z[:, 0], RF(), RF(), rng=rng)
fig, ax = wgcm.plot()

# %%
# DGP: classification
# ##

n = 300
rng = np.random.default_rng(1)
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
# Multiregression
# ##

mod = DefaultMultiRegression(LM(), X.shape[1])
mod.fit(Y=X, X=Z)
mod.predict(X=Z)
mod.residuals(Y=X, X=Z)

# %%
# Surv example
# ##

X, y = load_whas500()
X = X.astype(float)
gcm = GCM()
gcm.test(y, X.iloc[:, 0].to_numpy(), X.iloc[:, 1].to_numpy(), CoxPH(), LM())

# %%
