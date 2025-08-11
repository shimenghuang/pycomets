# %%
# libs
# ##
import numpy as np
from pycomets.gcm import GCM, WGCM, KGCM
from pycomets.pcm import PCM
from pycomets.regression import DefaultMultiRegression, LM, RF, RFC, CoxPH, KRR, XGB, XGBC
from sksurv.datasets import load_whas500

# %%
# DGP 1: linear SCM with X <- Z -> Y
# ##

n = 300
rng = np.random.default_rng(1)
Z = rng.normal(0, 1, (n, 2))
X = Z @ np.array([0.5, 1.0]) + rng.normal(0, 1, n)
Y = Z[:, 1] + rng.normal(0, 1, n)
print(f'Y shape {Y.shape}, X shape {X.shape}, Z shape {Z.shape}')

gcm = GCM()
gcm.test(Y, X, Z, LM(), LM())
fig, ax = gcm.plot()

gcm = GCM()
gcm.test(Y, X, Z, LM(), RF(random_state=1, param_grid={'max_depth': [2, 5]}))
fig, ax = gcm.plot()

gcm = GCM()
gcm.test(Y, X, Z, KRR(), KRR())
fig, ax = gcm.plot()

gcm = GCM()
gcm.test(Y, X, Z,
         KRR(kernel="rbf", param_grid={'alpha': [0.1, 1, 10]}),
         KRR(param_grid={'kernel': ('linear', 'rbf'), 'alpha': [0.1, 1]}))
fig, ax = gcm.plot()

gcm = GCM()
gcm.test(Y, X, Z, XGB(), XGB())
fig, ax = gcm.plot()

gcm = GCM()
gcm.test(Y, X, Z,
         XGB(param_grid={'n_estimators': [10, 100], 'max_depth': [2, 5]}),
         XGB(param_grid={'n_estimators': [10, 100], 'max_depth': [2, 5]}))
fig, ax = gcm.plot()

pcm = PCM()
pcm.test(Y, X, Z, rep=3, rng=rng)
fig, ax = pcm.plot()

wgcm = WGCM()
wgcm.test(Y, X, Z, RF(random_state=1), RF(random_state=1), rng=rng)
fig, ax = wgcm.plot()

kgcm = KGCM()
kgcm.test(Y, X, Z, LM(), LM(), rng=rng)
fig, ax = kgcm.plot()

# %%
# DGP 2: regression X -> Z <- Y
# ##

n = 300
rng = np.random.default_rng(1)
X = rng.normal(0, 1, (n, 2))
Y = rng.normal(0, 1, (n, 2)) # 2-dimensional Y
Z = X @ np.array([0.5, 1.0]) + Y @ np.array([1.0, 0.5]) + rng.normal(0, 1, n)
print(f'Y shape {Y.shape}, X shape {X.shape}, Z shape {Z.shape}')

gcm = GCM()
gcm.test(Y, X, Z, LM(), LM())
fig, ax = gcm.plot()

wgcm = WGCM()
wgcm.test(Y, X, Z, RF(random_state=1), RF(random_state=1), rng=rng)
fig, ax = wgcm.plot()


# %%
# DGP: classification X -> Y <- Z
# ##

n = 300
rng = np.random.default_rng(1)
X = rng.binomial(1, 0.5, n * 2).reshape(n, 2)
Z = rng.normal(0, 1, (n, 2))
Y = X[:, 0]**2 + Z[:, 1] + rng.normal(0, 1, n)
print(f'Y shape {Y.shape}, X shape {X.shape}, Z shape {Z.shape}')

gcm = GCM()
gcm.test(Y, X, Z, RF(), RFC())
gcm.summary(digits=2)
fig, ax = gcm.plot()

gcm = GCM()
gcm.test(Y, X, Z,
         XGB(param_grid={'n_estimators': [10, 100], 'max_depth': [2, 5]}),
         XGBC(param_grid={'n_estimators': [10, 100], 'max_depth': [2, 5]}))
fig, ax = gcm.plot()

pcm = PCM()
pcm.test(Y, X, Z, rep=3, rng=rng)
fig, ax = pcm.plot()

# %%
# Surv example
# ##

X, y = load_whas500()
X = X.astype(float)
gcm = GCM()
gcm.test(y, X.iloc[:, 0].to_numpy(), X.iloc[:, 1].to_numpy(), CoxPH(), LM())

# %%
