# %%
import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import cholesky
from pycomets.gcm import GCM
from pycomets.pcm import PCM
from pycomets.regression import LM, RF
import matplotlib.pyplot as plt
import pandas as pd

def ecdf(arr):
    arr_sorted = np.sort(arr)
    n = len(arr)
    return arr_sorted, np.arange(1, n+1)/n

# %%
# Generate data (numpy version)

def generate_obs_data(
    V,
    W,
    H,
    beta,  # coef instrument to treatment
    alpha1,  # coef H to treatment
    alpha2,  # coef H to response
    theta,  # coef treatment to response
):
    """
    Generate observed instrument data.
    :param Beta: Linear mixing parameters mapping VW to Z.
    :param V: Invalid component in Z.
    :param W: Valid component in Z.
    :param H: Unobserved confounder between V, D, and Y.
    """

    # Currently assuming V and H are of the same dimension
    V += H
    VW = np.concat((V, W), axis=1)
    # print(f"VW shape: {VW.shape}, Beta shape: {Beta.shape}, Z shape: {Z.shape}")

    # generate treatment and response
    D = VW @ beta + H @ alpha1 + np.random.normal(size=(V.shape[0], 1))
    Y = theta * D + H @ alpha2 + np.random.normal(size=(V.shape[0], 1))

    return VW, D, Y


Sig_v = toeplitz(np.array([2,1])/2)
Sig_w = toeplitz(np.array([3,2,1])/3)
L_v = cholesky(Sig_v)
L_w = cholesky(Sig_w)
beta = np.array([0.5, 1.0, 0.5, 1.0, 0.5])
alpha1 = np.array([1.0, 1.5])
alpha2 = np.array([1.5, 1.0])
theta = 1.0

num_obs = 200
num_rep = 50
gcm_pvals = []
for ii in range(num_rep):
    H = np.random.normal(loc=1.0, size=(num_obs, 2))
    V = np.random.normal(size=(num_obs, 2)) @ L_v.T
    W = np.random.normal(size=(num_obs, 3)) @ L_w.T
    VW, D, Y = generate_obs_data(V,
                W,
                H,
                beta,  # coef instrument to treatment
                alpha1,  # coef H to treatment
                alpha2,  # coef H to response
                theta)

    gcm = GCM()
    gcm.test(Y = Y, 
            X = np.concat((V,W), axis=1),
            Z = np.concat((H,D), axis=1),
            reg_xz=LM(),
            reg_yz=LM())
    gcm_pvals.append(gcm.pval)

gcm_pvals = np.array(gcm_pvals)
x, y = ecdf(gcm_pvals)
plt.scatter(x, y, s=5)
plt.axline((0,0), slope=1)

# %% 
# Load data

gcm_pvals = []
for ii in range(100):
    df = pd.read_csv(f"tmp_simulated_data/sim{ii}.csv")
    gcm = GCM()
    gcm.test(Y = df['Y'].to_numpy(), 
             X = np.column_stack((df.filter(regex="^V"),
                                  df.filter(regex="^W"))),
             Z = np.column_stack((df.filter(regex="^H"), 
                                  df['D'])),
             reg_xz=LM(),
             reg_yz=LM(), 
             test_type="quadratic",
             show_summary=False)
    gcm_pvals.append(gcm.pval)

gcm_pvals = np.array(gcm_pvals)
x, y = ecdf(gcm_pvals)
plt.scatter(x, y, s=5)
plt.axline((0,0), slope=1)
# %%
