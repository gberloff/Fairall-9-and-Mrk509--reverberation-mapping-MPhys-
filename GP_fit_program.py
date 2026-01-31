# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest, shapiro, kstest, anderson
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct,
    WhiteKernel, ConstantKernel as C, PairwiseKernel
)

csv_path = "F9LCs.csv"
BAND = "W2"
MJD_COL, FLUX_COL, ERR_COL = "MJD", "Flux", "Error"
RQ_LENGTH_SCALE = 10
RQ_ALPHA = 1.0
N_SAMPLES = 1000
NU = 0.5


df = pd.read_csv(csv_path)
first_col = df.columns[0]
mask = df[first_col].astype(str).str.strip().str.upper().eq(BAND)
vf = df.loc[mask, [MJD_COL, FLUX_COL, ERR_COL]].dropna().sort_values(MJD_COL)

y_mu, y_sd = vf[FLUX_COL].mean(), vf[FLUX_COL].std()
y_norm     = (vf[FLUX_COL].to_numpy() - y_mu) / y_sd
yerr_norm  = vf[ERR_COL].to_numpy() / y_sd
X = vf[MJD_COL].to_numpy().reshape(-1, 1)

t_lo, t_hi = np.quantile(X, [0.025, 0.975])
keep = (X[:, 0] >= t_lo) & (X[:, 0] <= t_hi)
X = X[keep]
y_norm = y_norm[keep]
yerr_norm = yerr_norm[keep]

t_grid = np.arange(np.floor(X.min()), np.ceil(X.max()) + 1, 1.0).reshape(-1, 1)


kernel = C(1.0, (1.0, 1.0)) * RationalQuadratic(length_scale=RQ_LENGTH_SCALE,
                                     length_scale_bounds=(0.00001, 10), alpha=RQ_ALPHA, alpha_bounds=(0.0001, 10))

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=yerr_norm**3,     
    normalize_y=False,
    random_state=42
)

gp.fit(X, y_norm)
print("Kernel (fixed):", gp.kernel_)
lml  = gp.log_marginal_likelihood(theta=gp.kernel_.theta)
nlml = -lml
print(f"log ML = {lml:.6f}")
print(f"NLML = {nlml:.6f}")

y_pred, y_std = gp.predict(t_grid, return_std=True)
samples = gp.sample_y(t_grid, n_samples=N_SAMPLES, random_state=42)
last_curve = samples[:, -1]
lo = np.percentile(samples, 2.5, axis=1)
hi = np.percentile(samples, 97.5, axis=1)

y_fit_at_data = gp.predict(X, return_std=False)
residuals = y_norm - y_fit_at_data
resid_std = (residuals - residuals.mean()) / residuals.std()


plt.figure(figsize=(10, 5))
plt.errorbar(X.ravel(), y_norm, yerr=yerr_norm, fmt='o', ms=3, alpha=0.35,
             color='red', label=f'{BAND}-band')
plt.fill_between(t_grid.ravel(), lo, hi, color='grey', alpha=0.25, label='95%')
plt.plot(t_grid.ravel(), y_pred, 'k-', lw=1.8, label='GP mean')
plt.plot(t_grid.ravel(), last_curve, ls='--', lw=1.0, color='purple',
         label='Last of 1000 samples')
plt.title(f"3C-273 {BAND}-band GPR (RationalQuadratic, ℓ={RQ_LENGTH_SCALE:g} d, NLML={nlml})")
plt.xlabel("MJD (days)")
plt.ylabel("Flux")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()




