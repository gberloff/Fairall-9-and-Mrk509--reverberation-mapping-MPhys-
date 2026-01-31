# -*- coding: utf-8 -*-


from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # allow OpenMP duplication if needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# User settings
# ---------------------------------------------------------
QSO       = "Fairall 9"                # change per object
LAG_FILE  = Path(f"TimelagsCorr.csv")
X_USER    = 4.97     # your chosen X
LAM0_SCALE = 1928.0                 # fallback λ_ref

# χ² grid ranges (for comparison)
M8_vals   = np.linspace(0.5, 5, 150)
mdot_vals = np.linspace(0.0035, 1.0, 150)

# GP training box & sample size
n_train = 500                       # GP scales ~N^3; keep moderate
M8_min_train, M8_max_train     = 0.5, 5
mdot_min_train, mdot_max_train = 0.0035, 1.0

rng = np.random.default_rng(12345)

# =========================================================
# 1. Load lag table
# =========================================================
df = pd.read_csv(LAG_FILE)

def find_col(sub: str) -> str:
    """Find first column whose name contains `sub` (case-insensitive)."""
    for c in df.columns:
        if sub.lower() in c.lower():
            return c
    raise KeyError(f"Could not find a column containing '{sub}' in {list(df.columns)}")

col_wave = find_col("wavel")
col_lag  = find_col("lag")
col_err  = find_col("err")

lam_all  = df[col_wave].to_numpy(float)
tau_all  = df[col_lag].to_numpy(float)
err_all  = df[col_err].to_numpy(float)

print("Loaded time-lag table:\n", df)

# =========================================================
# 2. Choose reference wavelength λ_ref (τ=0 row if present)
# =========================================================
zero_mask = (
    np.isfinite(lam_all) &
    np.isfinite(tau_all) &
    np.isclose(tau_all, 0.0, atol=1e-6)
)

if np.any(zero_mask):
    lam_ref = lam_all[zero_mask].max()
    print(f"\nUsing λ_ref = {lam_ref:.1f} Å as zero-lag reference.")
else:
    lam_ref = LAM0_SCALE
    print(f"\nNo explicit zero-lag row found; using λ_ref = {lam_ref:.1f} Å.")

# =========================================================
# 3. Select usable points
# =========================================================
mask_fit = (
    np.isfinite(lam_all) &
    np.isfinite(tau_all) &
    np.isfinite(err_all) &
    (err_all > 0.0) &
    (lam_all >= 0.0001)
)

lam_A     = lam_all[mask_fit]
tau_obs   = tau_all[mask_fit]
sigma_tau = err_all[mask_fit]

print("\nWavelengths used in fit (Å):", lam_A)
print("Observed lags τ (days):      ", tau_obs)
print("Lag errors σ_τ (days):       ", sigma_tau)

n_lags = lam_A.size

# =========================================================
# 4. Thin-disc model: RELATIVE lags τ(λ) − τ(λ_ref)
# =========================================================
def tau_model_rel(lam_angstrom: np.ndarray,
                  M8: float,
                  mdot: float,
                  X: float,
                  lam_ref_val: float) -> np.ndarray:
    """Relative lag: τ(λ) − τ(λ_ref). Negative for λ < λ_ref."""
    lam = np.array(lam_angstrom, dtype=float)
    ratio = lam / lam_ref_val
    base = X * (M8**(2.0/3.0)) * (mdot**(1.0/3.0))
    return base * (ratio**(4.0/3.0) - 1.0)

# =========================================================
# 5. χ² grid over (M8, mdot)
# =========================================================
chi2 = np.zeros((M8_vals.size, mdot_vals.size), dtype=float)

for i, M8 in enumerate(M8_vals):
    for j, mdot in enumerate(mdot_vals):
        tau_mod = tau_model_rel(lam_A, M8, mdot, X_USER, lam_ref)
        chi2[i, j] = np.sum(((tau_obs - tau_mod) / sigma_tau) ** 2)

chi2_min = np.nanmin(chi2)
i_min, j_min = np.unravel_index(np.nanargmin(chi2), chi2.shape)
M8_best   = M8_vals[i_min]
mdot_best = mdot_vals[j_min]
MBH_best  = M8_best * 1e8

print(f"\nχ² grid (X={X_USER:.2f}):")
print(f"  Best-fit M8   = {M8_best:.3f}")
print(f"  Best-fit mdot = {mdot_best:.3f}")
print(f"  Best-fit MBH  ≈ {MBH_best:.3e} Msun")
print(f"  χ²_min        = {chi2_min:.2f}")

dchi2  = chi2 - chi2_min
levels = [5, 10, 20, 40, 80]

fig, ax = plt.subplots(figsize=(6, 5))
cs = ax.contour(
    mdot_vals,
    M8_vals * 1e8,
    dchi2,
    levels=levels,
    colors=["C0", "C1", "C2", "C3", "C4"],
)
ax.clabel(cs, inline=True, fontsize=8, fmt=r"$\Delta\chi^2=%.0f$")
ax.set_xlabel(r"Eddington ratio $\dot m$")
ax.set_ylabel(r"Black hole mass $M_{\rm BH}\,[M_\odot]$")
ax.set_title(
    rf"$\Delta\chi^2$ contours (X={X_USER:.2f}, "
    rf"relative lags, $\lambda_\mathrm{{ref}}={lam_ref:.0f}\,\AA$)"
)
ax.grid(True, ls="--", alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# 6. Build synthetic training set for GP
# =========================================================
print(f"\n[GP inversion] Using X = {X_USER:.2f}")

M8_train   = rng.uniform(M8_min_train,   M8_max_train,   size=n_train)
mdot_train = rng.uniform(mdot_min_train, mdot_max_train, size=n_train)

lags_train = []
for M8_val, mdot_val in zip(M8_train, mdot_train):
    tau_mod = tau_model_rel(lam_A, M8_val, mdot_val, X_USER, lam_ref)
    noise   = rng.normal(0.0, 0.05, size=tau_mod.shape)
    lags_train.append(tau_mod + noise)

lags_train = np.vstack(lags_train)       # (n_train, n_lags)

# Targets: log10(M8), log10(mdot)
y_train = np.vstack([
    np.log10(M8_train),
    np.log10(mdot_train),
]).T                                      # (n_train, 2)

# ---------------------------------------------------------
# Scale inputs & outputs for GP
# ---------------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(lags_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# ---------------------------------------------------------
# Define GPs for each output dimension
# ---------------------------------------------------------
kernel_base = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(
    length_scale=np.ones(n_lags),
    length_scale_bounds=(1e-2, 1e2),
    nu=1.5,
)
kernel = kernel_base + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))

gp_M8   = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=3, normalize_y=False)
gp_mdot = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=3, normalize_y=False)

print("[GP inversion] Fitting GP for log10(M8)...")
gp_M8.fit(X_train_scaled, y_train_scaled[:, 0])
print("[GP inversion] Fitting GP for log10(mdot)...")
gp_mdot.fit(X_train_scaled, y_train_scaled[:, 1])
print("[GP inversion] GP training complete.")

# =========================================================
# 7. Predict M8, mdot for observed lags (mean + GP std)
# =========================================================
X_star = scaler_X.transform(tau_obs.reshape(1, -1))

y_M8_scaled_mean,  y_M8_scaled_std  = gp_M8.predict(X_star, return_std=True)
y_mdot_scaled_mean, y_mdot_scaled_std = gp_mdot.predict(X_star, return_std=True)

# De-scale
y_pred_mean = scaler_y.inverse_transform(
    np.column_stack([y_M8_scaled_mean, y_mdot_scaled_mean])
)[0]

logM8_pred_mean, logmdot_pred_mean = y_pred_mean

M8_pred   = 10.0**logM8_pred_mean
mdot_pred = 10.0**logmdot_pred_mean

M8_pred   = np.clip(M8_pred,   M8_min_train,   M8_max_train)
mdot_pred = np.clip(mdot_pred, mdot_min_train, mdot_max_train)
MBH_pred  = M8_pred * 1e8

print(f"\n[GP inversion] Mean prediction (no lag MC yet):")
print(f"  log10(M8)   = {logM8_pred_mean:.3f}")
print(f"  log10(mdot) = {logmdot_pred_mean:.3f}")
print(f"  M8          = {M8_pred:.3f}")
print(f"  mdot        = {mdot_pred:.3f}")
print(f"  MBH         ≈ {MBH_pred:.3e} Msun")

# =========================================================
# 8. Monte-Carlo over lag errors through the GP
# =========================================================
n_mc = 2000

MBH_samples  = []
mdot_samples = []

for _ in range(n_mc):
    # sample lags within observational errors
    tau_mc = rng.normal(tau_obs, sigma_tau).reshape(1, -1)
    X_mc   = scaler_X.transform(tau_mc)

    y_M8_sc, _   = gp_M8.predict(X_mc,   return_std=True)
    y_mdot_sc, _ = gp_mdot.predict(X_mc, return_std=True)

    y_mc = scaler_y.inverse_transform(
        np.column_stack([y_M8_sc, y_mdot_sc])
    )[0]

    logM8_mc, logmdot_mc = y_mc
    M8_mc   = 10.0**logM8_mc
    mdot_mc = 10.0**logmdot_mc

    M8_mc   = np.clip(M8_mc,   M8_min_train,   M8_max_train)
    mdot_mc = np.clip(mdot_mc, mdot_min_train, mdot_max_train)

    MBH_samples.append(M8_mc * 1e8)
    mdot_samples.append(mdot_mc)

MBH_samples  = np.array(MBH_samples)
mdot_samples = np.array(mdot_samples)

MBH_med  = np.median(MBH_samples)
MBH_low  = np.percentile(MBH_samples, 16)
MBH_high = np.percentile(MBH_samples, 84)

mdot_med  = np.median(mdot_samples)
mdot_low  = np.percentile(mdot_samples, 16)
mdot_high = np.percentile(mdot_samples, 84)

print("\n[GP + MC uncertainties] (lag errors folded through GP)")
print(f"  MBH  median ≈ {MBH_med:.3e} Msun")
print(f"        16–84% : {MBH_low:.3e} – {MBH_high:.3e} Msun")
print(f"  mdot median ≈ {mdot_med:.3f}")
print(f"        16–84% : {mdot_low:.3f} – {mdot_high:.3f}")

# =========================================================
# 9. χ² at GP mean prediction & plot on contours
# =========================================================
tau_gp = tau_model_rel(lam_A, M8_pred, mdot_pred, X_USER, lam_ref)
chi2_gp = np.sum(((tau_obs - tau_gp) / sigma_tau) ** 2)

print(f"\nχ² at GP mean prediction: {chi2_gp:.2f}  (χ²_min = {chi2_min:.2f})")
print(f"Δχ²(GP vs best-fit) = {chi2_gp - chi2_min:.2f}")

fig, ax = plt.subplots(figsize=(6, 5))
cs = ax.contour(
    mdot_vals,
    M8_vals * 1e8,
    dchi2,
    levels=levels,
    colors=["C0", "C1", "C2", "C3", "C4"],
)
ax.clabel(cs, inline=True, fontsize=8, fmt=r"$\Delta\chi^2=%.0f$")

# --- NEW: plot median with 16–84% error bars ---
x_center = mdot_med
y_center = MBH_med

xerr = np.array([[mdot_med - mdot_low], [mdot_high - mdot_med]])
yerr = np.array([[MBH_med - MBH_low], [MBH_high - MBH_med]])

ax.errorbar(
    x_center,
    y_center,
    xerr=xerr,
    yerr=yerr,
    fmt="X",
    markersize=8,
    color="magenta",
    ecolor="magenta",
    elinewidth=1.5,
    capsize=3,
    label="GP median prediction (16–84% CI)",
)

text = (
    rf"$M_{{BH}} = {MBH_med/1e8:.2f}\times 10^8\,M_\odot$"
    + "\n"
    + rf"$\dot{{m}} = {mdot_med:.3f}$"
    + "\n"
    + r"16–84% CIs from GP+MC"
)

ax.annotate(
    text,
    xy=(x_center, y_center),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=10,
    color="magenta",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="magenta", alpha=0.75),
)

ax.set_xlabel(r"Eddington ratio $\dot m$")
ax.set_ylabel(r"Black hole mass $M_{\rm BH}\,[M_\odot]$")
ax.set_title(f"{QSO} BH mass (Gaussian Process model, X={X_USER:.2f})")
ax.grid(True, ls="--", alpha=0.3)
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(f"chi2_with_GP_X{X_USER:.2f}.png", dpi=300)
plt.show()