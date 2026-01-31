# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 13:37:18 2025

@author: gberl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    RationalQuadratic,
    ConstantKernel as C,
)


# Data related constants
CSV_PATH = "F9LCs.csv"
BAND = "V"
SAMPLES_CSV = "gp_samples_advanced_V"


# Model hyperparameters
LENGTH_SCALE = 10.0
ALPHA = 10.0
N_SAMPLES = 1000
RANDOM_STATE = 42


def load_and_preprocess_data(csv_path, band):
    """Load and preprocess the data."""
    df = pd.read_csv(csv_path)
    first_col = df.columns[0]
    mask = df[first_col].astype(str).str.strip().str.upper().eq(band)
    vf = df.loc[mask, ["MJD", "Flux", "Error"]].dropna().sort_values("MJD")
 
    y_mu, y_sd = vf["Flux"].mean(), vf["Flux"].std()
    y_norm = (vf["Flux"].to_numpy() - y_mu) / y_sd
    yerr_norm = vf["Error"].to_numpy() / y_sd
    X = vf["MJD"].to_numpy().reshape(-1, 1)
 
    t_lo, t_hi = np.quantile(X, [0.025, 0.975])
    keep = (X[:, 0] >= t_lo) & (X[:, 0] <= t_hi)
    X = X[keep]
    y_norm = y_norm[keep]
    yerr_norm = yerr_norm[keep]


    return X, y_norm, yerr_norm


def fit_gaussian_process(X, y_norm, yerr_norm):
    """Fit the Gaussian Process model."""
    kernel = C(1.0, (1.0, 1.0)) * Matern(
        length_scale=LENGTH_SCALE, length_scale_bounds=(0.0000001, 10000000), nu=0.5
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=yerr_norm**3, normalize_y=False, random_state=RANDOM_STATE
    )
    gp.fit(X, y_norm)
    print(gp.kernel_)
    return gp


def generate_time_grid(X):
    """Generate a time grid for predictions."""
    t_min, t_max = X.min(), X.max()
    t_grid = np.arange(t_min, t_max+1, 1).reshape(-1, 1)
    return t_grid


def generate_samples(gp, t_grid):
    """Generate samples from the Gaussian Process."""
    samples = gp.sample_y(t_grid, n_samples=N_SAMPLES, random_state=RANDOM_STATE)
    return samples


if __name__ == "__main__":
    X, y_norm, yerr_norm = load_and_preprocess_data(CSV_PATH, BAND)
    gp = fit_gaussian_process(X, y_norm, yerr_norm)
    t_grid = generate_time_grid(X)
    samples = generate_samples(gp, t_grid)


    # save samples to CSV: first column = time (MJD), following columns = sample_0 ... sample_{N-1}
    df_samples = pd.DataFrame(samples, index=None, columns=[f"sample_{i}" for i in range(samples.shape[1])])
    df_samples.insert(0, "MJD", t_grid.ravel())
    df_samples.to_csv(SAMPLES_CSV, index=False)
 