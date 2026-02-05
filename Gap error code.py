# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 16:17:34 2026

@author: gberl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel



CSV_V  = Path("gp_samples_V.csv")
CSV_W2 = Path("gp_samples_W2.csv")


N_SEGMENTS = 15
APPLY_DETREND = True
APPLY_HANN = True


GAP_SIZE = 10
MAX_TOTAL_REMOVED = 300  


TARGET_LAG_MIN_DAYS = 2.5
TARGET_LAG_MAX_DAYS = 3.5
MAX_PAIR_TRIES = 5000


N_POST_SAMPLES = 300  


SIGMA_STOP = 1.0


RNG_SEED = None  


OUT_DIR = Path("gap_removed_lightcurves_singlepair")
OUT_DIR.mkdir(parents=True, exist_ok=True)




def _linear_detrend(y: np.ndarray) -> np.ndarray:
    n = y.size
    x = np.arange(n, dtype=float)
    b, a = np.polyfit(x, y, 1)
    return y - (a + b * x)


def _prep_segment(x: np.ndarray) -> np.ndarray:
    xs = x.astype(float).copy()
    if APPLY_DETREND:
        xs = _linear_detrend(xs)
    if APPLY_HANN:
        xs *= np.hanning(xs.size)
    return xs


def lag_lowest_bin_fft(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    FFT-style lag:
      - split into N_SEGMENTS equal segments
      - rFFT each segment
      - average cross-spectrum across segments
      - pick lowest non-zero bin -> tau = phase / (2π f)

    Returns (f1, tau1). tau in same time units as t (days).
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    
    idx = np.argsort(t)
    t = t[idx]
    x = x[idx]
    y = y[idx]

    
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return np.nan, np.nan

    N = t.size
    seg_len = N // N_SEGMENTS
    if seg_len < 2:
        return np.nan, np.nan

    N_use = seg_len * N_SEGMENTS
    t = t[:N_use]
    x = x[:N_use]
    y = y[:N_use]

    C_sum = None
    freqs_ref = None

    for s in range(N_SEGMENTS):
        lo = s * seg_len
        hi = (s + 1) * seg_len
        xs = _prep_segment(x[lo:hi])
        ys = _prep_segment(y[lo:hi])

        X = np.fft.rfft(xs)
        Y = np.fft.rfft(ys)
        Cxy = np.conj(X) * Y  

        freqs = np.fft.rfftfreq(seg_len, d=dt)

        if freqs_ref is None:
            freqs_ref = freqs
            C_sum = np.zeros_like(Cxy, dtype=np.complex128)
        else:
            if not np.allclose(freqs, freqs_ref, atol=1e-12, rtol=0):
                return np.nan, np.nan

        C_sum += Cxy

    C_mean = C_sum / float(N_SEGMENTS)
    freqs = freqs_ref

    phase = np.angle(C_mean)

    valid = (freqs > 0) & np.isfinite(phase)
    if not np.any(valid):
        return np.nan, np.nan

    idx0 = np.where(valid)[0][np.argmin(freqs[valid])]
    f1 = float(freqs[idx0])
    tau1 = float(phase[idx0] / (2.0 * np.pi * f1))
    return f1, tau1


def fit_gp_once(t: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """
    Fit a Matern-0.5 GP ONCE on the full (gap-free) truth series.
    Returns a fitted sklearn GPR model. We will reuse its learned kernel hyperparameters.
    """
    X = np.asarray(t, float).reshape(-1, 1)
    y = np.asarray(y, float)

    
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=10.0, length_scale_bounds=(1e-2, 1e4), nu=0.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0,
    )
    gpr.fit(X, y)
    return gpr


def condition_gp_fixed_hyperparams(
    gpr_fitted: GaussianProcessRegressor,
    t_obs: np.ndarray,
    y_obs: np.ndarray,
) -> GaussianProcessRegressor:
    """
    Condition on gappy observations WITHOUT re-optimizing hyperparameters.
    We build a new GPR with kernel = gpr_fitted.kernel_ and optimizer=None.
    """
    Xobs = np.asarray(t_obs, float).reshape(-1, 1)
    yobs = np.asarray(y_obs, float)

    gpr = GaussianProcessRegressor(
        kernel=gpr_fitted.kernel_,   # fixed hyperparams
        optimizer=None,              # do NOT refit hyperparams
        alpha=0.0,
        normalize_y=True,
        random_state=0,
    )
    gpr.fit(Xobs, yobs)
    return gpr


def posterior_samples_on_grid(
    gpr: GaussianProcessRegressor,
    t_grid: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw posterior samples on the full uniform grid.
    Returns array shape (n_points, n_samples).
    """
    X = np.asarray(t_grid, float).reshape(-1, 1)

    rs = int(rng.integers(0, 2**31 - 1))
    samples = gpr.sample_y(X, n_samples=n_samples, random_state=rs)
    return np.asarray(samples, float)


def pick_new_contiguous_gap(keep: np.ndarray, gap_size: int, rng: np.random.Generator) -> tuple[int, int]:
    n = keep.size
    starts = []
    for s in range(0, n - gap_size + 1):
        if keep[s:s + gap_size].all():
            starts.append(s)
    if not starts:
        raise RuntimeError("No valid non-overlapping contiguous placement remains for another gap.")
    start = int(rng.choice(starts))
    return start, start + gap_size


def save_gappy_observed(step_label: str, t_obs: np.ndarray, V_obs: np.ndarray, W_obs: np.ndarray) -> None:
    pd.DataFrame({"MJD": t_obs, "Flux": V_obs}).to_csv(OUT_DIR / f"V_obs_{step_label}.csv", index=False)
    pd.DataFrame({"MJD": t_obs, "Flux": W_obs}).to_csv(OUT_DIR / f"W2_obs_{step_label}.csv", index=False)


def main():
    rng = np.random.default_rng(RNG_SEED)

    dfV = pd.read_csv(CSV_V, sep=None, engine="python", encoding="utf-8-sig")
    dfW = pd.read_csv(CSV_W2, sep=None, engine="python", encoding="utf-8-sig")

    if str(dfV.columns[0]).strip().upper() != "MJD":
        raise ValueError("First column of gp_samples_V must be MJD.")
    if str(dfW.columns[0]).strip().upper() != "MJD":
        raise ValueError("First column of gp_samples_W2 must be MJD.")

   
    n_rows = min(dfV.shape[0], dfW.shape[0])
    dfV = dfV.iloc[:n_rows, :].reset_index(drop=True)
    dfW = dfW.iloc[:n_rows, :].reset_index(drop=True)

    if MAX_TOTAL_REMOVED % GAP_SIZE != 0:
        raise ValueError("MAX_TOTAL_REMOVED must be a multiple of GAP_SIZE.")
    if MAX_TOTAL_REMOVED >= n_rows:
        raise ValueError(f"MAX_TOTAL_REMOVED must be < n_rows ({n_rows}).")

    n_pairs = min(dfV.shape[1] - 1, dfW.shape[1] - 1)
    if n_pairs <= 0:
        raise ValueError("Need at least one realisation column besides MJD.")

    t_full = dfV.iloc[:, 0].to_numpy(float)

    
    chosen = None
    for _ in range(MAX_PAIR_TRIES):
        k = int(rng.integers(1, n_pairs + 1))
        V_truth = dfV.iloc[:, k].to_numpy(float)
        W_truth = dfW.iloc[:, k].to_numpy(float)
        f0, tau0 = lag_lowest_bin_fft(t_full, V_truth, W_truth)
        if np.isfinite(tau0) and (TARGET_LAG_MIN_DAYS <= abs(tau0) <= TARGET_LAG_MAX_DAYS):
            chosen = (k, f0, tau0, V_truth, W_truth)
            break

    if chosen is None:
        raise RuntimeError(
            f"Could not find a truth pair with |tau| in [{TARGET_LAG_MIN_DAYS},{TARGET_LAG_MAX_DAYS}] "
            f"after {MAX_PAIR_TRIES} tries. Widen the target range or increase tries."
        )

    k, f0, tau0, V_truth, W_truth = chosen
    print(f"Selected truth pair column = {k}")
    print(f"Baseline (no gaps): f1={f0:.6g}  tau={tau0:.6g} days  |tau|={abs(tau0):.3f} days")

    
    
    print("\nFitting GP hyperparameters once on the full gap-free truth series...")
    gpV_fit = fit_gp_once(t_full, V_truth)
    gpW_fit = fit_gp_once(t_full, W_truth)
    print("Done.")
    print(f"V fitted kernel:  {gpV_fit.kernel_}")
    print(f"W2 fitted kernel: {gpW_fit.kernel_}")

    
    removed_points: list[int] = []
    tau_med_list: list[float] = []
    sigma_list: list[float] = []
    S_list: list[float] = []

    keep = np.ones(n_rows, dtype=bool)
    gaps: list[tuple[int, int]] = []

    stop_removed: int | None = None
    stop_sigma: float | None = None

    for total_removed in range(0, MAX_TOTAL_REMOVED + 1, GAP_SIZE):
        
        removed_points.append(total_removed)

        
        if total_removed > 0:
            start, end = pick_new_contiguous_gap(keep, GAP_SIZE, rng)
            keep[start:end] = False
            gaps.append((start, end))

        
        t_obs = t_full[keep]
        V_obs = V_truth[keep]
        W_obs = W_truth[keep]

        step_label = f"removed_{total_removed:03d}"
        save_gappy_observed(step_label, t_obs, V_obs, W_obs)

        
        gpV = condition_gp_fixed_hyperparams(gpV_fit, t_obs, V_obs)
        gpW = condition_gp_fixed_hyperparams(gpW_fit, t_obs, W_obs)

        
        V_samps = posterior_samples_on_grid(gpV, t_full, N_POST_SAMPLES, rng)  # (N, S)
        W_samps = posterior_samples_on_grid(gpW, t_full, N_POST_SAMPLES, rng)

        
        tau_samples = np.full(N_POST_SAMPLES, np.nan, dtype=float)
        f1_ref = None

        for j in range(N_POST_SAMPLES):
            f1, tau = lag_lowest_bin_fft(t_full, V_samps[:, j], W_samps[:, j])
            tau_samples[j] = tau
            if f1_ref is None and np.isfinite(f1):
                f1_ref = f1

        valid = np.isfinite(tau_samples)
        if np.sum(valid) < max(10, N_POST_SAMPLES // 10):
            tau_med = np.nan
            sigma = np.nan
            S = np.nan
        else:
            tau_med = float(np.median(tau_samples[valid]))
            sigma = float(np.std(tau_samples[valid], ddof=1))
            S = float(abs(tau_med) / sigma) if sigma > 0 else np.nan

        tau_med_list.append(tau_med)
        sigma_list.append(sigma)
        S_list.append(S)

        print("\n" + "-" * 60)
        print(f"Total removed = {total_removed:3d} points")
        if total_removed > 0:
            print(f"New gap = [{start}:{end}] ; gaps so far = {gaps}")
        print(f"Lowest-bin f1 ≈ {f1_ref:.6g}  (days^-1)")
        print(f"tau (median over GP posterior samples) = {tau_med:.6g} days")
        print(f"sigma (std over GP posterior samples)  = {sigma:.6g} days")
        print(f"S = |tau|/sigma                        = {S:.6g}")

        
        if np.isfinite(sigma) and sigma >= SIGMA_STOP:
            stop_removed = total_removed
            stop_sigma = sigma
            print("\n" + "!" * 60)
            print(f"STOP: sigma reached {stop_sigma:.6g} days (>= {SIGMA_STOP}) at total_removed = {stop_removed} points.")
            print("!" * 60)
            break

    
    plt.figure(figsize=(8, 4.5))
    plt.plot(removed_points, S_list, marker="o")
    plt.axhline(3.0, linestyle="--")
    plt.xlabel("Total points removed (cumulative contiguous gaps)")
    plt.ylabel("S = |tau| / sigma")
    plt.title("Significance S vs contiguous gaps (GP reconstruction + FFT)")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(removed_points, sigma_list, marker="o")
    plt.axhline(SIGMA_STOP, linestyle="--")
    plt.xlabel("Total points removed (cumulative contiguous gaps)")
    plt.ylabel("sigma [days]")
    plt.title("Sigma vs contiguous gaps (posterior-sample std of tau)")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4.5))
    plt.plot(removed_points, tau_med_list, marker="o")
    plt.axhline(tau0, linestyle="--")
    plt.xlabel("Total points removed (cumulative contiguous gaps)")
    plt.ylabel("tau [days]")
    plt.title("Tau vs contiguous gaps (median over posterior samples)")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()