# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:25:19 2026

@author: gberl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt



CSV_V = Path("gp_samples_V.csv")


CSV_W_PRIMARY = Path("gp_samples_W2.csv")
CSV_W_FALLBACK = Path("gp_samples_W.csv")

N_SEGMENTS = 15
N_REMOVE = 500

RNG_SEED = None  
APPLY_DETREND = True
APPLY_HANN = True



def _linear_detrend(y: np.ndarray) -> np.ndarray:
    n = y.size
    x = np.arange(n, dtype=float)
    b, a = np.polyfit(x, y, 1)
    return y - (a + b * x)


def cross_spectrum_one_segment(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    apply_detrend: bool = True,
    apply_hann: bool = True,
):
    """Compute cross spectrum C(f) = X*(f) Y(f) for ONE segment."""
    if x.size != y.size:
        raise ValueError("Segment arrays must have equal length.")
    n = x.size
    if n < 2:
        raise ValueError("Segment too short for FFT.")

    xs = x.astype(float)
    ys = y.astype(float)

    if apply_detrend:
        xs = _linear_detrend(xs)
        ys = _linear_detrend(ys)

    if apply_hann:
        w = np.hanning(n)
        power_scale = np.mean(w**2)
        xs *= w
        ys *= w
    else:
        power_scale = 1.0

    X = np.fft.rfft(xs)
    Y = np.fft.rfft(ys)

    Cxy = np.conj(X) * Y / (power_scale * n)
    Pxx = (np.abs(X) ** 2) / (power_scale * n)
    Pyy = (np.abs(Y) ** 2) / (power_scale * n)

    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, Cxy, Pxx, Pyy


def make_uniform_overlap_grid(t1, y1, t2, y2, dt=None):
    """
    Build uniform time grid over overlapping time range, then interpolate both onto it.
    (Matches the Updated FFT approach.)
    """
    t1 = np.asarray(t1, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    m1 = np.isfinite(t1) & np.isfinite(y1)
    m2 = np.isfinite(t2) & np.isfinite(y2)
    t1, y1 = t1[m1], y1[m1]
    t2, y2 = t2[m2], y2[m2]

    i1 = np.argsort(t1); t1, y1 = t1[i1], y1[i1]
    i2 = np.argsort(t2); t2, y2 = t2[i2], y2[i2]

    tmin = max(t1.min(), t2.min())
    tmax = min(t1.max(), t2.max())
    if tmax <= tmin:
        raise ValueError("No temporal overlap between the two series.")

    if dt is None:
        d1 = np.median(np.diff(t1))
        d2 = np.median(np.diff(t2))
        dt = np.nanmax([d1, d2])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Cannot infer valid dt.")

    n = int(np.floor((tmax - tmin) / dt)) + 1
    t_grid = tmin + dt * np.arange(n, dtype=float)

    y1i = np.interp(t_grid, t1, y1)
    y2i = np.interp(t_grid, t2, y2)

    return t_grid, y1i, y2i, float(dt)


def make_equal_segments(t_grid, y1, y2, n_segments: int):
    """Split into n_segments equal-length pieces by truncating the tail."""
    N = t_grid.size
    seg_len = N // n_segments
    if seg_len < 2:
        raise ValueError(f"Segments too short: seg_len={seg_len}, N={N}, n_segments={n_segments}")

    N_use = seg_len * n_segments
    t_use = t_grid[:N_use]
    y1_use = y1[:N_use]
    y2_use = y2[:N_use]

    segs1 = []
    segs2 = []
    for i in range(n_segments):
        lo = i * seg_len
        hi = (i + 1) * seg_len
        segs1.append((t_use[lo:hi], y1_use[lo:hi]))
        segs2.append((t_use[lo:hi], y2_use[lo:hi]))
    return segs1, segs2, seg_len


def cross_spectra_across_segments(segs1, segs2, apply_detrend=True, apply_hann=True):
    """Compute mean cross spectrum and mean autospectra across segments."""
    C_list, Pxx_list, Pyy_list = [], [], []
    freqs_ref = None

    for (t1, x), (t2, y) in zip(segs1, segs2):
        dt = np.median(np.diff(t1))
        freqs, Cxy, Pxx, Pyy = cross_spectrum_one_segment(
            x, y, dt,
            apply_detrend=apply_detrend,
            apply_hann=apply_hann,
        )

        if freqs_ref is None:
            freqs_ref = freqs
        else:
            if not np.allclose(freqs, freqs_ref, atol=1e-12, rtol=0):
                raise ValueError("Frequency grids differ across segments (unexpected).")

        C_list.append(Cxy)
        Pxx_list.append(Pxx)
        Pyy_list.append(Pyy)

    C_mean = np.mean(np.vstack(C_list), axis=0)
    Pxx_mean = np.mean(np.vstack(Pxx_list), axis=0)
    Pyy_mean = np.mean(np.vstack(Pyy_list), axis=0)

    return freqs_ref, C_mean, Pxx_mean, Pyy_mean


def compute_S_lowest_bin(t, x, y, M_segments: int):
    """
    Compute S = |tau(f1)| / dtau(f1) at the lowest non-zero frequency bin,
    using analytic dtau from coherence and M_segments.
    """
   
    t_grid, x_u, y_u, _dt = make_uniform_overlap_grid(t, x, t, y)

  
    segs1, segs2, _seg_len = make_equal_segments(t_grid, x_u, y_u, M_segments)

    
    freqs, C_mean, Pxx_mean, Pyy_mean = cross_spectra_across_segments(
        segs1, segs2,
        apply_detrend=APPLY_DETREND,
        apply_hann=APPLY_HANN,
    )

    
    phase = np.angle(C_mean)
    tau = np.full_like(phase, np.nan, dtype=float)
    nonzero_f = freqs != 0.0
    tau[nonzero_f] = phase[nonzero_f] / (2.0 * np.pi * freqs[nonzero_f])

   
    denom = Pxx_mean * Pyy_mean
    gamma2 = np.full_like(denom, np.nan, dtype=float)
    valid = (denom > 0) & np.isfinite(denom) & np.isfinite(C_mean)
    gamma2[valid] = (np.abs(C_mean[valid]) ** 2) / denom[valid]
    # Numerical guard: coherence should be in [0,1] ideally
    gamma2 = np.clip(gamma2, 0.0, 1.0)

    gamma = np.sqrt(gamma2)

    
    M = float(M_segments)
    dphi = np.full_like(gamma, np.nan, dtype=float)
    dtau = np.full_like(gamma, np.nan, dtype=float)

    eps = 1e-12
    ok = (freqs > 0.0) & np.isfinite(gamma) & (gamma > eps) & np.isfinite(gamma2)
    dphi[ok] = np.sqrt(1.0 - gamma2[ok]) / (gamma[ok] * np.sqrt(2.0 * M))
    dtau[ok] = dphi[ok] / (2.0 * np.pi * freqs[ok])

    
    good = (freqs > 0.0) & np.isfinite(tau) & np.isfinite(dtau) & (dtau > 0)
    if not np.any(good):
        return np.nan, np.nan, np.nan, np.nan  

    idx = np.where(good)[0]
    idx0 = idx[np.argmin(freqs[idx])]

    f1 = freqs[idx0]
    tau1 = tau[idx0]
    dtau1 = dtau[idx0]
    S = np.abs(tau1) / dtau1
    return float(S), float(f1), float(tau1), float(dtau1)


def main():
    rng = np.random.default_rng(RNG_SEED)

    
    dfV = pd.read_csv(CSV_V)
    if not CSV_W_PRIMARY.exists() and CSV_W_FALLBACK.exists():
        dfW = pd.read_csv(CSV_W_FALLBACK)
        w_path_used = CSV_W_FALLBACK
    else:
        dfW = pd.read_csv(CSV_W_PRIMARY)
        w_path_used = CSV_W_PRIMARY

    if str(dfV.columns[0]).strip().upper() != "MJD":
        raise ValueError("First column of gp_samples_V.csv must be MJD.")
    if str(dfW.columns[0]).strip().upper() != "MJD":
        raise ValueError(f"First column of {w_path_used.name} must be MJD.")

    tV = dfV.iloc[:, 0].to_numpy(float)
    tW = dfW.iloc[:, 0].to_numpy(float)

    
    if tV.size != tW.size or not np.allclose(tV, tW, atol=0.0, rtol=0.0):

        print("WARNING: V and W time grids differ. Proceeding anyway, but 'same-index' removal may not correspond to the same MJD.")

    nV = dfV.shape[1] - 1
    nW = dfW.shape[1] - 1
    if nV <= 0 or nW <= 0:
        raise ValueError("CSV files contain no sample columns.")

   
    colV = rng.integers(1, nV + 1)  
    colW = rng.integers(1, nW + 1)  

    yV_full = dfV.iloc[:, colV].to_numpy(float)
    yW_full = dfW.iloc[:, colW].to_numpy(float)

    print(f"Selected V column:  {dfV.columns[colV]} (index {colV})")
    print(f"Selected W column:  {dfW.columns[colW]} (index {colW}) from {w_path_used.name}")
    print(f"Using M = {N_SEGMENTS} segments, removing up to {N_REMOVE} points.\n")

    
    N = min(tV.size, yV_full.size, yW_full.size)
    all_indices = np.arange(N)
    removed_set = set()

    S_values = []
    removed_counts = []

    for k in range(0, N_REMOVE + 1):
        if k > 0:
            
            remaining = np.array([i for i in all_indices if i not in removed_set], dtype=int)
            if remaining.size == 0:
                print("No points left to remove.")
                break
            idx_remove = int(rng.choice(remaining))
            removed_set.add(idx_remove)

        keep_mask = np.ones(N, dtype=bool)
        if removed_set:
            keep_mask[list(removed_set)] = False

        t = tV[:N][keep_mask]
        x = yV_full[:N][keep_mask]
        y = yW_full[:N][keep_mask]

        try:
            S, f1, tau1, dtau1 = compute_S_lowest_bin(t, x, y, M_segments=N_SEGMENTS)
        except Exception as e:
            S, f1, tau1, dtau1 = np.nan, np.nan, np.nan, np.nan
            print(f"Removed {k:2d} points -> ERROR: {e}")

        removed_counts.append(k)
        S_values.append(S)

        if np.isfinite(S):
            print(f"Removed {k:2d} points | f1={f1:.6g} | tau={tau1:.6g} | dtau={dtau1:.6g} | S=|tau|/dtau = {S:.6g}")
        else:
            print(f"Removed {k:2d} points | S = NaN (could not compute valid lowest-bin lag/error)")

   
    plt.figure(figsize=(7, 4))
    plt.plot(removed_counts, S_values, marker="o", linestyle="-")
    plt.axhline(3.0, linestyle="--")
    plt.xlabel("Number of points removed")
    plt.ylabel(r"Significance $S = |\tau(f_1)| / \delta\tau(f_1)$")
    plt.title("Lowest-bin lag significance vs removed points")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()