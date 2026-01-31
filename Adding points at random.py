# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 14:30:57 2026

@author: gberl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt



CSV_V  = Path("gp_samples_V.csv")
CSV_W2 = Path("gp_samples_W2.csv")

N_SEGMENTS = 15          
N_ADD = 50              
INITIAL_REDUCTION = 300 

RNG_SEED = 12345
APPLY_DETREND = True
APPLY_HANN = True




def _linear_detrend(y):
    x = np.arange(len(y))
    b, a = np.polyfit(x, y, 1)
    return y - (a + b * x)


def cross_spectrum_one_segment(x, y, dt):
    if APPLY_DETREND:
        x = _linear_detrend(x)
        y = _linear_detrend(y)

    if APPLY_HANN:
        w = np.hanning(len(x))
        x = x * w
        y = y * w

    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)

    Cxy = np.conj(X) * Y
    Pxx = np.abs(X) ** 2
    Pyy = np.abs(Y) ** 2

    freqs = np.fft.rfftfreq(len(x), dt)
    return freqs, Cxy, Pxx, Pyy


def make_uniform_overlap_grid(t, x, y):
    dt = np.median(np.diff(t))
    t_grid = np.arange(t.min(), t.max(), dt)
    return t_grid, np.interp(t_grid, t, x), np.interp(t_grid, t, y), dt


def make_equal_segments(t, x, y):
    seg_len = len(t) // N_SEGMENTS
    t = t[:seg_len * N_SEGMENTS]
    x = x[:seg_len * N_SEGMENTS]
    y = y[:seg_len * N_SEGMENTS]

    segs = []
    for i in range(N_SEGMENTS):
        lo = i * seg_len
        hi = (i + 1) * seg_len
        segs.append((t[lo:hi], x[lo:hi], y[lo:hi]))
    return segs


def run_updated_fft(dfV, dfW):
    t = dfV.iloc[:, 0].values
    n_pairs = min(dfV.shape[1] - 1, dfW.shape[1] - 1)

    tau_all = []

    for i in range(1, n_pairs + 1):
        x = dfV.iloc[:, i].values
        y = dfW.iloc[:, i].values

        t_u, x_u, y_u, dt = make_uniform_overlap_grid(t, x, y)
        segs = make_equal_segments(t_u, x_u, y_u)

        C_list = []
        for ts, xs, ys in segs:
            freqs, Cxy, _, _ = cross_spectrum_one_segment(xs, ys, dt)
            C_list.append(Cxy)

        C_mean = np.mean(C_list, axis=0)
        phase = np.angle(C_mean)
        tau = phase / (2 * np.pi * freqs)
        tau_all.append(tau)

    tau_all = np.array(tau_all)

    
    idx = np.where(freqs > 0)[0][0]
    tau_mean = np.nanmean(tau_all[:, idx])
    tau_std = np.nanstd(tau_all[:, idx])

    S = np.abs(tau_mean) / tau_std
    return S



def main():
    rng = np.random.default_rng(RNG_SEED)

    dfV_full = pd.read_csv(CSV_V)
    dfW_full = pd.read_csv(CSV_W2)

    n_rows = len(dfV_full)


    removed = set(rng.choice(n_rows, INITIAL_REDUCTION, replace=False))
    available = set(range(n_rows)) - removed

    added_counts = []
    S_values = []

    for k in range(N_ADD + 1):
        keep = sorted(available)
        dfV = dfV_full.iloc[keep].reset_index(drop=True)
        dfW = dfW_full.iloc[keep].reset_index(drop=True)

        S = run_updated_fft(dfV, dfW)

        print(f"Added {k:2d} points | S = {S:.4f}")

        added_counts.append(k)
        S_values.append(S)

        if k < N_ADD:
            idx_add = rng.choice(list(removed))
            removed.remove(idx_add)
            available.add(idx_add)

 
    plt.figure(figsize=(7,4))
    plt.plot(added_counts, S_values, marker='o')
    plt.axhline(3, ls='--')
    plt.xlabel("Number of points added")
    plt.ylabel(r"Significance $S = |\tau|/\sigma_\tau$")
    plt.title("Lowest-bin lag significance vs added points")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()