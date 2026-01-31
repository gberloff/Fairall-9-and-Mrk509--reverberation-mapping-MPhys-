# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 17:26:33 2026

@author: gberl
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("F9LCs.csv")
TIME_COL = "MJD"
BAND_COL = "Band"
FLUX_COL = "Flux"
ERR_COL  = "Error"

BANDS_TO_ANALYSE = ("HX", "W2", "V")
OUTDIR = Path("psdII_outputs")
OUTDIR.mkdir(exist_ok=True)

# Lomb–Scargle controls
OVERSAMPLING = 8             
N_MC_NOISE = 300             
N_LOG_BINS = 22              
NOISE_SUBTRACT_FOR_BINNED = True  


HAVE_ASTROPY = False
try:
    from astropy.timeseries import LombScargle  
    HAVE_ASTROPY = True
except Exception:
    HAVE_ASTROPY = False


HAVE_SCIPY = False
try:
    from scipy.signal import lombscargle  
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def _infer_object_column(df: pd.DataFrame) -> str | None:
    for c in ["Object", "Galaxy", "Source", "Target", "Name"]:
        if c in df.columns:
            return c
    return None


def _make_object_name(raw: str | None) -> str:
    if raw is None or str(raw).strip() == "":
        return "F9LCs"
    s = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(raw).strip())
    return s or "object"


def _baseline(t: np.ndarray) -> float:
    return float(np.max(t) - np.min(t)) if t.size else float("nan")


def _median_cadence(t: np.ndarray) -> float:
    if t.size < 2:
        return float("nan")
    dt = np.diff(np.sort(t))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if dt.size else float("nan")


def build_frequency_grid(t: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    fmin = 1/T
    fmax = 0.5 / median_dt (pseudo-Nyquist)
    """
    T = _baseline(t)
    dt_med = _median_cadence(t)

    fmin = 1.0 / T if np.isfinite(T) and T > 0 else 1e-4
    fmax = 0.5 / dt_med if np.isfinite(dt_med) and dt_med > 0 else 1.0

    # Guardrails
    fmin = max(fmin, 1e-8)
    fmax = max(fmax, fmin * 20.0)
    fmax = min(fmax, 5e2)

    # Oversampled log grid: nice for plotting/binned PSD
    nfreq = int(OVERSAMPLING * 800)  # deterministic size
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), nfreq)
    return freqs, float(fmin), float(fmax)


def ls_power(t: np.ndarray, y: np.ndarray, dy: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Returns Lomb–Scargle power at freqs [day^-1].
    Preferred: Astropy GLS with dy (heteroscedastic).
    Fallback: SciPy lombscargle (unweighted), with mean subtraction.
    """
    if HAVE_ASTROPY:
        
        
        ls = LombScargle(t, y, dy=dy, fit_mean=True, center_data=True)
        p = ls.power(freqs, normalization="psd")
        return np.asarray(p, dtype=float)

    if HAVE_SCIPY:
        
        
        y0 = y - np.mean(y)
        ang = 2.0 * np.pi * freqs
        p = lombscargle(t, y0, ang, precenter=False, normalize=True)
        return np.asarray(p, dtype=float)

    raise ImportError(
        "Neither astropy.timeseries.LombScargle nor scipy.signal.lombscargle is available.\n"
        "Install astropy (recommended) or scipy."
    )


def mc_noise_floor(t: np.ndarray, dy: np.ndarray, freqs: np.ndarray, nmc: int) -> tuple[np.ndarray, np.ndarray]:
    """
    MC estimate of the measurement-noise floor:
      y_i ~ Normal(0, dy_i), compute LS power, take median and 16-84 spread across MC.
    """
    P = np.empty((nmc, freqs.size), dtype=float)
    rng = np.random.default_rng(12345)

    for k in range(nmc):
        yk = rng.normal(0.0, dy, size=dy.size)
        P[k, :] = ls_power(t, yk, dy, freqs)

    med = np.median(P, axis=0)
    lo = np.percentile(P, 16, axis=0)
    hi = np.percentile(P, 84, axis=0)
    return med, np.vstack([lo, hi])


def log_bin_psd(freqs: np.ndarray, power: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Log-bin power vs frequency. Returns:
      f_centers, p_median, p_spread (16th, 84th)
    """
    mask = np.isfinite(freqs) & np.isfinite(power) & (freqs > 0)
    f = freqs[mask]
    p = power[mask]

    
    edges = np.logspace(np.log10(np.min(f)), np.log10(np.max(f)), nbins + 1)
    f_cent = []
    p_med = []
    p_lo = []
    p_hi = []

    for i in range(nbins):
        m = (f >= edges[i]) & (f < edges[i+1])
        if np.count_nonzero(m) < 5:
            continue
        fi = f[m]
        pi = p[m]
        f_cent.append(np.exp(np.mean(np.log(fi))))
        p_med.append(np.median(pi))
        p_lo.append(np.percentile(pi, 16))
        p_hi.append(np.percentile(pi, 84))

    return np.array(f_cent), np.array(p_med), np.vstack([np.array(p_lo), np.array(p_hi)])


def plot_sampling(obj: str, band: str, t: np.ndarray) -> None:
    ts = np.sort(t)
    dt = np.diff(ts)
    plt.figure()
    plt.hist(dt[np.isfinite(dt) & (dt > 0)], bins=50)
    plt.xlabel("Sampling gaps Δt [days]")
    plt.ylabel("Count")
    plt.title(f"{obj} | {band} | sampling gap distribution")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{obj}_{band}_sampling.png", dpi=200)
    plt.close()


def plot_ls_raw(obj: str, band: str, freqs: np.ndarray, p: np.ndarray,
                pnoise_med: np.ndarray, pnoise_band: np.ndarray) -> None:
    plt.figure()
    plt.loglog(freqs, p, label="LS power (data)")
    plt.loglog(freqs, pnoise_med, label="MC noise floor (median)")
    # 16–84% band
    plt.fill_between(freqs, pnoise_band[0], pnoise_band[1], alpha=0.2, label="noise 16–84%")
    plt.xlabel("Frequency f [day$^{-1}$]")
    plt.ylabel("LS power (consistent normalization per band)")
    plt.title(f"{obj} | {band} | Lomb–Scargle periodogram + noise floor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{obj}_{band}_ls_raw.png", dpi=200)
    plt.close()


def plot_ls_binned(obj: str, band: str,
                   fbin: np.ndarray, pbin: np.ndarray, pbin_spread: np.ndarray,
                   fbin_n: np.ndarray, pbin_n: np.ndarray) -> None:
    plt.figure()
    plt.errorbar(fbin, pbin,
                 yerr=[pbin - pbin_spread[0], pbin_spread[1] - pbin],
                 fmt="o", capsize=2, label="Binned LS (data)")
    plt.plot(fbin_n, pbin_n, label="Binned noise floor (median)", linewidth=1)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency f [day$^{-1}$]")
    plt.ylabel("Binned LS power")
    plt.title(f"{obj} | {band} | log-binned Lomb–Scargle PSD estimate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{obj}_{band}_ls_binned.png", dpi=200)
    plt.close()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)

    required = {BAND_COL, TIME_COL, FLUX_COL, ERR_COL}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}. Found: {list(df.columns)}")

    df[BAND_COL] = df[BAND_COL].astype(str).str.strip()

    obj_col = _infer_object_column(df)
    if obj_col is None:
        objects = [(None, df)]
    else:
        objects = [(obj, sub.copy()) for obj, sub in df.groupby(obj_col)]

    summary_rows = []

    print("=== PSD II: Lomb–Scargle ===")
    print(f"Astropy GLS available: {HAVE_ASTROPY}")
    print(f"SciPy lombscargle available (fallback): {HAVE_SCIPY}")
    if not HAVE_ASTROPY:
        print("NOTE: Without astropy, the fallback is unweighted Lomb–Scargle (less correct with heteroscedastic errors).")

    for obj_name, subdf in objects:
        obj = _make_object_name(obj_name)

        for band in BANDS_TO_ANALYSE:
            band_df = subdf[subdf[BAND_COL] == band].copy()
            if band_df.empty:
                summary_rows.append({"Object": obj, "Band": band, "ok": False, "reason": "No rows for this band"})
                continue

            t = band_df[TIME_COL].to_numpy(dtype=float)
            y = band_df[FLUX_COL].to_numpy(dtype=float)
            dy = band_df[ERR_COL].to_numpy(dtype=float)

            mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
            t, y, dy = t[mask], y[mask], dy[mask]

            if t.size < 8:
                summary_rows.append({"Object": obj, "Band": band, "ok": False, "reason": "Too few points after cleaning"})
                continue

            
            idx = np.argsort(t)
            t, y, dy = t[idx], y[idx], dy[idx]

            T = _baseline(t)
            dt_med = _median_cadence(t)

            
            freqs, fmin, fmax = build_frequency_grid(t)

            
            plot_sampling(obj, band, t)

            
            p = ls_power(t, y, dy, freqs)

            
            pnoise_med, pnoise_band = mc_noise_floor(t, dy, freqs, N_MC_NOISE)

            
            plot_ls_raw(obj, band, freqs, p, pnoise_med, pnoise_band)

            
            if NOISE_SUBTRACT_FOR_BINNED:
                p_use = p - pnoise_med
                
                p_use = np.where(p_use > 0, p_use, np.nan)
            else:
                p_use = p.copy()

            fbin, pbin, pbin_spread = log_bin_psd(freqs, p_use, N_LOG_BINS)
            fbin_n, pbin_n, _ = log_bin_psd(freqs, pnoise_med, N_LOG_BINS)

            plot_ls_binned(obj, band, fbin, pbin, pbin_spread, fbin_n, pbin_n)

            summary_rows.append({
                "Object": obj,
                "Band": band,
                "ok": True,
                "N": int(t.size),
                "baseline_days": T,
                "median_cadence_days": dt_med,
                "fmin_dayinv": fmin,
                "fmax_dayinv": fmax,
                "nfreq": int(freqs.size),
                "oversampling": int(OVERSAMPLING),
                "n_mc_noise": int(N_MC_NOISE),
                "n_log_bins": int(N_LOG_BINS),
                "used_astropy_gls": bool(HAVE_ASTROPY),
                "noise_subtract_binned": bool(NOISE_SUBTRACT_FOR_BINNED),
            })

    summary = pd.DataFrame(summary_rows)
    out_csv = OUTDIR / "summary_PSDII_LS.csv"
    summary.to_csv(out_csv, index=False)

    print("\n=== PSD II complete ===")
    print(f"Saved summary: {out_csv.resolve()}")
    print(f"Saved plots to:  {OUTDIR.resolve()}")
    print("\nSummary (head):")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()