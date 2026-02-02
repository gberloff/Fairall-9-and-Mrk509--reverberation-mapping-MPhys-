# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 18:05:20 2026

@author: gberl
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("F9LCs.csv")

TIME_COL = "MJD"
BAND_COL = "Band"
FLUX_COL = "Flux"
ERR_COL  = "Error"

BANDS = ("HX", "W2", "V")

OUTDIR = Path("psdIII_outputs_sf")
OUTDIR.mkdir(exist_ok=True)


N_LOG_BINS = 24              
MAX_PAIRS = 250_000          
BOOTSTRAP_PER_BIN = 250      
MIN_PAIRS_PER_BIN = 30       


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


def make_pairs(t: np.ndarray, y: np.ndarray, e: np.ndarray, max_pairs: int, rng: np.random.Generator):
    """
    Build pairwise lag and squared differences, with noise term.
    Returns arrays: dt, dy2, noise2
    Potentially huge O(N^2); we cap by random subsampling if needed.
    """
    n = t.size
    ii, jj = np.triu_indices(n, k=1)
    npairs = ii.size

    if npairs > max_pairs:
        sel = rng.choice(npairs, size=max_pairs, replace=False)
        ii = ii[sel]
        jj = jj[sel]

    dt = np.abs(t[jj] - t[ii])
    dy = y[jj] - y[ii]
    dy2 = dy * dy
    noise2 = e[ii] * e[ii] + e[jj] * e[jj]
    return dt, dy2, noise2


def log_bin_stats(x: np.ndarray, v: np.ndarray, nbins: int):
    """
    Log-bin v(x). Returns per-bin:
      x_center, v_mean, v_med, v16, v84, counts, edges
    """
    m = np.isfinite(x) & np.isfinite(v) & (x > 0)
    x = x[m]
    v = v[m]
    if x.size == 0:
        return (np.array([]),) * 6 + (None,)

    edges = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), nbins + 1)
    xc, meanv, medv, v16, v84, cnt = [], [], [], [], [], []

    for k in range(nbins):
        mk = (x >= edges[k]) & (x < edges[k + 1])
        if np.count_nonzero(mk) < MIN_PAIRS_PER_BIN:
            continue
        xv = x[mk]
        vv = v[mk]
        xc.append(np.exp(np.mean(np.log(xv))))
        meanv.append(float(np.mean(vv)))
        medv.append(float(np.median(vv)))
        v16.append(float(np.percentile(vv, 16)))
        v84.append(float(np.percentile(vv, 84)))
        cnt.append(int(np.count_nonzero(mk)))

    return (np.array(xc), np.array(meanv), np.array(medv),
            np.array(v16), np.array(v84), np.array(cnt), edges)


def bootstrap_bin_mean(x: np.ndarray, v: np.ndarray, edges: np.ndarray, nboot: int, rng: np.random.Generator):
    """
    For each bin, bootstrap the mean(v) to get 16–84% confidence bands.
    Returns: xc, mean, lo, hi, counts
    """
    xc_list, mean_list, lo_list, hi_list, cnt_list = [], [], [], [], []

    for k in range(len(edges) - 1):
        mk = (x >= edges[k]) & (x < edges[k + 1]) & np.isfinite(v) & np.isfinite(x)
        idx = np.where(mk)[0]
        if idx.size < MIN_PAIRS_PER_BIN:
            continue

        xv = x[idx]
        vv = v[idx]
        xc = float(np.exp(np.mean(np.log(xv))))
        m0 = float(np.mean(vv))

        # bootstrap means
        boots = np.empty(nboot, dtype=float)
        for b in range(nboot):
            res_idx = rng.choice(idx.size, size=idx.size, replace=True)
            boots[b] = float(np.mean(vv[res_idx]))

        lo = float(np.percentile(boots, 16))
        hi = float(np.percentile(boots, 84))

        xc_list.append(xc)
        mean_list.append(m0)
        lo_list.append(lo)
        hi_list.append(hi)
        cnt_list.append(int(idx.size))

    return (np.array(xc_list), np.array(mean_list),
            np.array(lo_list), np.array(hi_list), np.array(cnt_list))


def plot_sf(obj: str, band: str, lag: np.ndarray, mean: np.ndarray, lo: np.ndarray, hi: np.ndarray,
            title: str, ylabel: str, fname: str, logy: bool = True):
    plt.figure()
    plt.plot(lag, mean, marker="o", linewidth=1, label="binned mean")
    plt.fill_between(lag, lo, hi, alpha=0.2, label="bootstrap 16–84%")

    plt.xscale("log")
    if logy:
        plt.yscale("log")

    plt.xlabel(r"Lag $\Delta t$ [days]")
    plt.ylabel(ylabel)
    plt.title(f"{obj} | {band} | {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=200)
    plt.close()


def plot_counts(obj: str, band: str, lag: np.ndarray, counts: np.ndarray, fname: str):
    plt.figure()
    plt.plot(lag, counts, marker="o", linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Lag $\Delta t$ [days]")
    plt.ylabel("# pairs in bin")
    plt.title(f"{obj} | {band} | SF bin pair counts")
    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=200)
    plt.close()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)

    required = {BAND_COL, TIME_COL, FLUX_COL, ERR_COL}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}. Found: {list(df.columns)}")

    df[BAND_COL] = df[BAND_COL].astype(str).str.strip()

    obj_col = _infer_object_column(df)
    objects = [(None, df)] if obj_col is None else [(o, sub.copy()) for o, sub in df.groupby(obj_col)]

    rng = np.random.default_rng(20260131)
    summary_rows = []

    for obj_name, subdf in objects:
        obj = _make_object_name(obj_name)

        for band in BANDS:
            band_df = subdf[subdf[BAND_COL] == band].copy()
            if band_df.empty:
                summary_rows.append({"Object": obj, "Band": band, "ok": False, "reason": "No rows for this band"})
                continue

            t = band_df[TIME_COL].to_numpy(dtype=float)
            y = band_df[FLUX_COL].to_numpy(dtype=float)
            e = band_df[ERR_COL].to_numpy(dtype=float)

            m = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
            t, y, e = t[m], y[m], e[m]
            if t.size < 8:
                summary_rows.append({"Object": obj, "Band": band, "ok": False, "reason": "Too few points"})
                continue

            idx = np.argsort(t)
            t, y, e = t[idx], y[idx], e[idx]

            T = _baseline(t)
            dt_med = _median_cadence(t)

            
            lag, dy2, noise2 = make_pairs(t, y, e, MAX_PAIRS, rng)

            
            sf_raw = dy2
            sf_corr = dy2 - noise2

            
            lag_pos = lag[np.isfinite(lag) & (lag > 0)]
            if lag_pos.size < 50:
                summary_rows.append({"Object": obj, "Band": band, "ok": False, "reason": "Too few valid lags"})
                continue

            edges = np.logspace(np.log10(np.min(lag_pos)), np.log10(np.max(lag_pos)), N_LOG_BINS + 1)

            
            lagc_raw, mean_raw, lo_raw, hi_raw, cnt_raw = bootstrap_bin_mean(lag, sf_raw, edges, BOOTSTRAP_PER_BIN, rng)

            
            lagc_corr, mean_corr, lo_corr, hi_corr, cnt_corr = bootstrap_bin_mean(lag, sf_corr, edges, BOOTSTRAP_PER_BIN, rng)

            
            mean_corr_clip = np.where(mean_corr > 0, mean_corr, np.nan)
            lo_corr_clip   = np.where(lo_corr > 0, lo_corr, np.nan)
            hi_corr_clip   = np.where(hi_corr > 0, hi_corr, np.nan)

            
            plot_sf(
                obj, band,
                lagc_raw, mean_raw, lo_raw, hi_raw,
                title="Structure Function (raw)",
                ylabel=r"$\langle(\Delta y)^2\rangle$",
                fname=f"{obj}_{band}_sf_raw.png",
                logy=True
            )

            plot_sf(
                obj, band,
                lagc_corr, mean_corr_clip, lo_corr_clip, hi_corr_clip,
                title="Structure Function (noise-corrected; clipped for log)",
                ylabel=r"$\langle(\Delta y)^2 - (\sigma_i^2+\sigma_j^2)\rangle$",
                fname=f"{obj}_{band}_sf_noise_corr.png",
                logy=True
            )

            plot_counts(obj, band, lagc_raw, cnt_raw, fname=f"{obj}_{band}_sf_counts.png")

            
            k0 = max(1, int(0.8 * mean_corr.size))
            plateau = float(np.nanmedian(mean_corr[k0:])) if mean_corr.size else float("nan")

            summary_rows.append({
                "Object": obj,
                "Band": band,
                "ok": True,
                "N_points": int(t.size),
                "N_pairs_used": int(lag.size),
                "baseline_days": T,
                "median_cadence_days": dt_med,
                "min_lag_days": float(np.min(lag_pos)),
                "max_lag_days": float(np.max(lag_pos)),
                "n_log_bins": int(N_LOG_BINS),
                "bootstrap_per_bin": int(BOOTSTRAP_PER_BIN),
                "plateau_proxy_sf_corr": plateau,
                "note": "SF is a time-domain diagnostic; avoid direct PSD slope inference without calibration."
            })

    summary = pd.DataFrame(summary_rows)
    out_csv = OUTDIR / "summary_PSDIII_SF.csv"
    summary.to_csv(out_csv, index=False)

    print("\n=== PSD III (SF) complete ===")
    print(f"Saved summary: {out_csv.resolve()}")
    print(f"Saved plots to:  {OUTDIR.resolve()}")
    print("\nSummary (head):")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()