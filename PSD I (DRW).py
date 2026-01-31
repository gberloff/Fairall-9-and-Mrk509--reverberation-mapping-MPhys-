# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:59:53 2026

@author: gberl
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


DATA_PATH = Path("F9LCs.csv")
TIME_COL = "MJD"
BAND_COL = "Band"
FLUX_COL = "Flux"
ERR_COL  = "Error"

BANDS_TO_FIT = ("HX", "W2", "V")  

OUTDIR = Path("psdI_outputs")
OUTDIR.mkdir(exist_ok=True)


def _infer_object_column(df: pd.DataFrame) -> str | None:
    """Try to find a column that identifies object/galaxy; returns None if not present."""
    candidates = ["Object", "Galaxy", "Source", "Target", "Name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _safe_median_dt(t: np.ndarray) -> float:
    if len(t) < 2:
        return float("nan")
    d = np.diff(np.sort(t))
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else float("nan")

def _baseline(t: np.ndarray) -> float:
    if len(t) == 0:
        return float("nan")
    return float(np.max(t) - np.min(t))

def _make_object_name(raw: str | None) -> str:
    if raw is None or str(raw).strip() == "":
        return "F9LCs"
    # file-safe
    s = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(raw).strip())
    return s or "object"


def drw_kalman_loglike(y: np.ndarray, t: np.ndarray, yerr: np.ndarray,
                      log_tau: float, log_S: float, mu: float) -> tuple[float, np.ndarray]:
    """
    Returns (loglike, standardized_innovations)
    """
    tau = float(np.exp(log_tau))
    S = float(np.exp(log_S))
    n = len(y)

    
    if n == 0 or not np.isfinite(tau) or not np.isfinite(S) or tau <= 0 or S <= 0:
        return -np.inf, np.array([], dtype=float)

    
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]
    yerr = yerr[idx]

    
    m = mu                
    P = S                 
    logL = 0.0
    z_list = []

    for i in range(n):
        if i == 0:
            m_pred = m
            P_pred = P
        else:
            dt = float(t[i] - t[i-1])
            if dt < 0:
                return -np.inf, np.array([], dtype=float)
            a = math.exp(-dt / tau) if dt > 0 else 1.0
            q = S * (1.0 - a*a)     
            m_pred = mu + a * (m - mu)
            P_pred = a*a * P + q


        R = float(yerr[i] * yerr[i])
        if not np.isfinite(R) or R <= 0:
            return -np.inf, np.array([], dtype=float)

        v = float(y[i] - m_pred)           
        F = float(P_pred + R)              
        if not np.isfinite(F) or F <= 0:
            return -np.inf, np.array([], dtype=float)

        
        logL += -0.5 * (math.log(2.0 * math.pi) + math.log(F) + (v*v) / F)

        
        z_list.append(v / math.sqrt(F))

        
        K = P_pred / F
        m = m_pred + K * v
        P = (1.0 - K) * P_pred
        
        if P < 0:
            P = 0.0

    return float(logL), np.asarray(z_list, dtype=float)


def drw_fit(y: np.ndarray, t: np.ndarray, yerr: np.ndarray) -> dict:
    """
    Fits log_tau, log_S, mu by maximizing Kalman log-likelihood.
    Returns dict with fitted params, diagnostics, and approx uncertainties.
    """
    
    mask = np.isfinite(y) & np.isfinite(t) & np.isfinite(yerr) & (yerr > 0)
    y, t, yerr = y[mask], t[mask], yerr[mask]
    if len(y) < 5:
        return {"ok": False, "reason": "Too few points after cleaning (<5)."}

    
    mu0 = float(np.average(y, weights=1.0 / (yerr*yerr)))
    var_y = float(np.var(y - mu0)) if len(y) > 1 else 1.0
    S0 = max(var_y, 1e-6)

    T = _baseline(t)
    
    tau0 = max(min(0.2 * T, 1e4), 1e-3) if np.isfinite(T) and T > 0 else 10.0

    x0 = np.array([math.log(tau0), math.log(S0), mu0], dtype=float)

    
    bnds = [(math.log(1e-6), math.log(1e7)),
            (math.log(1e-12), math.log(1e12)),
            (None, None)]

    def nll(theta: np.ndarray) -> float:
        log_tau, log_S, mu = float(theta[0]), float(theta[1]), float(theta[2])
        ll, _ = drw_kalman_loglike(y, t, yerr, log_tau, log_S, mu)
        return 1e30 if not np.isfinite(ll) else -ll

    res = minimize(nll, x0, method="L-BFGS-B", bounds=bnds)
    if not res.success:
        return {"ok": False, "reason": f"Optimization failed: {res.message}"}

    log_tau_hat, log_S_hat, mu_hat = map(float, res.x)
    ll_hat, z = drw_kalman_loglike(y, t, yerr, log_tau_hat, log_S_hat, mu_hat)

    se_log_tau = float("nan")
    se_log_S = float("nan")
    se_mu = float("nan")
    try:
        Hinv = res.hess_inv.todense() if hasattr(res.hess_inv, "todense") else np.asarray(res.hess_inv)
        Hinv = np.asarray(Hinv, dtype=float)
        if Hinv.shape == (3, 3) and np.all(np.isfinite(Hinv)):
            se = np.sqrt(np.maximum(np.diag(Hinv), 0.0))
            se_log_tau, se_log_S, se_mu = map(float, se)
    except Exception:
        pass

    tau_hat = float(np.exp(log_tau_hat))
    S_hat = float(np.exp(log_S_hat))


    def ci_from_log(log_hat: float, se_log: float) -> tuple[float, float]:
        if not np.isfinite(se_log) or se_log <= 0:
            return (float("nan"), float("nan"))
        lo = math.exp(log_hat - 1.96 * se_log)
        hi = math.exp(log_hat + 1.96 * se_log)
        return (float(lo), float(hi))

    tau_ci = ci_from_log(log_tau_hat, se_log_tau)
    S_ci   = ci_from_log(log_S_hat, se_log_S)

    fb_hat = float(1.0 / (2.0 * math.pi * tau_hat))
    fb_ci = (float("nan"), float("nan"))
    if np.isfinite(tau_ci[0]) and np.isfinite(tau_ci[1]) and tau_ci[0] > 0 and tau_ci[1] > 0:
        fb_ci = (1.0 / (2.0 * math.pi * tau_ci[1]), 1.0 / (2.0 * math.pi * tau_ci[0]))  # monotone transform

    return {
        "ok": True,
        "n": int(len(y)),
        "mu": mu_hat,
        "tau": tau_hat,
        "S": S_hat,
        "loglike": float(ll_hat),
        "se_log_tau": se_log_tau,
        "se_log_S": se_log_S,
        "se_mu": se_mu,
        "tau_ci95": tau_ci,
        "S_ci95": S_ci,
        "fb": fb_hat,
        "fb_ci95": fb_ci,
        "t_sorted": np.sort(t),
        "y_sorted": y[np.argsort(t)],
        "yerr_sorted": yerr[np.argsort(t)],
        "z_innov": z,
    }


def plot_lightcurve(obj: str, band: str, t: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> None:
    plt.figure()
    plt.errorbar(t, y, yerr=yerr, fmt=".", capsize=0)
    plt.xlabel("MJD")
    plt.ylabel("Flux")
    plt.title(f"{obj} | {band} | raw flux light curve")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{obj}_{band}_lightcurve.png", dpi=200)
    plt.close()

def plot_residuals(obj: str, band: str, t: np.ndarray, z: np.ndarray) -> None:
    plt.figure()
    plt.axhline(0.0, linewidth=1)
    plt.scatter(t, z, s=12)
    plt.xlabel("MJD")
    plt.ylabel("Standardized innovation (z)")
    plt.title(f"{obj} | {band} | DRW fit diagnostic: standardized innovations")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{obj}_{band}_residuals.png", dpi=200)
    plt.close()

def plot_psd_shape(obj: str, band: str, tau_days: float, T: float, dt_med: float) -> None:
    """
    Shape-only PSD for DRW: P(f) = 1 / (1 + (2π f τ)^2), scaled so P(0)=1.
    Frequency units: 1/day
    """
    fmin = 1.0 / T if np.isfinite(T) and T > 0 else 1e-4
    fmax = 0.5 / dt_med if np.isfinite(dt_med) and dt_med > 0 else 1.0

    
    fmin = max(fmin, 1e-8)
    fmax = max(fmax, fmin * 10.0)
    fmax = min(fmax, 1e3)

    f = np.logspace(np.log10(fmin), np.log10(fmax), 400)
    Pshape = 1.0 / (1.0 + (2.0 * math.pi * f * tau_days) ** 2)

    plt.figure()
    plt.loglog(f, Pshape)
    plt.xlabel("Frequency f [day$^{-1}$]")
    plt.ylabel("PSD shape (scaled so P(0)=1)")
    plt.title(f"{obj} | {band} | DRW implied PSD shape (no normalization)")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"{obj}_{band}_psd_shape.png", dpi=200)
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
        objects = []
        for obj_name, sub in df.groupby(obj_col):
            objects.append((obj_name, sub.copy()))

    summary_rows = []

    for obj_name, subdf in objects:
        obj = _make_object_name(obj_name)

        for band in BANDS_TO_FIT:
            band_df = subdf[subdf[BAND_COL] == band].copy()
            if band_df.empty:
                summary_rows.append({
                    "Object": obj, "Band": band, "ok": False, "reason": "No rows for this band."
                })
                continue

            t = band_df[TIME_COL].to_numpy(dtype=float)
            y = band_df[FLUX_COL].to_numpy(dtype=float)
            e = band_df[ERR_COL].to_numpy(dtype=float)

            mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
            plot_lightcurve(obj, band, t[mask], y[mask], e[mask])

            fit = drw_fit(y, t, e)
            if not fit["ok"]:
                summary_rows.append({
                    "Object": obj, "Band": band, "ok": False, "reason": fit.get("reason", "Unknown failure")
                })
                continue

            t_sorted = fit["t_sorted"]
            y_sorted = fit["y_sorted"]
            yerr_sorted = fit["yerr_sorted"]
            z = fit["z_innov"]

            T = _baseline(t_sorted)
            dt_med = _safe_median_dt(t_sorted)

            plot_residuals(obj, band, t_sorted, z)

            plot_psd_shape(obj, band, fit["tau"], T, dt_med)

            summary_rows.append({
                "Object": obj,
                "Band": band,
                "ok": True,
                "N": fit["n"],
                "baseline_days": T,
                "median_cadence_days": dt_med,
                "mu": fit["mu"],
                "tau_days": fit["tau"],
                "tau_ci95_lo": fit["tau_ci95"][0],
                "tau_ci95_hi": fit["tau_ci95"][1],
                "S_var": fit["S"],  
                "S_ci95_lo": fit["S_ci95"][0],
                "S_ci95_hi": fit["S_ci95"][1],
                "bend_fb_dayinv": fit["fb"],
                "bend_fb_ci95_lo": fit["fb_ci95"][0],
                "bend_fb_ci95_hi": fit["fb_ci95"][1],
                "loglike": fit["loglike"],
                "se_log_tau": fit["se_log_tau"],
                "se_log_S": fit["se_log_S"],
                "se_mu": fit["se_mu"],
            })

    summary = pd.DataFrame(summary_rows)

    out_csv = OUTDIR / "summary_PSDI_DRW.csv"
    summary.to_csv(out_csv, index=False)

    print("\n=== PSD I complete ===")
    print(f"Saved summary: {out_csv.resolve()}")
    print(f"Saved plots to: {OUTDIR.resolve()}")
    print("\nSummary (head):")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()