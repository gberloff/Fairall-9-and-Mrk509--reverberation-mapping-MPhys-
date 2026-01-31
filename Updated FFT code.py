# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 12:12:04 2025

@author: gberl
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

filt = "V"

CSV_PATH_UVW2   = Path(f"gp_samples_{filt}_5548.csv")
CSV_PATH_V = Path(f"gp_samples_W2_5548.csv")  
N_SEGMENTS = 5


V_MEAN_ORIG    = None
V_STD_ORIG     = None
UVW2_MEAN_ORIG = None
UVW2_STD_ORIG  = None


_warned_units = False
def restore_units_if_set(y_norm: np.ndarray, mean_orig, std_orig) -> np.ndarray:
    """
    Undo normalisation if mean/std provided.
    Otherwise return y_norm (with a one-time warning).
    """
    global _warned_units
    if mean_orig is None or std_orig is None:
        if not _warned_units:
            print("Note: original mean/std not set — using normalized units as-is.")
            _warned_units = True
        return y_norm
    return y_norm * float(std_orig) + float(mean_orig)


def _linear_detrend(y: np.ndarray) -> np.ndarray:
    n = y.size
    x = np.arange(n, dtype=float)
    b, a = np.polyfit(x, y, 1)
    return y - (a + b*x)

def cross_spectrum_one_segment(
    v_flux: np.ndarray,
    uvw2_flux: np.ndarray,
    dt: float,
    apply_detrend: bool = True,
    apply_hann: bool = True
):
    """
    Compute cross spectrum C(f) = X*(f) * Y(f) for ONE segment.
    """
    if v_flux.size != uvw2_flux.size:
        raise ValueError("Segment arrays must have equal length.")

    n = v_flux.size
    if n < 2:
        raise ValueError("Segment too short for FFT.")

    x = v_flux.astype(float)
    y = uvw2_flux.astype(float)

    if apply_detrend:
        x = _linear_detrend(x)
        y = _linear_detrend(y)

    if apply_hann:
        w = np.hanning(n)
        power_scale = np.mean(w**2)
        x *= w
        y *= w
    else:
        power_scale = 1.0

    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)

    Cxy = np.conj(X) * Y / (power_scale * n)
    Pxx = (np.abs(X)**2) / (power_scale * n)
    Pyy = (np.abs(Y)**2) / (power_scale * n)

    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, Cxy, Pxx, Pyy


def make_uniform_overlap_grid(t1, y1, t2, y2, dt=None):
    """
    Build uniform time grid over the overlapping time range of (t1,y1) & (t2,y2),
    and interpolate both onto that grid.
    """
    m1 = np.isfinite(t1) & np.isfinite(y1)
    m2 = np.isfinite(t2) & np.isfinite(y2)
    t1, y1 = np.asarray(t1[m1]), np.asarray(y1[m1])
    t2, y2 = np.asarray(t2[m2]), np.asarray(y2[m2])

    i1 = np.argsort(t1); t1, y1 = t1[i1], y1[i1]
    i2 = np.argsort(t2); t2, y2 = t2[i2], y2[i2]

    tmin = max(t1.min(), t2.min())
    tmax = min(t1.max(), t2.max())

    if tmax <= tmin:
        raise ValueError("No temporal overlap between V and W2.")

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

    return t_grid, y1i, y2i, dt

def make_equal_segments(t_grid, y1, y2, n_segments):
    """
    Split onto n_segments strictly equal-length pieces by truncating the tail if needed.
    """
    N = t_grid.size
    seg_len = N // n_segments
    if seg_len < 2:
        raise ValueError("Segments too short.")

    N_use = seg_len * n_segments
    t_use = t_grid[:N_use]
    y1_use = y1[:N_use]
    y2_use = y2[:N_use]

    segments_V = []
    segments_U = []

    for i in range(n_segments):
        lo = i * seg_len
        hi = (i + 1) * seg_len
        segments_V.append((t_use[lo:hi], y1_use[lo:hi]))
        segments_U.append((t_use[lo:hi], y2_use[lo:hi]))

    return segments_V, segments_U, seg_len


def cross_spectra_across_segments(segments_V, segments_U):
    """
    Compute cross spectra for all segments of ONE pair.
    Returns:
        freqs_ref : frequency grid (identical across segments)
        C_list    : list of C_s(f) per segment s
        C_mean    : complex mean across segments
        Pxx_mean  : mean auto-spectrum of V across segments
        Pyy_mean  : mean auto-spectrum of other band across segments
    """
    C_list = []
    Pxx_list = []
    Pyy_list = []
    freqs_ref = None

    for (tV, fV), (tU, fU) in zip(segments_V, segments_U):
        dt = np.median(np.diff(tV))
        freqs, Cxy, Pxx, Pyy = cross_spectrum_one_segment(fV, fU, dt)

        if freqs_ref is None:
            freqs_ref = freqs
        else:
            if not np.allclose(freqs, freqs_ref, atol=1e-12, rtol=0):
                raise ValueError("Frequency grids differ across segments")

        C_list.append(Cxy)
        Pxx_list.append(Pxx)
        Pyy_list.append(Pyy)

    C_mean   = np.mean(np.vstack(C_list),   axis=0)
    Pxx_mean = np.mean(np.vstack(Pxx_list), axis=0)
    Pyy_mean = np.mean(np.vstack(Pyy_list), axis=0)

    return freqs_ref, C_list, C_mean, Pxx_mean, Pyy_mean


def plot_mean_cross_spectrum(freqs, C_mean, title_suffix=""):
    f = freqs[1:]               
    C = C_mean[1:]

    amplitude = np.abs(C)
    phase = np.unwrap(np.angle(C))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    ax1.loglog(f, amplitude)
    ax1.set_ylabel(r"$|C_{\text{mean}}(\nu)|$")
    ax1.set_title(f"Mean Cross Spectrum {title_suffix}")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    ax2.semilogx(f, phase)
    ax2.set_xlabel(r"Frequency $\nu$ [1/time unit]")
    ax2.set_ylabel("Phase [rad]")
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_average_time_lag(freqs, mean_time_lag, std_time_lag):
    mask = np.isfinite(freqs) & np.isfinite(mean_time_lag)
    f = freqs[mask]
    tau = mean_time_lag[mask]
    err = std_time_lag[mask]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(f, tau, yerr=err, fmt='o', markersize=3, linewidth=1, capsize=2)
    ax.set_xscale('log')
    ax.set_xlabel(r"Frequency $\nu$ [1/time unit]")
    ax.set_ylabel(r"Mean time lag $\tau$ [time unit]")
    ax.set_title("Average time lag vs frequency (1000 pairs)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    import os
    print("Current working directory:", os.getcwd())

    df_V = pd.read_csv(CSV_PATH_V, sep=None, engine="python", encoding="utf-8-sig")
    df_W2 = pd.read_csv(CSV_PATH_UVW2, sep=None, engine="python", encoding="utf-8-sig")

    if str(df_V.columns[0]).strip().upper() != "MJD":
        raise ValueError("First column of V CSV must be MJD.")
    if str(df_W2.columns[0]).strip().upper() != "MJD":
        raise ValueError("First column of W2 CSV must be MJD.")

    t_V = df_V.iloc[:, 0].to_numpy(float)
    t_W2 = df_W2.iloc[:, 0].to_numpy(float)

    n_samples_V = df_V.shape[1] - 1
    n_samples_W2 = df_W2.shape[1] - 1
    n_pairs = min(n_samples_V, n_samples_W2)

    print(f"Found {n_samples_V} V samples and {n_samples_W2} HX samples.")
    print(f"Processing {n_pairs} matched pairs (1 with 1, 2 with 2, ..., {n_pairs} with {n_pairs}).")

    rows = []
    coh_sum_C   = None
    coh_sum_Pxx = None
    coh_sum_Pyy = None
    coh_count   = 0

    sum_cos = None
    sum_sin = None
    count_phase = None
    sum_tau = None
    sum_tau2 = None
    count_tau = None
    freqs_ref_global = None

    for pair_idx in range(1, n_pairs + 1):
        print(f"\n=== Pair {pair_idx}/{n_pairs} ===")

        yV_norm = df_V.iloc[:, pair_idx].to_numpy(float)
        yW2_norm = df_W2.iloc[:, pair_idx].to_numpy(float)

        yV = restore_units_if_set(yV_norm, V_MEAN_ORIG, V_STD_ORIG)
        yW2 = restore_units_if_set(yW2_norm, UVW2_MEAN_ORIG, UVW2_STD_ORIG)

        t_grid, yV_u, yW2_u, base_dt = make_uniform_overlap_grid(t_V, yV, t_W2, yW2)

        segments_V, segments_W2, seg_len = make_equal_segments(t_grid, yV_u, yW2_u, N_SEGMENTS)

        freqs, C_list, C_mean, Pxx_mean, Pyy_mean = cross_spectra_across_segments(
            segments_V, segments_W2
        )

        print(f"  Segment length = {seg_len}, freq bins = {freqs.size}")

        if freqs_ref_global is None:
            freqs_ref_global = freqs
            n_freq = freqs.size
            sum_cos = np.zeros(n_freq, dtype=float)
            sum_sin = np.zeros(n_freq, dtype=float)
            count_phase = np.zeros(n_freq, dtype=int)
            sum_tau = np.zeros(n_freq, dtype=float)
            sum_tau2 = np.zeros(n_freq, dtype=float)
            count_tau = np.zeros(n_freq, dtype=int)
        else:
            if not np.allclose(freqs, freqs_ref_global, atol=1e-12, rtol=0):
                raise ValueError("Frequency grid differs between pairs.")
        if coh_sum_C is None:
            coh_sum_C   = np.zeros_like(C_mean,   dtype=complex)
            coh_sum_Pxx = np.zeros_like(Pxx_mean, dtype=float)
            coh_sum_Pyy = np.zeros_like(Pyy_mean, dtype=float)

        coh_sum_C   += C_mean
        coh_sum_Pxx += Pxx_mean
        coh_sum_Pyy += Pyy_mean
        coh_count   += 1
        
        phase = np.angle(C_mean)  

        time_lag = np.full_like(phase, np.nan, dtype=float)
        nonzero = freqs != 0.0
        time_lag[nonzero] = phase[nonzero] / (2.0 * np.pi * freqs[nonzero])

        valid_phase = np.isfinite(phase)
        sum_cos[valid_phase] += np.cos(phase[valid_phase])
        sum_sin[valid_phase] += np.sin(phase[valid_phase])
        count_phase[valid_phase] += 1

        valid_tau = np.isfinite(time_lag)
        sum_tau[valid_tau] += time_lag[valid_tau]
        sum_tau2[valid_tau] += time_lag[valid_tau]**2
        count_tau[valid_tau] += 1

        if pair_idx == 1:
            plot_mean_cross_spectrum(freqs, C_mean, title_suffix="(Pair 1)")
        elif pair_idx == n_pairs:
            plot_mean_cross_spectrum(freqs, C_mean, title_suffix=f"(Pair {n_pairs})")

        for f_val, c_val, phi_val, tau_val in zip(freqs, C_mean, phase, time_lag):
            rows.append({
                "pair":         pair_idx,      
                "freq":         f_val,
                "C_mean_real":  c_val.real,
                "C_mean_imag":  c_val.imag,
                "phase_lag":    phi_val,       
                "time_lag":     tau_val        
            })

    df_out = pd.DataFrame(rows)
    out_name = "cross_spectra_mean_all_pairs_with_lags.csv"
    df_out.to_csv(out_name, index=False)
    print(f"\nSaved mean cross spectra + lags for {n_pairs} pairs to '{Path(out_name).resolve()}'")
    print(f"Total rows: {len(df_out)} (pairs × freq bins).")

    mean_phase = np.full_like(sum_cos, np.nan, dtype=float)
    nonzero_count_phase = count_phase > 0
    mean_phase[nonzero_count_phase] = np.arctan2(
        sum_sin[nonzero_count_phase],
        sum_cos[nonzero_count_phase]
    )

    mean_time_lag = np.full_like(mean_phase, np.nan, dtype=float)
    nonzero_f = freqs_ref_global != 0.0
    mean_time_lag[nonzero_f] = mean_phase[nonzero_f] / (2.0 * np.pi * freqs_ref_global[nonzero_f])

    simple_mean_tau = np.full_like(sum_tau, np.nan, dtype=float)
    nonzero_count_tau = count_tau > 0
    simple_mean_tau[nonzero_count_tau] = sum_tau[nonzero_count_tau] / count_tau[nonzero_count_tau]

    var_tau = np.full_like(sum_tau, np.nan, dtype=float)
    nonzero_count_tau = count_tau > 0
    var_tau[nonzero_count_tau] = (
        sum_tau2[nonzero_count_tau] / count_tau[nonzero_count_tau]
        - simple_mean_tau[nonzero_count_tau]**2
    )
    var_tau[var_tau < 0] = 0.0
    std_tau = np.sqrt(var_tau)

    df_avg = pd.DataFrame({
        "freq":            freqs_ref_global,
        "mean_phase_lag":  mean_phase,      
        "mean_time_lag":   mean_time_lag,   
        "time_lag_std":    std_tau          
    })

    avg_name = "average_lags_over_pairs.csv"
    df_avg.to_csv(avg_name, index=False)
    print(f"Saved averaged phase/time lags + std to '{Path(avg_name).resolve()}'")

    plot_average_time_lag(freqs_ref_global, mean_time_lag, std_tau)
    freqs = freqs_ref_global
    tau   = mean_time_lag
    err   = std_tau

    valid = (
        np.isfinite(freqs) &
        np.isfinite(tau) &
        np.isfinite(err) &
        (freqs > 0.0)
    )

    if np.any(valid):
        idx_valid = np.where(valid)[0]
        idx_sorted = idx_valid[np.argsort(freqs[idx_valid])]

        idx0 = idx_sorted[0]
        best_f   = freqs[idx0]
        best_tau = tau[idx0]
        best_err = err[idx0]

        print("\n=== Selected reverberation lag (lowest non-zero frequency bin) ===")
        print(f"Frequency bin ν = {best_f:.6g} [1/days]")
        print(f"Time lag τ = {best_tau:.6g} ± {best_err:.6g} [days]")
    else:
        print("\nCould not determine reverberation lag: no valid non-zero frequency bins.")
   
    coherence = None
    if (coh_count > 0) and (coh_sum_C is not None):
        C_bar   = coh_sum_C   / coh_count
        Pxx_bar = coh_sum_Pxx / coh_count
        Pyy_bar = coh_sum_Pyy / coh_count

        denom = Pxx_bar * Pyy_bar
        coherence = np.full_like(denom, np.nan, dtype=float)

        valid_coh = (denom > 0) & np.isfinite(C_bar) & np.isfinite(denom)
        coherence[valid_coh] = (np.abs(C_bar[valid_coh])**2) / denom[valid_coh]

    # ==================================================================
    # SMALL CHANGE: use one common mask + log ν for BOTH lag & coherence
    # ==================================================================
    if coherence is not None:
        f_all   = freqs_ref_global
        tau_all = mean_time_lag
        err_all = std_tau
        coh_all = coherence

        common_mask = (
            np.isfinite(f_all) &
            (f_all > 0.0) &           # drop ν = 0 bin
            np.isfinite(tau_all) &
            np.isfinite(err_all) &
            np.isfinite(coh_all)
        )

        if np.any(common_mask):
            f_plot   = f_all[common_mask]
            tau_plot = tau_all[common_mask]
            err_plot = err_all[common_mask]
            coh_plot = coh_all[common_mask]

            if f_all.size > 1:
                df_bin = f_all[1] - f_all[0]      # full bin width
                xerror = np.full_like(f_plot, 0.5 * df_bin)
            else:
                xerror = np.zeros_like(f_plot)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

            # Time lag (same ν grid)
            ax1.errorbar(
                f_plot, tau_plot,
                xerr=xerror,
                yerr=err_plot,
                fmt='o', markersize=3, linewidth=1, capsize=2
            )
            ax1.set_xscale('log')
            ax1.set_ylabel(r"Mean time lag $\tau$ [days]")
            ax1.set_title(f"Time lag and coherence for {filt}-band, V as baseline")
            ax1.grid(True, which="both", ls="--", alpha=0.3)

            # Coherence (same ν grid)
            ax2.plot(f_plot, coh_plot, marker='o', linestyle='-', markersize=3)
            ax2.set_xscale('log')
            ax2.set_xlabel(r"Frequency $\nu$ [1/days]")
            ax2.set_ylabel(r"Coherence $\gamma^2$")
            ax2.set_ylim(0.0, 1.05)
            ax2.grid(True, which="both", ls="--", alpha=0.3)

            if ('best_tau' in locals()) and np.isfinite(best_tau):
                textstr = rf"$\tau = {best_tau:.3f} \pm {best_err:.3f}\ \mathrm{{days}}$"
                ax1.text(
                    0.98, 0.95,
                    textstr,
                    transform=ax1.transAxes,
                    fontsize=10,
                    ha='right',
                    va='top',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='black')
                )

            plt.tight_layout()
            fig.savefig("lag_and_coherence_vs_frequency.png", dpi=300)
            plt.show()
        else:
            print("No common valid frequency bins for combined lag+coherence plot.")
    else:
        print("Coherence could not be computed (no valid cross/auto spectra).")
    

if __name__ == "__main__":
    main()