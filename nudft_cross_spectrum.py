"""
NUDFT cross-spectrum lag pipeline for irregularly sampled AGN light curves.

Computes the frequency-dependent time lag tau(f) between two photometric
bands (W2 and V) of Fairall 9, using a multitaper Non-Uniform Discrete
Fourier Transform cross-spectrum, directly on the raw irregular observation
times with no interpolation, GP regression, or resampling.

Only dependencies: numpy, pandas, matplotlib (no scipy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt


CSV_PATH = Path("F9LCs.csv")

FILTER_1    = "W2"
FILTER_2_LIST = ["V", "B", "W1", "M2"]

N_FREQ = 350
FMAX_FACTOR = 0.25

WINDOW_LOCAL_MEDIAN_WIDTH = 5
WINDOW_QW_ETA = 8.0

K_TAPERS = 4
ORTHO_EPS = 1e-2

SEG_DUR_DAYS = 400.0
MIN_POINTS_PER_SEG = 15
MIN_SPAN_PER_SEG_DAYS = 150.0
S_MIN_SUPPORT = 3

COH_MIN = 0.50

N_MC = 200
RNG_SEED = 123

FREQ_CHUNK = 128
EPS = 1e-30

OUT_DIR = Path("nudft_cross_spectrum_outputs")



def _detect_band_column(df: pd.DataFrame) -> str:
    """Detect the band-identifying column by exclusion.

    Parameters
    ----------
    df : pd.DataFrame
        Raw CSV dataframe.

    Returns
    -------
    str
        Name of the band column.

    Notes
    -----
    Finds the column whose upper-cased, stripped name is not MJD, FLUX, or
    ERROR.  Falls back to the first non-numeric column if multiple candidates
    exist.
    """
    known = {"MJD", "FLUX", "ERROR"}
    candidates = []
    for c in df.columns:
        if c.strip().upper() not in known:
            candidates.append(c)
    if len(candidates) == 1:
        return candidates[0]
    for c in candidates:
        if "band" in c.strip().lower():
            return c
    for c in candidates:
        if not np.issubdtype(df[c].dtype, np.number):
            return c
    if candidates:
        return candidates[0]
    raise ValueError(f"Cannot identify band column among: {list(df.columns)}")


def _detect_column(df: pd.DataFrame, target: str) -> str:
    """Find a column whose stripped upper name matches `target`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target : str
        Upper-case name to match (e.g. "MJD", "FLUX", "ERROR").

    Returns
    -------
    str
        Actual column name in df.
    """
    for c in df.columns:
        if c.strip().upper() == target:
            return c
    raise ValueError(f"Cannot find column matching '{target}' in {list(df.columns)}")


def _force_positive_errors(err: np.ndarray) -> np.ndarray:
    """Replace missing/non-positive errors with the median of valid errors.

    Parameters
    ----------
    err : np.ndarray
        Raw 1-sigma flux errors.

    Returns
    -------
    np.ndarray
        Cleaned errors, all positive and >= 1e-12.
    """
    e = np.asarray(err, float)
    good = np.isfinite(e) & (e > 0)
    if np.any(good):
        fill = float(np.median(e[good]))
    else:
        fill = 1.0
    e2 = e.copy()
    e2[~good] = fill
    e2 = np.clip(e2, 1e-12, np.inf)
    return e2


def read_data(csv_path: Path,
              filter_1: str = FILTER_1,
              filter_2: str = FILTER_2_LIST[0]
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, np.ndarray]:
    """Read CSV and extract two bands without joining or interpolating.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file containing light-curve data.

    Returns
    -------
    tx, x, xe : np.ndarray
        Time (MJD), flux, error for FILTER_1 (W2).
    ty, y, ye : np.ndarray
        Time (MJD), flux, error for FILTER_2 (V).
    """
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")

    col_band = _detect_band_column(df)
    col_t = _detect_column(df, "MJD")
    col_flux = _detect_column(df, "FLUX")
    col_err = _detect_column(df, "ERROR")

    print(f"[INFO] Detected columns: band='{col_band}', time='{col_t}', "
          f"flux='{col_flux}', error='{col_err}'")

    def _extract(band: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = df[df[col_band].astype(str).str.strip() == band]
        if d.empty:
            raise ValueError(f"No rows for band '{band}'. "
                             f"Available: {df[col_band].unique()}")
        t = pd.to_numeric(d[col_t], errors="coerce").to_numpy(float)
        x = pd.to_numeric(d[col_flux], errors="coerce").to_numpy(float)
        e = pd.to_numeric(d[col_err], errors="coerce").to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(x)
        t, x, e = t[ok], x[ok], e[ok]
        e = _force_positive_errors(e)
        idx = np.argsort(t)
        return t[idx], x[idx], e[idx]

    tx, x, xe = _extract(filter_1)
    ty, y, ye = _extract(filter_2)
    print(f"[INFO] {filter_1}: N={tx.size}  |  {filter_2}: N={ty.size}")
    return tx, x, xe, ty, y, ye




def weighted_mean_subtract(x: np.ndarray, sigma: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, float]:
    """Subtract the inverse-variance weighted mean from a flux array.

    Parameters
    ----------
    x : np.ndarray
        Flux values.
    sigma : np.ndarray
        1-sigma uncertainties.

    Returns
    -------
    x_centered : np.ndarray
        Mean-subtracted flux.
    w : np.ndarray
        Inverse-variance weights 1/sigma^2.
    mu : float
        The weighted mean that was subtracted.
    """
    w = 1.0 / (sigma * sigma)
    mu = float(np.sum(w * x) / np.sum(w))
    return x - mu, w, mu




def make_frequency_grid(tx: np.ndarray, ty: np.ndarray, n_freq: int
                        ) -> tuple[np.ndarray, float, float, float]:
    """Build a log-spaced frequency grid from both bands' time arrays.

    Parameters
    ----------
    tx, ty : np.ndarray
        Observation times (MJD) for bands X and Y.
    n_freq : int
        Number of frequency bins.

    Returns
    -------
    freqs : np.ndarray
        Log-spaced frequencies in day^-1, shape (n_freq,).
    fmin, fmax : float
        Boundary frequencies in day^-1.
    T : float
        Global time span in days.
    """
    tmin = float(min(np.min(tx), np.min(ty)))
    tmax = float(max(np.max(tx), np.max(ty)))
    T = tmax - tmin
    if not np.isfinite(T) or T <= 0:
        raise ValueError("Global time span must be > 0.")

    fmin = 1.0 / T

    dtx = np.diff(np.sort(tx))
    dty = np.diff(np.sort(ty))
    all_dt = np.concatenate([dtx, dty])
    all_dt = all_dt[np.isfinite(all_dt) & (all_dt > 0)]
    if all_dt.size == 0:
        raise ValueError("No positive time differences found in either band.")
    dt_med = float(np.median(all_dt))

    fmax = FMAX_FACTOR / dt_med
    if fmax <= fmin:
        print(f"[WARNING] fmax={fmax:.3g} <= fmin={fmin:.3g}; "
              f"using fallback fmax = 10*fmin")
        fmax = 10.0 * fmin

    freqs = np.exp(np.linspace(np.log(fmin), np.log(fmax), n_freq))
    return freqs, fmin, fmax, T




def nudft_weighted(t: np.ndarray, x: np.ndarray, w: np.ndarray,
                   freqs: np.ndarray, chunk: int = FREQ_CHUNK) -> np.ndarray:
    """Compute the weighted NUDFT: X(f) = sum_n w_n x_n exp(-i 2pi f t_n).

    Parameters
    ----------
    t : np.ndarray
        Observation times (days).
    x : np.ndarray
        Signal values (already mean-subtracted and tapered if applicable).
    w : np.ndarray
        Inverse-variance weights.
    freqs : np.ndarray
        Frequency grid (day^-1).
    chunk : int
        Number of frequencies per matrix-multiply chunk.

    Returns
    -------
    np.ndarray, complex128
        NUDFT values at each frequency, shape (len(freqs),).
    """
    out = np.zeros(freqs.size, dtype=np.complex128)
    if t.size == 0:
        return out
    vx = w * x
    twopi = 2.0 * np.pi
    for i0 in range(0, freqs.size, chunk):
        i1 = min(freqs.size, i0 + chunk)
        ff = freqs[i0:i1].reshape(-1, 1)
        ph = -1j * twopi * (ff * t.reshape(1, -1))
        E = np.exp(ph)
        out[i0:i1] = E @ vx
    return out


def window_function(t: np.ndarray, w: np.ndarray, freqs: np.ndarray,
                    chunk: int = FREQ_CHUNK) -> np.ndarray:
    """Spectral window function: NUDFT of weights (x_n = 1 for all n).

    Parameters
    ----------
    t : np.ndarray
        Observation times.
    w : np.ndarray
        Inverse-variance weights.
    freqs : np.ndarray
        Frequency grid.
    chunk : int
        Chunk size.

    Returns
    -------
    np.ndarray, complex128
        Window function values.
    """
    return nudft_weighted(t, np.ones_like(t, dtype=float), w, freqs, chunk)




def local_median_log(arr: np.ndarray, width: int) -> np.ndarray:
    """Running median in log space with a window of `width` bins.

    Parameters
    ----------
    arr : np.ndarray
        Input array (positive values expected).
    width : int
        Running window width (bins).

    Returns
    -------
    np.ndarray
        Smoothed baseline, same length as input.
    """
    width = max(width, 3)
    a = np.clip(np.asarray(arr, float), EPS, np.inf)
    la = np.log(a)
    n = la.size
    out = np.zeros(n, float)
    hw = width // 2
    for i in range(n):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        out[i] = float(np.median(la[lo:hi]))
    return np.exp(out)


def compute_window_mask(Pw: np.ndarray, width: int, eta: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Compute the window contamination mask.

    Parameters
    ----------
    Pw : np.ndarray
        Cross-window power |Wx*(f) Wy(f)|^2.
    width : int
        Local median window width.
    eta : float
        Contamination threshold.

    Returns
    -------
    win_ok : np.ndarray, bool
        True where Qw <= eta and is finite.
    Qw : np.ndarray
        Window contamination score Pw / median(Pw).
    """
    med = local_median_log(Pw, width=width)
    Qw = Pw / np.clip(med, EPS, np.inf)
    win_ok = np.isfinite(Qw) & (Qw <= eta)
    return win_ok, Qw




def initial_tapers(u: np.ndarray, K: int) -> np.ndarray:
    """Construct initial sine taper matrix evaluated at normalised times.

    Parameters
    ----------
    u : np.ndarray
        Normalised times in [0, 1].
    K : int
        Number of tapers.

    Returns
    -------
    H : np.ndarray, shape (N, K)
        Taper matrix.  H[:,0] = sin(pi*u), H[:,k] = sin(pi*k*u) for k>=1.
    """
    K = max(1, int(K))
    N = u.size
    H = np.zeros((N, K), float)
    for k in range(K):
        H[:, k] = np.sin(np.pi * (k + 1) * u)
    return H


def weighted_gram_schmidt(H: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Orthogonalise columns of H w.r.t. weighted inner product <u,v>_w.

    Parameters
    ----------
    H : np.ndarray, shape (N, K)
        Input taper matrix.
    w : np.ndarray, shape (N,)
        Weights (inverse-variance).

    Returns
    -------
    Q : np.ndarray, shape (N, K_eff)
        Orthonormal taper matrix; K_eff <= K columns retained.
    """
    w = np.asarray(w, float)
    Q = np.zeros_like(H, float)
    kept = 0
    for j in range(H.shape[1]):
        v = H[:, j].copy()
        for i in range(kept):
            qi = Q[:, i]
            denom = np.sum(w * qi * qi)
            if denom <= EPS:
                continue
            proj = np.sum(w * qi * v) / denom
            v -= proj * qi
        norm2 = np.sum(w * v * v)
        if norm2 <= EPS:
            continue
        Q[:, kept] = v / np.sqrt(norm2)
        kept += 1
    return Q[:, :max(1, kept)]


def orthogonality_check(Q: np.ndarray, w: np.ndarray) -> float:
    """Return max |off-diagonal| of the weighted correlation matrix.

    Parameters
    ----------
    Q : np.ndarray, shape (N, K)
        Orthonormal taper matrix.
    w : np.ndarray, shape (N,)
        Weights.

    Returns
    -------
    float
        Maximum absolute off-diagonal correlation; 0.0 if K <= 1.
    """
    K = Q.shape[1]
    if K <= 1:
        return 0.0
    G = np.zeros((K, K), float)
    for i in range(K):
        for j in range(K):
            G[i, j] = float(np.sum(w * Q[:, i] * Q[:, j]))
    d = np.sqrt(np.clip(np.diag(G), EPS, np.inf))
    C = G / (d.reshape(-1, 1) * d.reshape(1, -1))
    off = C - np.diag(np.diag(C))
    return float(np.max(np.abs(off)))


def prepare_tapers(u: np.ndarray, w: np.ndarray, K: int
                   ) -> np.ndarray:
    """Build and orthogonalise tapers; reduce if orthogonality fails.

    Parameters
    ----------
    u : np.ndarray
        Normalised times in [0, 1].
    w : np.ndarray
        Weights.
    K : int
        Desired number of tapers.

    Returns
    -------
    Q : np.ndarray, shape (N, K_eff)
        Orthonormal taper matrix.
    """
    H = initial_tapers(u, K)
    Q = weighted_gram_schmidt(H, w)
    if orthogonality_check(Q, w) > ORTHO_EPS:
        H2 = initial_tapers(u, min(2, K))
        Q = weighted_gram_schmidt(H2, w)
    return Q




def build_time_windows(tmin: float, tmax: float
                       ) -> list[tuple[float, float]]:
    """Divide [tmin, tmax] into contiguous non-overlapping windows.

    Parameters
    ----------
    tmin, tmax : float
        Global time range (days).

    Returns
    -------
    list of (float, float)
        (start, end) of each window.
    """
    windows = []
    a = tmin
    while a < tmax:
        b = min(tmax, a + SEG_DUR_DAYS)
        windows.append((a, b))
        a = b
    return windows


def slice_segment(t: np.ndarray, x: np.ndarray, e: np.ndarray,
                  t0: float, t1: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract observations within [t0, t1).

    Parameters
    ----------
    t, x, e : np.ndarray
        Full time, flux, error arrays.
    t0, t1 : float
        Segment boundaries.

    Returns
    -------
    Sliced t, x, e arrays.
    """
    m = (t >= t0) & (t < t1)
    return t[m], x[m], e[m]


def segment_valid(t_seg: np.ndarray) -> bool:
    """Check whether a segment has enough points and time span.

    Parameters
    ----------
    t_seg : np.ndarray
        Observation times in the segment.

    Returns
    -------
    bool
    """
    if t_seg.size < MIN_POINTS_PER_SEG:
        return False
    span = float(np.max(t_seg) - np.min(t_seg))
    return span >= MIN_SPAN_PER_SEG_DAYS




def estimate_cross_spectrum(
    tx: np.ndarray, x: np.ndarray, xe: np.ndarray,
    ty: np.ndarray, y: np.ndarray, ye: np.ndarray,
    freqs: np.ndarray, win_ok: np.ndarray
) -> dict:
    """Compute the multitaper NUDFT cross-spectrum with segmentation.

    Implements Requirements 1 (global+local mean subtraction), 3, 5, 6.

    Parameters
    ----------
    tx, x, xe : np.ndarray
        Time, flux, error for band X (W2).
    ty, y, ye : np.ndarray
        Time, flux, error for band Y (V).
    freqs : np.ndarray
        Log-spaced frequency grid (day^-1).
    win_ok : np.ndarray, bool
        Window contamination mask.

    Returns
    -------
    dict with keys: C_mean, Pxx_mean, Pyy_mean, coh, phi_circular_mean,
        support, K_eff_total, n_windows_used.
    """
    F = freqs.size

    x0, _, _ = weighted_mean_subtract(x, xe)
    y0, _, _ = weighted_mean_subtract(y, ye)

    tmin = float(min(np.min(tx), np.min(ty)))
    tmax = float(max(np.max(tx), np.max(ty)))
    windows = build_time_windows(tmin, tmax)

    C_acc = np.zeros(F, dtype=np.complex128)
    Pxx_acc = np.zeros(F, dtype=float)
    Pyy_acc = np.zeros(F, dtype=float)
    U_acc = np.zeros(F, dtype=np.complex128)
    taper_count = np.zeros(F, dtype=float)
    support = np.zeros(F, dtype=int)

    n_used = 0

    for (t0, t1) in windows:
        tsx, xs, xes = slice_segment(tx, x0, xe, t0, t1)
        tsy, ys, yes = slice_segment(ty, y0, ye, t0, t1)

        print(
            f"[DIAG] Seg {t0:.0f}–{t1:.0f}: "
            f"W2 N={tsx.size} span={float(np.max(tsx)-np.min(tsx)) if tsx.size>1 else 0:.0f}d  "
            f"V  N={tsy.size} span={float(np.max(tsy)-np.min(tsy)) if tsy.size>1 else 0:.0f}d  "
            f"valid={segment_valid(tsx) and segment_valid(tsy)}"
        )

        if not segment_valid(tsx) or not segment_valid(tsy):
            continue

        Tx = float(np.max(tsx) - np.min(tsx))
        Ty = float(np.max(tsy) - np.min(tsy))
        Tseg = min(Tx, Ty)
        if not np.isfinite(Tseg) or Tseg <= 0:
            continue
        seg_ok_f = freqs >= (1.0 / Tseg)

        xs_loc, wxs, _ = weighted_mean_subtract(xs, xes)
        ys_loc, wys, _ = weighted_mean_subtract(ys, yes)

        ux = (tsx - float(np.min(tsx))) / max(Tx, EPS)
        uy = (tsy - float(np.min(tsy))) / max(Ty, EPS)

        Qx = prepare_tapers(ux, wxs, K_TAPERS)
        Qy = prepare_tapers(uy, wys, K_TAPERS)

        K_eff = int(min(Qx.shape[1], Qy.shape[1]))
        if K_eff <= 0:
            continue

        for k in range(K_eff):
            hx = Qx[:, k]
            hy = Qy[:, k]

            Xk = nudft_weighted(tsx, hx * xs_loc, wxs, freqs)
            Yk = nudft_weighted(tsy, hy * ys_loc, wys, freqs)

            Ck = np.conj(Xk) * Yk
            Pxxk = np.abs(Xk) ** 2
            Pyyk = np.abs(Yk) ** 2

            Ck = np.where(seg_ok_f, Ck, 0.0 + 0.0j)
            Pxxk = np.where(seg_ok_f, Pxxk, 0.0)
            Pyyk = np.where(seg_ok_f, Pyyk, 0.0)

            C_acc += Ck
            Pxx_acc += Pxxk
            Pyy_acc += Pyyk

            mag = np.abs(Ck)
            unit = np.where(mag > 0, Ck / np.clip(mag, EPS, np.inf),
                            0.0 + 0.0j)
            U_acc += np.where(seg_ok_f, unit, 0.0 + 0.0j)

            taper_count += seg_ok_f.astype(float)

        support += seg_ok_f.astype(int)
        n_used += 1

    denom = np.clip(taper_count, 1.0, np.inf)

    C_mean = C_acc / denom
    Pxx_mean = Pxx_acc / denom
    Pyy_mean = Pyy_acc / denom

    coh = (np.abs(C_mean) ** 2) / np.clip(Pxx_mean * Pyy_mean, EPS, np.inf)
    coh = np.clip(coh, 0.0, 1.0)

    U_mean = U_acc / denom
    phi_circ = np.angle(U_mean)

    return dict(
        C_mean=C_mean,
        Pxx_mean=Pxx_mean,
        Pyy_mean=Pyy_mean,
        coh=coh,
        phi_circular_mean=phi_circ,
        support=support,
        taper_count=taper_count,
        n_windows_used=n_used,
    )




def unwrap_in_islands(phi: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Unwrap phase only within contiguous islands of good frequency bins.

    Parameters
    ----------
    phi : np.ndarray
        Raw phase in (-pi, pi].
    good : np.ndarray, bool
        Mask of trusted frequency bins.

    Returns
    -------
    np.ndarray
        Unwrapped phase; NaN outside good islands.
    """
    phi = np.asarray(phi, float)
    good = np.asarray(good, bool)
    out = np.full_like(phi, np.nan, dtype=float)
    n = phi.size
    i = 0
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i
        while j < n and good[j]:
            j += 1
        out[i:j] = np.unwrap(phi[i:j])
        i = j
    return out


def wrap_to_pi(phi: np.ndarray) -> np.ndarray:
    """Wrap phase array to (-pi, pi]."""
    phi = np.asarray(phi, float)
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def tau_from_wrapped_phase(freqs: np.ndarray,
                           phi_wrapped: np.ndarray) -> np.ndarray:
    """Convert wrapped phase to time lag tau = phi / (2*pi*f).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array (day^-1).
    phi_wrapped : np.ndarray
        Phase array (radians), already wrapped to (-pi, pi].

    Returns
    -------
    np.ndarray
        Time lag in days; NaN where freqs <= 0.
    """
    f = np.asarray(freqs, float)
    out = np.full_like(f, np.nan, dtype=float)
    m = f > 0.0
    out[m] = np.asarray(phi_wrapped, float)[m] / (2.0 * np.pi * f[m])
    return out


def model_tau_wrapped_two_param(freqs: np.ndarray,
                                Zphase_fit: float,
                                Ztime_fit: float,
                                f0: float) -> np.ndarray:
    """Two-parameter lag model combining constant phase and constant time lag.

    The total model phase is:
        phi_total = wrap( 2*pi*f0*Zphase + 2*pi*f*Ztime )
    and the returned lag is phi_total / (2*pi*f).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array (day^-1).
    Zphase_fit : float
        Constant-phase parameter (days); sets phase offset at f0.
    Ztime_fit : float
        Constant-time-lag parameter (days); the reverberation lag.
    f0 : float
        Anchor frequency (lowest valid bin, day^-1).

    Returns
    -------
    np.ndarray
        Model time lag tau(f) in days.
    """
    f = np.asarray(freqs, float)
    phi0 = 2.0 * np.pi * float(f0) * float(Zphase_fit)
    phi_total = wrap_to_pi(phi0 + 2.0 * np.pi * f * float(Ztime_fit))
    return tau_from_wrapped_phase(f, phi_total)


def constant_time_curve_wrapped(freqs: np.ndarray,
                                Ztime: float) -> np.ndarray:
    """Pure constant-time-lag curve: tau(f) = Ztime for all f (after wrap).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array (day^-1).
    Ztime : float
        Constant time lag (days).

    Returns
    -------
    np.ndarray
        Wrapped time lag in days.
    """
    f = np.asarray(freqs, float)
    phi_time = wrap_to_pi(2.0 * np.pi * f * float(Ztime))
    return tau_from_wrapped_phase(f, phi_time)




def main() -> None:
    """Run the full NUDFT cross-spectrum lag pipeline for all target bands."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results_rows = []

    for FILTER_2 in FILTER_2_LIST:
        print(f"\n{'='*60}")
        print(f"  Processing: {FILTER_1} (ref) vs {FILTER_2} (target)")
        print(f"{'='*60}\n")

        rng = np.random.default_rng(RNG_SEED)

        try:
            tx, x, xe, ty, y, ye = read_data(CSV_PATH, FILTER_1, FILTER_2)
        except ValueError as _e:
            print(f"[WARNING] Skipping {FILTER_2}: {_e}")
            continue

        if tx.size == 0 or ty.size == 0:
            print("[WARNING] One or both bands have no data. Exiting.")
            continue

        freqs, fmin, fmax, T = make_frequency_grid(tx, ty, N_FREQ)
        print(f"[INFO] T={T:.1f} d  fmin={fmin:.3g}  fmax={fmax:.3g}  "
              f"Nf={freqs.size}")

        if freqs.size == 0:
            print("[WARNING] Empty frequency grid. Exiting.")
            continue

        _, wx_global, _ = weighted_mean_subtract(x, xe)
        _, wy_global, _ = weighted_mean_subtract(y, ye)

        Wx = window_function(tx, wx_global, freqs)
        Wy = window_function(ty, wy_global, freqs)

        Wxy = np.conj(Wx) * Wy
        Pw = np.abs(Wxy) ** 2

        win_ok, Qw = compute_window_mask(Pw, WINDOW_LOCAL_MEDIAN_WIDTH,
                                         WINDOW_QW_ETA)
        Pw_median = local_median_log(Pw, WINDOW_LOCAL_MEDIAN_WIDTH)

        est = estimate_cross_spectrum(tx, x, xe, ty, y, ye, freqs, win_ok)

        coh = est["coh"]
        support = est["support"]
        taper_count = est["taper_count"]
        phi = est["phi_circular_mean"]

        print(f"[INFO] Segments used: {est['n_windows_used']}")

        support_ok = support >= S_MIN_SUPPORT
        coh_ok = coh >= COH_MIN
        good = win_ok & support_ok & coh_ok

        M_eff = taper_count.copy()

        phi_unw = unwrap_in_islands(phi, good)
        tau = -phi_unw / (2.0 * np.pi * freqs)

        out_csv = OUT_DIR / "nudft_cross_spectrum_summary.csv"
        pd.DataFrame({
            "f_dayinv": freqs,
            "window_Pw": Pw,
            "window_Qw": Qw,
            "window_ok": win_ok.astype(int),
            "support": support,
            "support_ok": support_ok.astype(int),
            "coherence": coh,
            "M_eff": M_eff,
            "coh_ok": coh_ok.astype(int),
            "good": good.astype(int),
            "phi_circular_mean": phi,
            "phi_unwrapped": phi_unw,
            "tau_days": tau,
        }).to_csv(out_csv, index=False)
        print(f"[INFO] Saved: {out_csv.resolve()}")

        print(f"[INFO] Monte Carlo: N_MC={N_MC} ...")
        tau_mc = np.full((N_MC, freqs.size), np.nan, dtype=float)

        for r in range(N_MC):
            xr = x + rng.normal(0.0, xe)
            yr = y + rng.normal(0.0, ye)

            est_r = estimate_cross_spectrum(tx, xr, xe, ty, yr, ye, freqs,
                                            win_ok)
            phi_r = est_r["phi_circular_mean"]
            coh_r = est_r["coh"]
            support_r = est_r["support"]

            good_r = win_ok & (support_r >= S_MIN_SUPPORT) & (coh_r >= COH_MIN)
            phi_unw_r = unwrap_in_islands(phi_r, good_r)
            tau_mc[r, :] = -phi_unw_r / (2.0 * np.pi * freqs)

            if (r + 1) % 50 == 0:
                print(f"[INFO]   MC realisation {r + 1}/{N_MC}")

        tau_med = np.nanmedian(tau_mc, axis=0)
        tau_lo = np.nanpercentile(tau_mc, 16, axis=0)
        tau_hi = np.nanpercentile(tau_mc, 84, axis=0)

        out_mc = OUT_DIR / "nudft_tau_mc_bands.csv"
        pd.DataFrame({
            "f_dayinv": freqs,
            "tau_med": tau_med,
            "tau_p16": tau_lo,
            "tau_p84": tau_hi,
        }).to_csv(out_mc, index=False)
        print(f"[INFO] Saved: {out_mc.resolve()}")

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.loglog(freqs, np.clip(Pw, EPS, np.inf), label=r"$P_W(f)$")
        ax.loglog(freqs, np.clip(Pw_median, EPS, np.inf), ls="--",
                  label=r"local median $\bar{P}_W(f)$")
        ax.set_xlabel(r"Frequency $f$ [day$^{-1}$]")
        ax.set_ylabel(r"Cross-window power $P_W(f)$")
        ax.set_title("Spectral window power and local median")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "window_power.png", dpi=200)
        plt.show()

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.semilogx(freqs, Qw, label=r"$Q_W(f)$")
        ax.axhline(WINDOW_QW_ETA, ls="--", color="r",
                   label=rf"$\eta = {WINDOW_QW_ETA:g}$")
        ax.set_xlabel(r"Frequency $f$ [day$^{-1}$]")
        ax.set_ylabel(r"Window anomaly score $Q_W(f)$")
        ax.set_title("Window contamination score")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "window_Qw.png", dpi=200)
        plt.show()

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.semilogx(freqs, coh, label=r"$\hat{\gamma}^2(f)$")
        ax.axhline(COH_MIN, ls="--", color="r",
                   label=rf"$\gamma^2_{{\min}} = {COH_MIN:g}$")
        ax.set_xlabel(r"Frequency $f$ [day$^{-1}$]")
        ax.set_ylabel(r"Squared coherence $\hat{\gamma}^2(f)$")
        ax.set_title("Coherence spectrum")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "coherence.png", dpi=200)
        plt.show()

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.semilogx(freqs, support, label="support $S(f)$")
        ax.axhline(S_MIN_SUPPORT, ls="--", color="r",
                   label=f"$S_{{\\min}} = {S_MIN_SUPPORT}$")
        ax.set_xlabel(r"Frequency $f$ [day$^{-1}$]")
        ax.set_ylabel("Support count (segments)")
        ax.set_title("Per-frequency segment support")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "support.png", dpi=200)
        plt.show()

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.semilogx(freqs, phi, alpha=0.6, label="raw circular-mean phase")
        ax.semilogx(freqs, phi_unw, label="island-unwrapped phase")
        ax.set_xlabel(r"Frequency $f$ [day$^{-1}$]")
        ax.set_ylabel(r"Phase $\hat{\phi}(f)$ [rad]")
        ax.set_title("Cross-spectrum phase")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "phase.png", dpi=200)
        plt.show()

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        f_good = np.where(good, freqs, np.nan)
        tau_good = np.where(good, tau, np.nan)
        tau_med_good = np.where(good, tau_med, np.nan)
        tau_lo_good = np.where(good, tau_lo, np.nan)
        tau_hi_good = np.where(good, tau_hi, np.nan)

        ax.semilogx(f_good, tau_good, "o-", ms=3, label=r"$\tau(f)$ nominal")
        ax.semilogx(f_good, tau_med_good, "s-", ms=2.5, alpha=0.8,
                    label=r"$\tau(f)$ MC median")
        ax.fill_between(freqs, np.where(good, tau_lo, np.nan),
                         np.where(good, tau_hi, np.nan),
                         alpha=0.25, label="16–84% MC band")
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.set_xlabel(r"Frequency $f$ [day$^{-1}$]")
        ax.set_ylabel(r"Time lag $\tau(f)$ [days]")
        ax.set_title("Lag spectrum (good bins only)")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "tau.png", dpi=200)
        plt.show()

        from scipy.optimize import curve_fit

        tau_std = (tau_hi - tau_lo) / 2.0

        valid = (
            good
            & np.isfinite(freqs)
            & (freqs > 0.0)
            & np.isfinite(tau_med)
            & np.isfinite(tau_std)
            & (tau_std > 0.0)
        )

        if not np.any(valid):
            print("[WARNING] No valid frequency bins for spectral fit. "
                  "Skipping reverberation lag extraction.")
            results_rows.append({
                "filter":         FILTER_2,
                "lag_days":       float("nan"),
                "lag_error_days": float("nan"),
            })
        else:
            idx_valid = np.where(valid)[0]
            idx0 = idx_valid[np.argsort(freqs[idx_valid])][0]
            f0 = float(freqs[idx0])
            tau0 = float(tau_med[idx0])
            err0 = float(tau_std[idx0])

            print(f"\n[FIT] Anchor frequency f0 = {f0:.6g} day^-1")
            print(f"[FIT] tau at f0 = {tau0:.4g} +/- {err0:.4g} days "
                  f"(used as Zphase initial guess)")

            f_fit   = freqs[valid]
            tau_fit = tau_med[valid]
            err_fit = tau_std[valid]

            def _fit_model_1param(f, Ztime_fit):
                return model_tau_wrapped_two_param(f, tau0, Ztime_fit, f0)

            try:
                popt, pcov = curve_fit(
                    _fit_model_1param,
                    f_fit,
                    tau_fit,
                    sigma=err_fit,
                    absolute_sigma=True,
                    p0=[1.0],
                    maxfev=20000,
                )
            except RuntimeError as _e:
                print(f"[WARNING] curve_fit did not converge: {_e}")
                popt = np.array([1.0])
                pcov = np.full((1, 1), np.nan)

            Zphase_fit = tau0
            Ztime_fit  = float(popt[0])
            Zphase_err = float("nan")
            Ztime_err  = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) \
                         else np.nan

            print("\n==============================")
            print("REVERBERATION LAG RESULT")
            print("==============================")
            print(f"  Anchor frequency f0     = {f0:.6g} day^-1")
            print(f"  Zphase (frozen = tau0)  = {Zphase_fit:.6g} days "
                  f"(anchored at f0, not fitted)")
            print(f"  Reverberation lag Ztime = {Ztime_fit:.6g} "
                  f"+/- {Ztime_err:.6g} days")
            print("==============================\n")

            results_rows.append({
                "filter":        FILTER_2,
                "lag_days":      Ztime_fit,
                "lag_error_days": Ztime_err,
            })

            phi0_anchor = 2.0 * np.pi * f0 * tau0
            tau_phase_curve = tau_from_wrapped_phase(
                freqs, wrap_to_pi(phi0_anchor * np.ones_like(freqs))
            )
            tau_time_curve     = constant_time_curve_wrapped(freqs, Ztime_fit)
            tau_combined_curve = model_tau_wrapped_two_param(
                                     freqs, Zphase_fit, Ztime_fit, f0)

            fig_r, ax_r = plt.subplots(figsize=(8.6, 5.2))

            ax_r.errorbar(
                freqs[valid], tau_med[valid],
                yerr=tau_std[valid],
                fmt="o", markersize=3, linewidth=1, capsize=2,
                label=r"Measured $\bar{\tau}(f)$ $\pm$ $\sigma_\tau$",
                zorder=3,
            )

            ax_r.plot(
                freqs[valid], tau_phase_curve[valid],
                linestyle="--", linewidth=1.8,
                label=(
                    rf"Constant phase: $\tau(f_0) = {tau0:.3g}$ d, "
                    rf"$\phi_0 = 2\pi f_0 \tau_0$ (anchored at $f_0 = {f0:.4g}$ d$^{{-1}}$)"
                ),
                zorder=1,
            )

            ax_r.plot(
                freqs[valid], tau_time_curve[valid],
                linestyle="-", linewidth=1.8,
                label=(
                    rf"Constant time lag: $Z_t = {Ztime_fit:.4g}$ d"
                ),
                zorder=1,
            )

            ax_r.plot(
                freqs[valid], tau_combined_curve[valid],
                linestyle="-", linewidth=2.8,
                label="Combined best fit",
                zorder=4,
            )

            ax_r.axhline(0.0, ls="--", color="gray", lw=0.8)

            result_str = (
                rf"$Z_t = {Ztime_fit:.4g} \pm {Ztime_err:.4g}$ days"
            )
            ax_r.text(
                0.97, 0.05, result_str,
                transform=ax_r.transAxes,
                ha="right", va="bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8),
            )

            ax_r.set_xscale("log")
            ax_r.set_xlabel(r"Frequency $f$ [day$^{-1}$]")
            ax_r.set_ylabel(r"Time lag $\tau(f)$ [days]")
            ax_r.set_title(
                f"Reverberation lag: {FILTER_2} relative to {FILTER_1}"
            )
            ax_r.grid(True, which="both", ls="--", alpha=0.3)
            ax_r.legend(fontsize=8)
            fig_r.tight_layout()

            result_path = Path(f"nudft_result_{FILTER_2}.png")
            fig_r.savefig(result_path, dpi=300)
            print(f"[INFO] Final result figure saved: {result_path.resolve()}")
            plt.show()

    summary_path = Path("nudft_lag_results.csv")
    pd.DataFrame(results_rows).to_csv(summary_path, index=False)
    print(f"\n[INFO] Lag summary saved: {summary_path.resolve()}")
    print(pd.DataFrame(results_rows).to_string(index=False))
    print(f"\n[DONE] All outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
