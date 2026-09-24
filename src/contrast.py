"""Local-contrast masking, replacing the within-shell global percentile rule.

WHY THE OLD RULE FAILED

pc.shell_threshold_mask keeps the slowest N per cent of each depth shell. Because
the transition zone carries a strong global population of slow anomalies, the p1
cut there sits near -2.7 %, against -1.4 % at 1500 km. A conduit comfortably
inside the slowest one per cent of the lower mantle is nowhere near it in the
transition zone, so every column is severed at 400-660 km and no conduit weaker
than about -3 % is ever recovered (see out/hawaii/DETECTION_LIMIT.md).

THE REPLACEMENT

Within each shell, compare the anomaly to a laterally smoothed version of the
same shell rather than to the shell's own global distribution:

    c(x) = a(x) - smooth(a, sigma)(x)

so a conduit competes with its own surroundings. Thresholding c at a percentile
within each shell then means "locally slow", which is what a conduit is, rather
than "globally slow", which is what a large province is. sigma sets the length
scale above which structure is treated as background: it must be larger than the
conduit and smaller than the provinces.

Smoothing is done on the sphere by a longitude-periodic Gaussian filter with the
latitude kernel widened by 1/cos(lat) so the smoothing length is roughly constant
in kilometres. NaNs are handled by normalised convolution.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d

R_E = 6371.0


def _smooth_shell(s, lat, dlat_deg, dlon_deg, sigma_km):
    """Approximately isotropic Gaussian smoothing of one shell, in km."""
    f = np.isfinite(s)
    a = np.where(f, s, 0.0).astype(np.float32)
    w = f.astype(np.float32)
    sj = sigma_km / (dlat_deg * np.pi * R_E / 180.0)
    a = gaussian_filter1d(a, sj, axis=0, mode='nearest')
    w = gaussian_filter1d(w, sj, axis=0, mode='nearest')
    # longitude: kernel widens toward the poles so the km-scale stays constant
    coslat = np.clip(np.cos(np.radians(lat)), 0.02, None)
    km_per_deg_lon = dlon_deg * np.pi * R_E / 180.0
    out_a = np.empty_like(a)
    out_w = np.empty_like(w)
    for j in range(a.shape[0]):
        si = sigma_km / (km_per_deg_lon * coslat[j])
        si = float(min(si, a.shape[1] / 6.0))
        out_a[j] = gaussian_filter1d(a[j], si, mode='wrap')
        out_w[j] = gaussian_filter1d(w[j], si, mode='wrap')
    with np.errstate(invalid='ignore', divide='ignore'):
        r = out_a / out_w
    return np.where(out_w > 1e-3, r, np.nan)


def contrast_field(arr, lat, lon, sigma_km=1200.0):
    """anomaly minus its laterally smoothed background, shell by shell."""
    dlat = float(abs(lat[1] - lat[0])); dlon = float(abs(lon[1] - lon[0]))
    out = np.empty_like(arr, dtype=np.float32)
    for k in range(arr.shape[0]):
        out[k] = arr[k] - _smooth_shell(arr[k], lat, dlat, dlon, sigma_km)
    return out


def contrast_mask(cfield, pct):
    """Slowest `pct` per cent of the CONTRAST field within each shell."""
    m = np.empty(cfield.shape, bool)
    for k in range(cfield.shape[0]):
        s = cfield[k]
        f = np.isfinite(s)
        if not f.any():
            m[k] = False
            continue
        m[k] = f & (s <= np.percentile(s[f], pct))
    return m
