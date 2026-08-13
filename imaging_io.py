"""Imaging helpers for the labeler app: discover AF sections, read crops,
build the myelin index, and load myelin-gene transcripts for guidance.

Self-contained (only numpy/tifffile/pyarrow) so the app has no heavy deps.
"""
from __future__ import annotations
import glob
import os
import numpy as np
import tifffile

import mconfig as M


# --- section discovery ---------------------------------------------------
def _af_path(sample_dir_full: str) -> str | None:
    p = os.path.join(sample_dir_full, "morphology_focus", "ch0000_dapi.ome.tif")
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    try:
        with tifffile.TiffFile(p) as t:
            nch = t.series[0].shape[0]
    except Exception:
        return None
    return p if nch >= M.MIN_AF_CHANNELS else None


def af_sections() -> dict[str, tuple[str, str]]:
    """{sample_id: (sample_dir, dapi_path)} for AF-capable sections on disk."""
    out: dict[str, tuple[str, str]] = {}
    for base in M.DATA_DIRS:
        if not os.path.isdir(base):
            continue
        for sd_full in sorted(glob.glob(os.path.join(base, "output-XETG*"))):
            if "/backup_dt/" in sd_full:
                continue
            sample_dir = os.path.basename(sd_full)
            sid = sample_dir.split("__")[2]
            if sid in out:
                continue
            p = _af_path(sd_full)
            if p is not None:
                out[sid] = (sample_dir, p)
    return out


def slide_of(sample_dir: str) -> str:
    return sample_dir.split("__")[1]


# --- reading + index -----------------------------------------------------
def level_shape(path: str, level: int) -> tuple[int, int]:
    with tifffile.TiffFile(path) as t:
        lv = t.series[0].levels[level]
        return int(lv.shape[1]), int(lv.shape[2])


def read_channels(path: str, level: int, y0: int, x0: int, h: int, w: int) -> dict:
    out = {}
    with tifffile.TiffFile(path) as t:
        lv = t.series[0].levels[level]
        for name, ci in M.CH.items():
            out[name] = lv.asarray(key=ci)[y0:y0 + h, x0:x0 + w].astype(np.float32)
    return out


def pnorm(a: np.ndarray, lo: float = 2, hi: float = 99.7, mask=None) -> np.ndarray:
    ref = a[mask] if mask is not None else a
    if ref.size == 0:
        ref = a
    p, q = np.percentile(ref, [lo, hi])
    return np.clip((a - p) / (q - p + 1e-6), 0, 1)


def tissue_mask(ch: dict) -> np.ndarray:
    from skimage.filters import threshold_otsu
    from scipy import ndimage as ndi
    s = sum(ch[k] for k in ("blue", "green", "yellow", "red"))
    s = ndi.gaussian_filter(s, 2.0)
    m = s > threshold_otsu(s)
    return ndi.binary_fill_holes(m)


def myelin_index(ch: dict, tissue=None) -> np.ndarray:
    """Corrected myelin index: clip(red - W_BLUE*blue, 0) on tissue-normed channels."""
    r = pnorm(ch["red"], mask=tissue)
    b = pnorm(ch["blue"], mask=tissue)
    return np.clip(r - M.W_BLUE * b, 0, None)


# --- transcripts (myelin ground-truth guide) -----------------------------
def myelin_transcripts(sample_dir_full: str) -> np.ndarray:
    """Return (N,2) array of myelin-gene transcript pixel coords at level 0.

    Reads transcripts.parquet from the section dir. Empty array if unavailable.
    """
    import pyarrow.parquet as pq
    p = os.path.join(sample_dir_full, "transcripts.parquet")
    if not os.path.exists(p):
        return np.empty((0, 2), np.float32)
    tx = pq.read_table(p, columns=["feature_name", "x_location", "y_location", "qv"]).to_pandas()
    tx = tx[(tx.qv >= M.QV_MIN) & tx.feature_name.isin(M.MYELIN_GENES)]
    xy = np.stack([tx.x_location.values / M.PX0, tx.y_location.values / M.PX0], axis=1)
    return xy.astype(np.float32)
