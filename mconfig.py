"""Configuration for the myelin labeling app.

Self-contained (no dependency on the myelin_af package) so the app can run
anywhere, including Streamlit Cloud, given access to the Xenium image dirs.
"""
from __future__ import annotations
import os

# --- data locations ------------------------------------------------------
# directories scanned for Xenium `output-XETG...` section dirs.
# override with env MYELIN_DATA_DIRS (colon-separated) for other machines.
_DEFAULT_DATA_DIRS = [
    "/home/yzy21/yy/cidp/xenium_images",
    "/home/yzy21/yy/cidp/myelin_af/dt",
]
DATA_DIRS = [d for d in os.environ.get("MYELIN_DATA_DIRS", ":".join(_DEFAULT_DATA_DIRS)).split(":") if d]

# where masks/labels are written locally before (optionally) pushing to GitHub
OUT_DIR = os.environ.get(
    "MYELIN_LABELER_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels"))
os.makedirs(OUT_DIR, exist_ok=True)

# --- imaging -------------------------------------------------------------
CH = {"dapi": 0, "blue": 4, "green": 5, "yellow": 6, "red": 7}
PX = {0: 0.2125, 1: 0.425, 2: 0.85}
PX0 = PX[0]
W_BLUE = 0.6                    # corrected myelin index: clip(red - W_BLUE*blue, 0)
MIN_AF_CHANNELS = 8            # protein-panel stacks carry AF background channels

MYELIN_GENES = ["MPZ", "MBP", "PRX", "PMP22", "PLP1", "CNP", "DRP2", "PMP2"]
QV_MIN = 20

# --- GitHub defaults (token comes from st.secrets, never hard-coded) -----
GH_REPO_DEFAULT = os.environ.get("MYELIN_GH_REPO", "")     # "owner/name"
GH_BRANCH_DEFAULT = os.environ.get("MYELIN_GH_BRANCH", "main")
GH_PATH_PREFIX = os.environ.get("MYELIN_GH_PREFIX", "myelin_labels")  # dir in repo
