#!/usr/bin/env python3
"""Streamlit app: label myelin rings across many Xenium AF crops and save the
segmentation masks (and a labels index) back to GitHub.

Run:
    cd /home/yzy21/yy/cidp/myelin_af/labeler
    streamlit run app.py

A GitHub PAT with repo write access must be provided in .streamlit/secrets.toml:
    [github]
    token = "ghp_..."
"""
from __future__ import annotations
import io
import json
import os
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.draw import polygon as sk_polygon

# --- compat shim: streamlit-drawable-canvas 0.9.3 imports
# `streamlit.elements.image.image_to_url`, removed in Streamlit >= 1.60.
# Provide a self-contained data-URL implementation before importing the canvas.
import streamlit.elements.image as _st_image  # noqa: E402

if not hasattr(_st_image, "image_to_url"):
    import base64 as _b64
    import io as _io

    def image_to_url(image, width=None, clamp=False, channels="RGB",
                     output_format="PNG", image_id="", *args, **kwargs):
        if isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr)
        buf = _io.BytesIO()
        image.save(buf, format="PNG")
        data = _b64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{data}"

    _st_image.image_to_url = image_to_url

from streamlit_drawable_canvas import st_canvas  # noqa: E402

import mconfig as M
import imaging_io as IO
import gh

st.set_page_config(page_title="Myelin ring labeler", layout="wide")
Z = 1000                       # crop size at level 0 (~213 um)
DISP = 700                     # canvas display size (px)


# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_sections() -> dict:
    return IO.af_sections()


@st.cache_data(show_spinner=True)
def transcript_field(sample_dir_full: str, H0: int, W0: int):
    """All myelin transcripts + a coarse per-crop count grid for auto-suggest."""
    xy = IO.myelin_transcripts(sample_dir_full)
    if len(xy) == 0:
        return xy, None
    hy = np.arange(0, H0 + Z, Z)
    hx = np.arange(0, W0 + Z, Z)
    grid, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[hy, hx])
    return xy, grid


@st.cache_data(show_spinner=True)
def load_crop(dapi_path: str, y0: int, x0: int):
    ch = IO.read_channels(dapi_path, 0, y0, x0, Z, Z)
    tissue = IO.tissue_mask(ch)
    idxn = IO.pnorm(IO.myelin_index(ch, tissue=tissue), mask=tissue)
    from skimage import exposure
    disp = exposure.equalize_adapthist(np.clip(idxn, 0, 1), clip_limit=0.02)
    return ch, tissue, idxn, disp


def _read_repo_csv(repo, branch, path, token):
    import base64
    from urllib.parse import quote
    if "/" not in repo:
        return None
    owner, name = repo.split("/", 1)
    url = (f"https://api.github.com/repos/{owner.strip()}/{name.strip()}"
           f"/contents/{quote(path)}?ref={quote(branch)}")
    status, data = gh._request(url, token, "GET")
    if status == 200 and data.get("content"):
        try:
            return pd.read_csv(io.StringIO(base64.b64decode(data["content"]).decode("utf-8")))
        except Exception:
            return None
    return None


def rgb_display(disp: np.ndarray, xy_local: np.ndarray) -> Image.Image:
    rgb = (np.dstack([disp, disp, disp]) * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((DISP, DISP), Image.BILINEAR)
    # draw transcript dots as magenta pixels (downsampled)
    if len(xy_local):
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        s = DISP / Z
        for x, y in xy_local:
            px, py = int(x * s), int(y * s)
            d.ellipse([px - 1, py - 1, px + 1, py + 1], fill=(255, 55, 208))
    return img


# ----------------------------------------------------------------------------
st.title("🧫 Myelin ring labeler")
st.caption("Trace myelin sheaths (red-dominant index) guided by myelin-gene transcripts (magenta). "
           "Masks + labels are saved locally and pushed to GitHub.")

secs = get_sections()
if not secs:
    st.error("No AF-capable Xenium sections found. Set MYELIN_DATA_DIRS.")
    st.stop()

with st.sidebar:
    st.header("1 · Pick image")
    sid = st.selectbox("Section", sorted(secs), index=0)
    sample_dir, dapi_path = secs[sid]
    base_dir = os.path.dirname(os.path.dirname(dapi_path))  # section dir
    slide = IO.slide_of(sample_dir)
    H0, W0 = IO.level_shape(dapi_path, 0)
    st.write(f"slide **{slide}** · {W0}×{H0} px")

    xy_all, grid = transcript_field(base_dir, H0, W0)
    st.header("2 · Choose crop")
    mode = st.radio("Locate crop", ["Auto (myelin-dense)", "Manual"], horizontal=True)
    if mode.startswith("Auto") and grid is not None:
        order = np.dstack(np.unravel_index(np.argsort(grid.ravel())[::-1], grid.shape))[0]
        rank = st.number_input("Field rank (0 = densest)", 0, max(len(order) - 1, 0), 0)
        by, bx = order[int(rank)]
        y0 = int(np.clip(by * Z, 0, max(H0 - Z, 0)))
        x0 = int(np.clip(bx * Z, 0, max(W0 - Z, 0)))
        st.caption(f"~{int(grid[by, bx])} myelin transcripts in this field")
    else:
        y0 = st.number_input("y0", 0, max(H0 - Z, 0), 0, step=Z // 2)
        x0 = st.number_input("x0", 0, max(W0 - Z, 0), 0, step=Z // 2)
    crop_id = f"{sid}_y{y0}_x{x0}"

    st.header("3 · Save target")
    repo = st.text_input("GitHub repo (owner/name)", M.GH_REPO_DEFAULT)
    branch = st.text_input("Branch", M.GH_BRANCH_DEFAULT)
    prefix = st.text_input("Path prefix in repo", M.GH_PATH_PREFIX)
    token_ok = bool(gh.get_token())
    st.caption("token from secrets: " + ("✅ found" if token_ok else "❌ missing"))

ch, tissue, idxn, disp = load_crop(dapi_path, y0, x0)
xy_local = xy_all[(xy_all[:, 0] >= x0) & (xy_all[:, 0] < x0 + Z) &
                  (xy_all[:, 1] >= y0) & (xy_all[:, 1] < y0 + Z)] - [x0, y0] if len(xy_all) else np.empty((0, 2))
bg = rgb_display(disp, xy_local)

st.subheader(f"Crop: {crop_id}  ·  {len(xy_local)} myelin transcripts")
c1, c2 = st.columns([3, 1])
with c2:
    tool = st.radio("Tool", ["polygon (rings)", "point (negatives)"])
    stroke = st.slider("Stroke width", 1, 5, 2)
    st.markdown("- **polygon**: click to add points, close to finish a ring\n"
                "- **point**: click non-myelin spots (collagen/background)")

with c1:
    canvas = st_canvas(
        background_image=bg,
        drawing_mode="polygon" if tool.startswith("polygon") else "point",
        stroke_width=stroke, stroke_color="#FFE94A",
        fill_color="rgba(255,233,74,0.25)", point_display_radius=3,
        update_streamlit=True, height=DISP, width=DISP, key=f"canvas_{crop_id}",
    )


# ---- rasterize canvas -> instance mask + negatives -------------------------
def parse_canvas(canvas_result):
    rings, negs = [], []
    if canvas_result is None or canvas_result.json_data is None:
        return rings, negs
    scale = Z / DISP
    for obj in canvas_result.json_data.get("objects", []):
        if obj.get("type") == "path" or obj.get("type") == "polygon":
            pts = obj.get("path") or obj.get("points")
            verts = []
            if obj.get("type") == "polygon":
                left, top = obj.get("left", 0), obj.get("top", 0)
                for p in obj.get("points", []):
                    verts.append(((p["x"]) * scale, (p["y"]) * scale))
            else:
                for seg in pts:
                    if len(seg) >= 3:
                        verts.append((seg[1] * scale, seg[2] * scale))
            if len(verts) >= 3:
                rings.append(verts)
        elif obj.get("type") == "circle":
            negs.append((obj.get("left", 0) * scale, obj.get("top", 0) * scale))
    return rings, negs


rings, negs = parse_canvas(canvas)
st.write(f"**{len(rings)} rings**, **{len(negs)} negatives** drawn")


def build_mask(rings):
    m = np.zeros((Z, Z), np.int32)
    for i, verts in enumerate(rings, 1):
        a = np.array(verts)
        rr, cc = sk_polygon(a[:, 1], a[:, 0], shape=(Z, Z))
        m[rr, cc] = i
    return m


# ---- save ------------------------------------------------------------------
st.subheader("Save")
save_local = st.checkbox("Save locally", True)
push_gh = st.checkbox("Push masks + index to GitHub", token_ok)
if st.button("💾 Save this crop", type="primary", disabled=(len(rings) == 0)):
    mask = build_mask(rings)
    meta = dict(sample_id=sid, sample_dir=sample_dir, slide=slide, level=0,
                px_um=M.PX0, y0=y0, x0=x0, size=Z, n_rings=int(mask.max()),
                n_negatives=len(negs), annotated_at=dt.datetime.utcnow().isoformat())
    # encode mask as PNG (16-bit instance labels) + negatives/meta json
    mask_png = io.BytesIO()
    Image.fromarray(mask.astype(np.uint16)).save(mask_png, format="PNG")
    mask_bytes = mask_png.getvalue()
    poly = json.dumps({"meta": meta, "rings": rings, "negatives": negs}, indent=1)

    stem = crop_id
    if save_local:
        os.makedirs(M.OUT_DIR, exist_ok=True)
        with open(os.path.join(M.OUT_DIR, f"{stem}_mask.png"), "wb") as f:
            f.write(mask_bytes)
        with open(os.path.join(M.OUT_DIR, f"{stem}_polygons.json"), "w") as f:
            f.write(poly)
        st.success(f"saved locally to {M.OUT_DIR}/{stem}_*")

    if push_gh:
        token = gh.get_token()
        msgs = []
        ok1, m1 = gh.commit_bytes(mask_bytes, repo, branch,
                                  f"{prefix}/masks/{stem}_mask.png", token,
                                  f"myelin mask {stem}")
        ok2, m2 = gh.commit_text(poly, repo, branch,
                                 f"{prefix}/polygons/{stem}_polygons.json", token,
                                 f"myelin polygons {stem}")
        # append to a shared index CSV in the repo
        row = pd.DataFrame([meta])
        idx_path = f"{prefix}/index.csv"
        prev = _read_repo_csv(repo, branch, idx_path, token)
        allrows = pd.concat([prev, row], ignore_index=True) if prev is not None else row
        allrows = allrows.drop_duplicates(subset=["sample_id", "y0", "x0"], keep="last")
        ok3, m3 = gh.commit_text(allrows.to_csv(index=False), repo, branch, idx_path,
                                 token, f"index += {stem}")
        for ok, msg in [(ok1, m1), (ok2, m2), (ok3, m3)]:
            (st.success if ok else st.error)(msg)
