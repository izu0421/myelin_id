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
#
# On Streamlit Cloud the MediaFileManager `/media/...` URL is redirected to an
# auth endpoint and CORS-blocked inside the component iframe, so the canvas
# background never loads. When a public URL for the image is available (our
# bundle crops live in a public GitHub repo) we return that directly instead —
# raw.githubusercontent.com serves `Access-Control-Allow-Origin: *`, so fabric
# can load it cross-origin. Otherwise we fall back to the media URL (fine locally).
import streamlit.elements.image as _st_image  # noqa: E402

# set by the app just before calling st_canvas to force a specific background URL
_CANVAS_BG_URL: str | None = None
_orig_image_to_url = getattr(_st_image, "image_to_url", None)


def _image_to_url(image, width=None, clamp=False, channels="RGB",
                  output_format="PNG", image_id="", *args, **kwargs):
    # Force a public URL for the canvas background when the app asks for it
    # (bundle crops on raw.githubusercontent.com are CORS-friendly; the Cloud
    # /media/ URL is auth-gated + CORS-blocked inside the component iframe).
    if _CANVAS_BG_URL:
        return _CANVAS_BG_URL
    if _orig_image_to_url is not None:
        try:
            return _orig_image_to_url(image, width, clamp, channels,
                                      output_format, image_id, *args, **kwargs)
        except Exception:
            pass
    import io as _io
    if isinstance(image, np.ndarray):
        arr = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    buf = _io.BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()
    try:
        from streamlit.runtime import Runtime, caching
        mfm = Runtime.instance().media_file_mgr
        url = mfm.add(data, "image/png", str(image_id))
        caching.save_media_data(data, "image/png", str(image_id))
        return url
    except Exception:
        import base64 as _b64
        return "data:image/png;base64," + _b64.b64encode(data).decode("ascii")


# always install our wrapper so the _CANVAS_BG_URL override works regardless of
# which Streamlit version is present (older ones still ship image_to_url).
_st_image.image_to_url = _image_to_url

from streamlit_drawable_canvas import st_canvas  # noqa: E402

import mconfig as M
import imaging_io as IO
import gh

st.set_page_config(page_title="Myelin ring labeler", layout="wide")
Z = 1000                       # crop size at level 0 (~213 um)
DISP = 700                     # canvas display size (px)


BUNDLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bundle")


# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_sections() -> dict:
    return IO.af_sections()


@st.cache_data(show_spinner=False)
def get_bundle() -> dict:
    """{sid: [crop_meta, ...]} from the pre-exported bundle, if present."""
    mpath = os.path.join(BUNDLE_DIR, "manifest.json")
    if not os.path.exists(mpath):
        return {}
    with open(mpath) as f:
        rows = json.load(f)
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r["sample_id"], []).append(r)
    return out


@st.cache_data(show_spinner=True)
def load_bundle_crop(crop_id: str):
    """Return the grayscale display image for a pre-exported crop."""
    p = os.path.join(BUNDLE_DIR, "crops", f"{crop_id}.png")
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


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


def rgb_display(disp: np.ndarray) -> Image.Image:
    rgb = (np.dstack([disp, disp, disp]) * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((DISP, DISP), Image.BILINEAR)


# ----------------------------------------------------------------------------
st.title("🧫 Myelin ring labeler")
st.caption("Trace each myelin sheath (bright red-dominant index), press **Ring labelled**, "
           "repeat. Masks + labels are saved locally and pushed to GitHub.")

secs = get_sections()
bundle = get_bundle()
BUNDLED = not secs and bool(bundle)
if not secs and not bundle:
    st.error("No AF-capable Xenium sections found (set MYELIN_DATA_DIRS) and no "
             "pre-exported bundle/ present. Run export_crops.py where the raw data lives.")
    with st.expander("Diagnostics"):
        st.write("BUNDLE_DIR:", BUNDLE_DIR)
        st.write("manifest exists:", os.path.exists(os.path.join(BUNDLE_DIR, "manifest.json")))
        try:
            st.write("bundle dir contents:", os.listdir(BUNDLE_DIR))
        except Exception as e:
            st.write("listdir error:", repr(e))
        st.write("MYELIN_DATA_DIRS:", M.DATA_DIRS)
    if st.button("🔄 Reload / clear cache"):
        st.cache_data.clear()
        st.rerun()
    st.stop()
if BUNDLED:
    st.info("Running from pre-exported crops (bundle/). Raw Xenium stacks not on this "
            "machine, so crop location is fixed to the exported fields.")

with st.sidebar:
    st.header("1 · Pick image")
    section_ids = sorted(bundle) if BUNDLED else sorted(secs)
    sid = st.selectbox("Section", section_ids, index=0)

    st.header("2 · Choose crop")
    if BUNDLED:
        crops = sorted(bundle[sid], key=lambda r: -r["n_transcripts"])
        labels = [f"{c['crop_id']}  (~{c['n_transcripts']} tx)" for c in crops]
        pick = st.selectbox("Crop (myelin-dense fields)", range(len(crops)),
                            format_func=lambda i: labels[i])
        cmeta = crops[pick]
        sample_dir, slide = cmeta["sample_dir"], cmeta["slide"]
        y0, x0 = cmeta["y0"], cmeta["x0"]
        crop_id = cmeta["crop_id"]
        st.write(f"slide **{slide}**")
    else:
        sample_dir, dapi_path = secs[sid]
        base_dir = os.path.dirname(os.path.dirname(dapi_path))  # section dir
        slide = IO.slide_of(sample_dir)
        H0, W0 = IO.level_shape(dapi_path, 0)
        st.write(f"slide **{slide}** · {W0}×{H0} px")

        xy_all, grid = transcript_field(base_dir, H0, W0)
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

if BUNDLED:
    disp = load_bundle_crop(crop_id)
else:
    ch, tissue, idxn, disp = load_crop(dapi_path, y0, x0)
bg = rgb_display(disp)

# per-crop annotation state (reset when the crop changes)
if st.session_state.get("crop_key") != crop_id:
    st.session_state.crop_key = crop_id
    st.session_state.rings = []
    st.session_state.negs = []
    st.session_state.cseq = st.session_state.get("cseq", 0) + 1

st.subheader(f"Crop: {crop_id}")
tc1, tc2 = st.columns([1, 2])
with tc1:
    tool = st.radio("Tool", ["ring (freehand)", "negative (point)"], horizontal=True)
    stroke = st.slider("Stroke width", 1, 6, 3)
with tc2:
    st.markdown(
        "1. **ring**: drag a freehand loop around **one** sheath, then press "
        "**✅ Ring labelled** — it stores the ring and clears the canvas.\n"
        "2. **negative**: click non-myelin spots, then press **➕ Add negatives**.")

# On Streamlit Cloud the /media/ background URL is auth-gated + CORS-blocked
# inside the canvas iframe. In bundled mode the crop is public on GitHub, so
# point the canvas background at the raw URL (CORS-friendly) instead.
#
# The canvas lib builds the final URL as `server.baseUrlPath + image_to_url(...)`.
# On Cloud baseUrlPath is non-empty and corrupts our absolute https:// URL, so we
# force that option to "" for the duration of the st_canvas call.
_CANVAS_BG_URL = f"{M.BUNDLE_URL_PREFIX}/{crop_id}.png" if BUNDLED else None

_saved_base = st._config.get_option("server.baseUrlPath")
if _CANVAS_BG_URL:
    try:
        st._config.set_option("server.baseUrlPath", "")
    except Exception:
        pass
try:
    canvas = st_canvas(
        background_image=bg,
        drawing_mode="freedraw" if tool.startswith("ring") else "point",
        stroke_width=stroke, stroke_color="#FFE94A",
        fill_color="rgba(255,233,74,0.25)", point_display_radius=3,
        update_streamlit=True, height=DISP, width=DISP,
        key=f"canvas_{crop_id}_{st.session_state.cseq}",
    )
finally:
    if _CANVAS_BG_URL:
        try:
            st._config.set_option("server.baseUrlPath", _saved_base)
        except Exception:
            pass
    _CANVAS_BG_URL = None


# ---- rasterize canvas -> instance mask + negatives -------------------------
def parse_canvas(canvas_result):
    rings, negs = [], []
    if canvas_result is None or canvas_result.json_data is None:
        return rings, negs
    scale = Z / DISP
    for obj in canvas_result.json_data.get("objects", []):
        otype = obj.get("type")
        if otype in ("path", "polygon", "polyline"):
            verts = []
            if otype == "polygon" or otype == "polyline":
                for p in obj.get("points", []):
                    verts.append((p["x"] * scale, p["y"] * scale))
            else:
                # fabric freedraw path: list of SVG segments; endpoint is the
                # last two numbers of each segment (M x y / Q cx cy x y / L x y).
                for seg in obj.get("path", []):
                    nums = [v for v in seg[1:] if isinstance(v, (int, float))]
                    if len(nums) >= 2:
                        verts.append((nums[-2] * scale, nums[-1] * scale))
            if len(verts) >= 3:
                rings.append(verts)
        elif otype == "circle":
            r = obj.get("radius", 0)
            negs.append(((obj.get("left", 0) + r) * scale,
                         (obj.get("top", 0) + r) * scale))
    return rings, negs


cur_rings, cur_negs = parse_canvas(canvas)

bcol = st.columns(4)
if bcol[0].button(f"✅ Ring labelled  ({len(cur_rings)} on canvas)",
                  type="primary", disabled=len(cur_rings) == 0):
    st.session_state.rings.extend(cur_rings)
    st.session_state.cseq += 1
    st.rerun()
if bcol[1].button(f"➕ Add negatives  ({len(cur_negs)})",
                  disabled=len(cur_negs) == 0):
    st.session_state.negs.extend(cur_negs)
    st.session_state.cseq += 1
    st.rerun()
if bcol[2].button("↩ Undo last ring", disabled=len(st.session_state.rings) == 0):
    st.session_state.rings.pop()
    st.rerun()
if bcol[3].button("🗑 Clear all",
                  disabled=not (st.session_state.rings or st.session_state.negs)):
    st.session_state.rings, st.session_state.negs = [], []
    st.session_state.cseq += 1
    st.rerun()

rings = st.session_state.rings
negs = st.session_state.negs
st.write(f"**{len(rings)} rings labelled** · **{len(negs)} negatives**  "
         f"(on canvas now: {len(cur_rings)} ring / {len(cur_negs)} pts)")


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
