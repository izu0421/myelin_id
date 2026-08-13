# Myelin ring labeler (Streamlit)

A web app version of `label_myelin.ipynb` for labelling myelin sheaths across
**many** Xenium autofluorescence crops, saving the segmentation masks back to
**GitHub**.

## What it does

- Discovers AF-capable Xenium sections on disk (`MYELIN_DATA_DIRS`).
- For a chosen section, auto-suggests **myelin-transcript-dense** crops (or pick
  a location manually).
- Shows the corrected myelin index (`red - 0.6*blue`) with myelin-gene
  transcripts overlaid (magenta) as a labelling guide.
- **Draw** ring outlines (polygon tool) and drop **negative** points; the app
  rasterises rings into an instance mask.
- **Saves** per crop: `masks/<crop>_mask.png` (16-bit instance labels),
  `polygons/<crop>_polygons.json`, and appends a row to `index.csv` — locally
  and, if enabled, committed to a GitHub repo via the Contents API.

## Setup

```bash
cd /home/yzy21/yy/cidp/myelin_af/labeler
micromamba run -n cellpose pip install -r requirements.txt   # already installed in `cellpose`
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your GitHub PAT
```

`secrets.toml`:
```toml
[github]
token = "ghp_..."   # PAT with contents:read/write on the target repo
```

## Run

```bash
micromamba run -n cellpose streamlit run app.py
```

Then in the sidebar set the target **repo (owner/name)**, **branch**, and **path
prefix** (default `myelin_labels`). Masks land under `<prefix>/masks/` in the repo.

## Config (env overrides)

| var | default |
|---|---|
| `MYELIN_DATA_DIRS` | `xenium_images:myelin_af/dt` (colon-separated) |
| `MYELIN_GH_REPO` | (empty; set in UI) |
| `MYELIN_GH_BRANCH` | `main` |
| `MYELIN_GH_PREFIX` | `myelin_labels` |
| `MYELIN_LABELER_OUT` | `labeler/labels` |

## Notes

- The token is read only from `st.secrets` and never written to disk by the app.
- Masks are 16-bit PNGs (one integer label per ring); reload with
  `np.array(PIL.Image.open(...))`.
