#!/usr/bin/env python3
"""Pre-export compact myelin-index crops so the labeler runs without raw TIFFs.

For every AF-capable section it picks the top myelin-transcript-dense fields and
writes each as a full-resolution grayscale PNG (the myelin index the app draws
on) plus a manifest.json with the crop provenance (section, slide, y0, x0, size,
px_um). Committed into the repo, these let the app run on Streamlit Cloud / any
machine with no access to the multi-GB Xenium stacks.

Run (this machine, where the raw data lives):
    micromamba run -n cellpose python export_crops.py --per-section 7
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
from PIL import Image
from skimage import exposure

import mconfig as M
import imaging_io as IO

Z = 1000  # crop size at level 0 (matches app.py)


def display_crop(dapi_path: str, y0: int, x0: int) -> np.ndarray:
    ch = IO.read_channels(dapi_path, 0, y0, x0, Z, Z)
    tissue = IO.tissue_mask(ch)
    idxn = IO.pnorm(IO.myelin_index(ch, tissue=tissue), mask=tissue)
    disp = exposure.equalize_adapthist(np.clip(idxn, 0, 1), clip_limit=0.02)
    return (disp * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-section", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "bundle"))
    ap.add_argument("--only", default="", help="comma-separated section ids (default: all)")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "crops")
    os.makedirs(img_dir, exist_ok=True)
    secs = IO.af_sections()
    want = set(s for s in args.only.split(",") if s) or set(secs)

    manifest = []
    for sid in sorted(secs):
        if sid not in want:
            continue
        sample_dir, dapi = secs[sid]
        base = os.path.dirname(os.path.dirname(dapi))
        slide = IO.slide_of(sample_dir)
        H0, W0 = IO.level_shape(dapi, 0)
        xy = IO.myelin_transcripts(base)
        if len(xy) == 0:
            print(f"{sid}: no myelin transcripts, skipping")
            continue
        hy = np.arange(0, H0 + Z, Z)
        hx = np.arange(0, W0 + Z, Z)
        grid, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[hy, hx])
        order = np.dstack(np.unravel_index(np.argsort(grid.ravel())[::-1], grid.shape))[0]

        n = 0
        for by, bx in order:
            if n >= args.per_section or grid[by, bx] <= 0:
                break
            y0 = int(np.clip(by * Z, 0, max(H0 - Z, 0)))
            x0 = int(np.clip(bx * Z, 0, max(W0 - Z, 0)))
            crop_id = f"{sid}_y{y0}_x{x0}"
            png = display_crop(dapi, y0, x0)
            Image.fromarray(png).save(os.path.join(img_dir, f"{crop_id}.png"),
                                      optimize=True)
            manifest.append(dict(crop_id=crop_id, sample_id=sid, sample_dir=sample_dir,
                                 slide=slide, y0=y0, x0=x0, size=Z, px_um=M.PX0,
                                 n_transcripts=int(grid[by, bx])))
            n += 1
        print(f"{sid}: exported {n} crops")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n{len(manifest)} crops -> {args.out}")


if __name__ == "__main__":
    main()
