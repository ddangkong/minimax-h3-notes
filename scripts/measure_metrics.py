# -*- coding: utf-8 -*-
"""Recompute the two image metrics used in the H3 writeups, with the definitions stated.

The numbers in SETTINGS.md / WORKFLOW.md were computed ad hoc in an earlier session and only
the results survived, not the method. That is a real weakness for a measurement writeup: a
reader cannot tell what "sharpness 20.45" or "saturation 16.6" actually measured. This script
fixes that by defining both metrics explicitly and recomputing them from the same video files,
so the published figures can either be confirmed or corrected.

Definitions
-----------
sharpness   variance of the Laplacian (3x3 kernel, CV_64F) of the greyscale frame,
            averaged over the sampled frames. Higher = more high-frequency detail.
            This is the standard blur-detection statistic; it is scale-dependent, which
            is exactly why frames must be compared at native resolution.

saturation  mean of the S channel of HSV (0-255), averaged over the sampled frames.

Frames are sampled every STRIDE frames to keep a full run to a few seconds; every measurement
below uses the same stride, so comparisons are like-for-like.

Usage:  py measure_metrics.py
"""
import json
import os
import sys

import cv2
import numpy as np

STRIDE = 5

LORA_AB = r"D:\h3\output\lora_ab"
RES_AB = r"D:\h3\output\res_ab"
CHAIN = r"D:\h3\output\chain"


def measure(path, stride=STRIDE):
    """Return (sharpness, saturation, frames_sampled, width, height) for one video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("cannot open: " + path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sharp, sat, n, i = [], [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp.append(cv2.Laplacian(grey, cv2.CV_64F).var())
            sat.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
            n += 1
        i += 1
    cap.release()
    return float(np.mean(sharp)), float(np.mean(sat)), n, w, h


def row(label, path, documented=None):
    if not os.path.exists(path):
        print(f"  {label:<34} MISSING  {path}")
        return None
    s, sa, n, w, h = measure(path)
    doc = ""
    if documented is not None:
        diff = (s - documented) / documented * 100 if documented else 0
        doc = f"   documented {documented:.2f}  ({diff:+.1f}%)"
    print(f"  {label:<34} sharp {s:7.2f}   sat {sa:6.2f}   {w}x{h}  n={n}{doc}")
    return {"label": label, "sharpness": round(s, 2), "saturation": round(sa, 2),
            "frames_sampled": n, "width": w, "height": h}


out = {"stride": STRIDE, "metrics": {
    "sharpness": "mean over sampled frames of var(Laplacian(grey, CV_64F))",
    "saturation": "mean over sampled frames of mean(HSV S channel, 0-255)",
}}

print("\n== LoRA A/B (all 1344x768, so directly comparable) ==")
out["lora_ab"] = [r for r in [
    row("A  existing 8-step, shift 12", os.path.join(LORA_AB, "A_old8step_shift12.mp4"), 20.45),
    row("B  v4 LoRA 6-step, shift 12", os.path.join(LORA_AB, "B_v4lora_6step_shift12.mp4"), 12.16),
    row("C  v4 LoRA 6-step, shift 8", os.path.join(LORA_AB, "C_v4lora_6step_shift8.mp4"), 12.91),
] if r]

print("\n== Resolution A/B (NOT comparable by this metric - different pixel counts) ==")
out["res_ab"] = [r for r in [
    row("native 1344x768", os.path.join(RES_AB, "native1344_old8step.mp4")),
    row("1920x1088", os.path.join(RES_AB, "1080p_old8step.mp4")),
] if r]

print("\n== Chain segments (saturation is the figure of interest) ==")
chain = []
for i in (1, 2, 3, 6, 12):
    p = os.path.join(CHAIN, f"seg{i}.mp4")
    r = row(f"seg{i}", p)
    if r:
        r["seg"] = i
        chain.append(r)
if chain:
    base = chain[0]["saturation"]
    print("\n  saturation vs seg1:")
    for r in chain:
        print(f"    seg{r['seg']:<3} {r['saturation']:6.2f}   {(r['saturation'] - base) / base * 100:+6.1f}%")
out["chain"] = chain

dest = r"D:\video_bench\metrics_verified.json"
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("\nwrote", dest)
