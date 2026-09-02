# -*- coding: utf-8 -*-
"""Measure the 10 lora_n.py outputs with the definitions from measure_metrics.py."""
import glob, json, os, re
import cv2, numpy as np

STRIDE = 5
OUT = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\loran"

def measure(path):
    cap = cv2.VideoCapture(path)
    sharp, sat, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % STRIDE == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
            sat.append(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
        i += 1
    cap.release()
    return float(np.mean(sharp)), float(np.mean(sat)), len(sharp)

rows = []
for p in sorted(glob.glob(os.path.join(OUT, "*.mp4"))):
    tag = re.sub(r"_00001_\.mp4$", "", os.path.basename(p))
    s, sa, n = measure(p)
    cell, seed = tag.rsplit("_s", 1)
    rows.append({"tag": tag, "cell": cell, "seed": int(seed),
                 "sharpness": round(s, 2), "saturation": round(sa, 2), "frames": n})
    print(f"  {tag:<20} sharp {s:8.2f}   sat {sa:6.2f}   n={n}")

print("\n  cell means (2 seeds each):")
cells = {}
for r in rows:
    cells.setdefault(r["cell"], []).append(r["sharpness"])
for c in sorted(cells):
    v = cells[c]
    print(f"    {c:<14} {np.mean(v):8.2f}   (seeds: {v[0]:.1f}, {v[1]:.1f})")

with open(r"D:\video_bench\results_loran_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"stride": STRIDE, "rows": rows}, f, indent=1)
print("\n  -> results_loran_metrics.json")
