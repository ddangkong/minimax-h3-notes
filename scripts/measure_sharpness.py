# -*- coding: utf-8 -*-
"""Sharpness at two points in a clip, plus a left/right symmetry number.

Two rules from the README apply and both were broken here at least once:

  - measure at native size, never after a downscale
  - compare whole frames, never a fixed crop, because a parameter change moves
    the composition and the same coordinates then hold different content

Sharpness is Laplacian variance, same definition as measure_metrics.py, but
sampled at the midpoint and the last frame rather than averaged over the clip.
Softening in these generations is not uniform: the first second holds and the
degradation arrives later, so a clip-wide mean hides it.

Symmetry is the mean absolute difference between the frame and its own mirror,
subtracted from 100. Higher means more mirror-like. It is a crude number but it
tracks the "this looks like a game asset" complaint well enough to steer by.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

FF = os.environ.get("FFMPEG", r"C:\Users\sonbw\ffmpeg\bin\ffmpeg.exe")


def grab(mp4, n, dst):
    subprocess.run([FF, "-v", "error", "-y", "-i", mp4,
                    "-vf", "select=eq(n\\,%d)" % n, "-vframes", "1", dst], check=False)
    return os.path.exists(dst)


def sharpness(img):
    """Laplacian variance at native resolution. Do not resize before calling."""
    a = np.asarray(img.convert("L"), np.float64)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float64)
    h, w = a.shape
    out = np.zeros((h - 2, w - 2))
    for dy in range(3):
        for dx in range(3):
            if k[dy, dx]:
                out += k[dy, dx] * a[dy:h - 2 + dy, dx:w - 2 + dx]
    return float(out.var())


def symmetry(img):
    """100 - mean|left - mirrored right|. Higher is more mirror-symmetric."""
    a = np.asarray(img.convert("RGB"), np.float32)
    w = a.shape[1]
    half = w // 2
    return 100 - float(np.abs(a[:, :half] - a[:, w - half:][:, ::-1]).mean())


def saturation(img):
    """Mean HSV saturation as a percentage, without requiring cv2."""
    a = np.asarray(img.convert("RGB"), np.float32)
    mx, mn = a.max(axis=2), a.min(axis=2)
    return float(np.mean((mx - mn) / (mx + 1e-6)) * 100)


def report(mp4, frames, tmp):
    os.makedirs(tmp, exist_ok=True)
    out = {}
    for label, n in (("mid", frames // 2), ("end", frames - 1)):
        f = os.path.join(tmp, "%s_%s.png" % (os.path.basename(mp4)[:-4], label))
        if not grab(mp4, n, f):
            continue
        im = Image.open(f).convert("RGB")
        out[label] = {"sharpness": round(sharpness(im), 1),
                      "symmetry": round(symmetry(im), 1),
                      "saturation": round(saturation(im), 1)}
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: measure_sharpness.py <clip.mp4> <frame_count> [more clips...]")
        raise SystemExit(1)
    frames = int(sys.argv[2])
    tmp = os.path.join(os.path.dirname(sys.argv[1]) or ".", "_frames")
    for mp4 in [sys.argv[1]] + sys.argv[3:]:
        r = report(mp4, frames, tmp)
        print("%-24s %s" % (os.path.basename(mp4), r))
