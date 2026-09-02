# -*- coding: utf-8 -*-
"""Re-score the blind set from multiple frames per clip instead of one.

The first pass judged each 5.2s clip from a single frame at t=2.0s. That is the wrong instrument
for this failure: the LoRA author describes it as motion-smear / trailing ghosting, i.e. an
artifact that comes and goes with the motion. One still can miss a clip that fails elsewhere, or
condemn a clip on a transient frame.

Same anonymous ids as blind_prep.py (same seed, same shuffle), so the key still applies and the
scoring stays blind. Each clip now gets a horizontal strip of 5 frames across its length.

Criterion is unchanged and still fixed in advance:
    FAIL = at least one frame where a hand region is a broadly uniform light mass with no
           discernible finger or knuckle separation, at a scale where separation would show.
"""
import glob
import json
import os
import random
import re
import subprocess

FF = r"C:\Users\sonbw\ffmpeg\bin\ffmpeg.exe"
SRC = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\loran"
WORK = (r"C:\Users\sonbw\AppData\Local\Temp\claude\D--"
        r"\7f58a4d7-5292-41be-8e5c-f0e1ac793e6c\scratchpad\h3\blind2")
FONT = r"C\:/Windows/Fonts/arialbd.ttf"
TIMES = [0.8, 1.6, 2.4, 3.2, 4.0]

os.makedirs(WORK, exist_ok=True)

clips = sorted(p for p in glob.glob(os.path.join(SRC, "*fast*.mp4")))
random.seed(20260902)                      # identical shuffle to blind_prep.py
ids = [f"{i:02d}" for i in range(1, len(clips) + 1)]
random.shuffle(ids)
pairs = sorted(zip(ids, clips))

print(f"{len(clips)} clips x {len(TIMES)} frames")

for anon, path in pairs:
    ins, filt, labels = [], [], []
    for i, t in enumerate(TIMES):
        ins += ["-ss", str(t), "-i", path]
        filt.append(f"[{i}:v]scale=300:-1,"
                    f"drawtext=fontfile='{FONT}':text='{t}s':x=6:y=6:"
                    f"fontsize=17:fontcolor=white:box=1:boxcolor=black@0.55[v{i}]")
        labels.append(f"[v{i}]")
    chain = (";".join(filt) + ";" + "".join(labels)
             + f"hstack=inputs={len(TIMES)},pad=1500:{171+34}:0:34:color=0x141414,"
             + f"drawtext=fontfile='{FONT}':text='{anon}':x=10:y=5:fontsize=24:fontcolor=yellow[o]")
    subprocess.run([FF, "-v", "error", "-y", *ins, "-filter_complex", chain,
                    "-map", "[o]", "-q:v", "3", os.path.join(WORK, f"{anon}.jpg")], check=True)

# 4 strips per sheet
order = [a for a, _ in pairs]
for page, start in enumerate(range(0, len(order), 4), 1):
    chunk = order[start:start + 4]
    args = [FF, "-v", "error", "-y"]
    for a in chunk:
        args += ["-i", os.path.join(WORK, f"{a}.jpg")]
    filt = "".join(f"[{i}:v]" for i in range(len(chunk))) + f"vstack=inputs={len(chunk)}[o]"
    args += ["-filter_complex", filt, "-map", "[o]", "-q:v", "3",
             os.path.join(WORK, f"sheet{page}.jpg")]
    subprocess.run(args, check=True)
    print(f"  sheet{page}.jpg  {' '.join(chunk)}")

print("\nkey unchanged: D:\\video_bench\\blind_key.json")
