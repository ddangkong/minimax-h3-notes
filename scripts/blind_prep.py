# -*- coding: utf-8 -*-
"""Build a blind scoring set for the 24 fast-motion clips.

The thing being scored - "are the hands rendered as featureless blobs" - is a judgement call,
and I already have a hypothesis about which cells should fail. That is exactly the setup where
expectation leaks into scoring. So: extract one frame per clip, give each a random anonymous id,
write the id -> cell mapping to a key file, and build contact sheets that show only the ids.

Score from the sheets, write the scores down, then read the key. Not perfect blinding (the
configurations do have visual tells beyond the hands) but it removes the label from the moment
of judgement, which is where the bias would enter.

Criterion, fixed before looking at anything:
    FAIL = a hand region rendered as a broadly uniform light mass with no discernible finger
           or knuckle separation, at a scale where separation would be visible.
"""
import glob
import json
import os
import random
import re
import subprocess

FF = r"C:\Users\sonbw\ffmpeg\bin\ffmpeg.exe"
SRC = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\loran"
WORK = r"C:\Users\sonbw\AppData\Local\Temp\claude\D--\7f58a4d7-5292-41be-8e5c-f0e1ac793e6c\scratchpad\h3\blind"
FONT = r"C\:/Windows/Fonts/arialbd.ttf"

os.makedirs(WORK, exist_ok=True)

clips = sorted(p for p in glob.glob(os.path.join(SRC, "*fast*.mp4")))
print(f"{len(clips)} fast-motion clips")

random.seed(20260902)
ids = [f"{i:02d}" for i in range(1, len(clips) + 1)]
random.shuffle(ids)

key = {}
for anon, path in zip(ids, clips):
    tag = re.sub(r"_00001_\.mp4$", "", os.path.basename(path))
    key[anon] = tag
    out = os.path.join(WORK, f"{anon}.png")
    subprocess.run([FF, "-v", "error", "-y", "-ss", "2.0", "-i", path,
                    "-frames:v", "1", "-vf", "scale=400:-1", out], check=True)

with open(r"D:\video_bench\blind_key.json", "w", encoding="utf-8") as f:
    json.dump(key, f, indent=1, sort_keys=True)

# contact sheets, 4 across, anonymous ids only
order = sorted(key)
for page, start in enumerate(range(0, len(order), 8), 1):
    chunk = order[start:start + 8]
    args = [FF, "-v", "error", "-y"]
    for a in chunk:
        args += ["-i", os.path.join(WORK, f"{a}.png")]
    parts, labels = [], []
    for i, a in enumerate(chunk):
        parts.append(f"[{i}:v]pad=400:266:0:36:color=0x141414,"
                     f"drawtext=fontfile='{FONT}':text='{a}':x=12:y=6:"
                     f"fontsize=26:fontcolor=yellow[v{i}]")
        labels.append(f"[v{i}]")
    top = "".join(labels[:4]) + f"hstack=inputs={len(labels[:4])}[r1]"
    filt = ";".join(parts) + ";" + top
    if len(chunk) > 4:
        bot = "".join(labels[4:]) + f"hstack=inputs={len(labels[4:])}[r2]"
        filt += ";" + bot + ";[r1][r2]vstack=inputs=2[o]"
    else:
        filt += ";[r1]null[o]"
    args += ["-filter_complex", filt, "-map", "[o]", "-q:v", "3",
             os.path.join(WORK, f"sheet{page}.jpg")]
    subprocess.run(args, check=True)
    print(f"  sheet{page}.jpg  ids {' '.join(chunk)}")

print(f"\nkey written to D:\\video_bench\\blind_key.json - do not open until scoring is done")
