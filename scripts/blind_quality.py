# -*- coding: utf-8 -*-
"""Blind paired comparison: old 8-step vs v4 @ 8 steps, overall quality.

The hand-blob criterion was a binary failure test and it missed the broader question. Watching
the clips, the older LoRA looks better overall - background texture, line art, general crispness -
independently of whether the hands survive. That is a different claim and needs its own test.

Same seed on both sides so the composition is comparable, left/right randomised per pair, no
labels. Judge which side looks better, then open the key.

Question fixed in advance:
    Which side has more retained detail - background texture, line definition, material
    surfaces - ignoring the hands specifically?
"""
import json
import os
import random
import subprocess

FF = r"C:\Users\sonbw\ffmpeg\bin\ffmpeg.exe"
SRC = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\loran"
WORK = (r"C:\Users\sonbw\AppData\Local\Temp\claude\D--"
        r"\7f58a4d7-5292-41be-8e5c-f0e1ac793e6c\scratchpad\h3\quality")
FONT = r"C\:/Windows/Fonts/arialbd.ttf"
SEEDS = [42, 1234, 7, 99, 555, 2024, 31337, 8888]

os.makedirs(WORK, exist_ok=True)
random.seed(777)

key = {}
pair_ids = [f"P{i}" for i in range(1, len(SEEDS) + 1)]
random.shuffle(pair_ids)

for pid, seed in zip(pair_ids, SEEDS):
    a = os.path.join(SRC, f"A_old8_fast_s{seed}_00001_.mp4")
    c = os.path.join(SRC, f"C_v4_8_fast_s{seed}_00001_.mp4")
    left_is_old = random.random() < 0.5
    left, right = (a, c) if left_is_old else (c, a)
    key[pid] = {"seed": seed, "left": "old8" if left_is_old else "v4_8",
                "right": "v4_8" if left_is_old else "old8"}
    subprocess.run([
        FF, "-v", "error", "-y", "-ss", "2.4", "-i", left, "-ss", "2.4", "-i", right,
        "-frames:v", "1",
        "-filter_complex",
        "[0:v]scale=620:-1,pad=620:390:0:36:color=0x141414,"
        f"drawtext=fontfile='{FONT}':text='L':x=10:y=6:fontsize=24:fontcolor=white[l];"
        "[1:v]scale=620:-1,pad=620:390:0:36:color=0x141414,"
        f"drawtext=fontfile='{FONT}':text='R':x=10:y=6:fontsize=24:fontcolor=white[r];"
        f"[l][r]hstack=inputs=2,pad=1240:426:0:0:color=0x141414,"
        f"drawtext=fontfile='{FONT}':text='{pid}':x=12:y=396:fontsize=24:fontcolor=yellow[o]",
        "-map", "[o]", "-q:v", "2", os.path.join(WORK, f"{pid}.jpg")], check=True)
    print(f"  {pid}  seed {seed}")

# two sheets of four pairs
order = sorted(key)
for page, start in enumerate(range(0, len(order), 4), 1):
    chunk = order[start:start + 4]
    args = [FF, "-v", "error", "-y"]
    for p in chunk:
        args += ["-i", os.path.join(WORK, f"{p}.jpg")]
    filt = "".join(f"[{i}:v]" for i in range(len(chunk))) + f"vstack=inputs={len(chunk)}[o]"
    args += ["-filter_complex", filt, "-map", "[o]", "-q:v", "3",
             os.path.join(WORK, f"qsheet{page}.jpg")]
    subprocess.run(args, check=True)
    print(f"  qsheet{page}.jpg  {' '.join(chunk)}")

with open(r"D:\video_bench\quality_key.json", "w", encoding="utf-8") as f:
    json.dump(key, f, indent=1, sort_keys=True)
print("\nkey: D:\\video_bench\\quality_key.json - do not open until judged")
