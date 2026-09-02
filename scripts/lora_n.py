# -*- coding: utf-8 -*-
"""Raise n on the v4-LoRA question.

The earlier comparison was a single seed on a single fast-motion prompt, which is not enough
to say anything general. The LoRA author's README is specific about where v4 struggles:

    "The one trade-off shows up only at 4 steps with large, fast motion, where v4 can produce
     motion-smear / trailing ghosting ... Using 6-8 steps largely removes it."

We only ever tested 6. So this run varies three things:

  steps    v4 at 6 AND 8 - does the upper end of the author's stated range clear the hands?
  content  fast fight AND slow/static - v4's recommended domain, where it should win
  seed     two seeds - separates "this configuration" from "this particular generation"

10 runs at roughly 290s each, so about 50 minutes. Videos land in ComfyUI's output under
loran/, and timings/VRAM go to D:\\video_bench\\results_loran.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench5

SEEDS = [42, 1234]

FAST = bench5.PROMPT  # the existing fast fight prompt

# Same subject, same style, deliberately slow and close so the face and hands are large in
# frame and there is no fast motion for v4 to smear. This is the case the README recommends
# v4 for, so it is the fair half of the comparison.
SLOW = ("Japanese anime style, hand-drawn cel-shaded 2D animation, high budget anime. "
        "A young martial artist girl with a long dark ponytail in a red training uniform stands "
        "still in a quiet courtyard, slowly raising one open hand in front of her chest and "
        "turning it over to look at her palm. Her face is clearly visible with sharp detailed "
        "eyes. Medium close-up, calm soft daylight, almost no camera movement, subtle breathing "
        "and a light breeze in her ponytail.")

# (tag, lora, steps, shift, prompt_name)
CELLS = [
    ("A_old8_fast", bench5.LORA_OLD, 8, None, "fast"),
    ("B_v4_6_fast", bench5.LORA_NEW, 6, None, "fast"),
    ("C_v4_8_fast", bench5.LORA_NEW, 8, None, "fast"),
    ("D_old8_slow", bench5.LORA_OLD, 8, None, "slow"),
    ("E_v4_6_slow", bench5.LORA_NEW, 6, None, "slow"),
]

PROMPTS = {"fast": FAST, "slow": SLOW}


def build(lora, steps, shift, tag, prompt, seed):
    """bench5.graph hard-codes PROMPT and SEED at module level; patch both, then restore."""
    old_prompt, old_seed = bench5.PROMPT, bench5.SEED
    bench5.PROMPT, bench5.SEED = prompt, seed
    try:
        g = bench5.graph(lora, steps, shift, tag)
    finally:
        bench5.PROMPT, bench5.SEED = old_prompt, old_seed
    g["92"]["inputs"]["filename_prefix"] = f"loran/{tag}"
    return g


def main():
    if not bench5.alive():
        raise SystemExit("ComfyUI is not responding on " + bench5.SERVER)

    runs = [(f"{tag}_s{seed}", lora, steps, shift, pname, seed)
            for seed in SEEDS for (tag, lora, steps, shift, pname) in CELLS]

    print(f"{len(runs)} runs, {bench5.W}x{bench5.H} / {bench5.FRAMES}f\n")
    results, t0 = [], time.time()

    for i, (tag, lora, steps, shift, pname, seed) in enumerate(runs, 1):
        g = build(lora, steps, shift, tag, PROMPTS[pname], seed)
        print(f"[{i}/{len(runs)}] {tag}  lora={'v4' if lora is bench5.LORA_NEW else 'old8'} "
              f"steps={steps} content={pname} seed={seed}")
        r = bench5.run(tag, g)
        r.update({"tag": tag, "steps": steps, "content": pname, "seed": seed,
                  "lora": "v4" if lora == bench5.LORA_NEW else "old8"})
        results.append(r)
        with open(r"D:\video_bench\results_loran.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"      {r.get('seconds')}s  peak {r.get('peak_vram_mb')}MB   "
              f"elapsed {int(time.time() - t0) // 60}m\n")

    print(f"done in {int(time.time() - t0) // 60} min -> D:\\video_bench\\results_loran.json")


if __name__ == "__main__":
    main()
