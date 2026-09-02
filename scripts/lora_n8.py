# -*- coding: utf-8 -*-
"""Extend the LoRA grid to n=8 per cell on fast motion.

At n=2 the result was "v4 at 6 steps blobbed hands in 1 of 2 seeds, v4 at 8 steps was clean in
2 of 2". Both of those are almost meaningless as rates - 1/2 is consistent with a true failure
rate anywhere from 10% to 90%, and 0/2 clean says very little about whether 8 steps ever fails.

This adds six more seeds to the three fast-motion cells, giving n=8 each. Slow content is left
at n=2 because nothing failed there and it is the less interesting half.

18 runs, roughly 67 minutes. Combine with results_loran.json for the full n=8.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench5
import lora_n

NEW_SEEDS = [7, 99, 555, 2024, 31337, 8888]

# fast-motion cells only
CELLS = [c for c in lora_n.CELLS if c[4] == "fast"]


def main():
    if not bench5.alive():
        raise SystemExit("ComfyUI is not responding on " + bench5.SERVER)

    runs = [(f"{tag}_s{seed}", lora, steps, shift, pname, seed)
            for seed in NEW_SEEDS for (tag, lora, steps, shift, pname) in CELLS]

    print(f"{len(runs)} runs -> n=8 per fast-motion cell\n")
    results, t0 = [], time.time()

    for i, (tag, lora, steps, shift, pname, seed) in enumerate(runs, 1):
        g = lora_n.build(lora, steps, shift, tag, lora_n.PROMPTS[pname], seed)
        print(f"[{i}/{len(runs)}] {tag}  lora={'v4' if lora == bench5.LORA_NEW else 'old8'} "
              f"steps={steps} seed={seed}")
        r = bench5.run(tag, g)
        r.update({"tag": tag, "steps": steps, "content": pname, "seed": seed,
                  "lora": "v4" if lora == bench5.LORA_NEW else "old8"})
        results.append(r)
        with open(r"D:\video_bench\results_loran8.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        el = int(time.time() - t0)
        print(f"      {r.get('seconds')}s  peak {r.get('peak_vram_mb')}MB   "
              f"elapsed {el // 60}m  eta {(el // i * (len(runs) - i)) // 60}m\n")

    print(f"done in {int(time.time() - t0) // 60} min -> results_loran8.json")


if __name__ == "__main__":
    main()
