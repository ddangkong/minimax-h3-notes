# Reddit drafts

Two posts, deliberately separate. The first is a PSA that stands alone and needs no
self-promotion to be useful. The second only makes sense after you have something to be wrong
about, so post it later if at all.

Both are written to be complete in the comment box — nobody has to click anything.

---

## Draft 1 — r/comfyui

**Title:** PSA: MiniMax H3 `ref_images` fails silently if you nest it — and your output still looks fine

**Body:**

Lost a day to this, so posting in case it saves someone else.

If you're driving the H3 reference-to-video node from the API, the obvious way to pass multiple
reference images is a nested dict. It is silently ignored — no error, no warning, and the node
runs with **zero** references:

    # WRONG - silently ignored
    "ref_images": {"ref_image_0": ["40", 0]}

    # RIGHT - flat dot notation
    "ref_images.ref_image_0": ["40", 0]

The execution engine only resolves a value into an image link when it's literally
`[node_id, output_index]` at the position it inspects. It does not descend into nested dicts, so
the inner list is never substituted and nothing raises.

Three things make it nasty:

**1. The output still looks correct.** If your prompt describes the character's appearance —
which is the natural thing to do — you get that character anyway, from the prompt. So it looks
like the reference is working.

**2. The error message points the wrong way.** Put `ref_image_0` at the top level and you get:

    unexpected keyword argument 'ref_image_0'. Did you mean 'ref_images'?

which reads as "wrap it in ref_images". The node source backs up the misreading too — it
iterates `(ref_images or {}).values()`. The correct form is a third shape neither one suggests.

**3. The answer is in the workflow JSON, not the code.** The shipped template declares the input
name as `"name": "ref_images.ref_image_0"`:

    ComfyUI\python_embeded\Lib\site-packages\comfyui_workflow_templates_json\templates\video_minimax_h3_r2v.json

General lesson I'll be keeping: when a node's API is ambiguous, read the official working
workflow before reading the source. The source tells you what the function accepts; the workflow
tells you what the engine actually sends.

**How to check whether yours is working**

Since broken and working look similar, remove the confound — strip the appearance out of the
prompt entirely and leave only the scene:

| prompt | refs | result |
|---|---|---|
| scene only | 3 images | matches your character sheet |
| scene only | none | some random person |

If those two look the same, your references are being ignored no matter what your code says.
Cruder signal: reference processing adds roughly a minute per cut here, so a suspiciously fast
run is a hint.

Once it actually works, the thing that made multi-cut consistency click for me was generating
the character sheet on **flat chroma-key green** (front face / side face / full body, same seed)
and then describing only the scene in the video prompt. A reference transfers everything in it,
so a sheet with a background drags that background into every cut. And whatever you state in the
prompt overrides the reference — so if you describe the character, you're not really using the
reference at all.

Scripts and the rest of my notes, if useful: https://github.com/ddangkong/minimax-h3-notes

---

## Draft 2 — r/comfyui or r/StableDiffusion (post later, or not at all)

**Title:** I benchmarked the v4 turbo LoRA against the old 8-step one, published the result, and then a second seed reversed it

**Body:**

Cautionary tale about n=1, with numbers.

I ran a fast anime fight scene through the older 8-step LoRA and through v4 at 6 steps. v4's
hands came out as pale featureless blobs; the old one rendered knuckles. Clear result, big
visible difference, wrote it up.

Then I re-ran it properly: 2 content types × 3 configurations × 2 seeds, 10 generations.

| content | config | hands | time |
|---|---|---|---|
| fast motion | old 8-step @ 8 | clean in both seeds | 245s |
| fast motion | v4 @ 6 | **blobbed in 1 of 2 seeds** | 184s |
| fast motion | v4 @ 8 | clean in both seeds | 237s |
| slow / static | old 8-step @ 8 | clean in both seeds | 245s |
| slow / static | v4 @ 6 | clean in both seeds | 184s |

The failure is real but occasional, it disappears at 8 steps, and it never showed up on slow
content. The v4 README already says the trade-off appears "only at 4 steps with large, fast
motion" and that "using 6–8 steps largely removes it" — which is exactly right. My
counter-example was seed luck.

The part I think is actually worth sharing is why the metric didn't catch it. Sharpness
(variance of the Laplacian) per cell across two seeds:

| cell | seed 42 | seed 1234 |
|---|---|---|
| old 8-step, fast | 133.6 | 77.1 |
| v4 @ 6, fast | 81.7 | 83.7 |

One cell swings 77 → 134 depending on seed. That spread is wider than the gap between any two
configurations. My original comparison happened to draw the 133.6 seed for the old LoRA and an
81.7 seed for v4 — that's the whole "result". At the other seed the ordering flips.

When seed-to-seed variance is bigger than the effect you're measuring, a single-seed comparison
isn't evidence, and the metric will still hand you a confident number. The cause is visible once
you look: different seeds gave completely different framing, so sharpness was measuring
composition as much as quality.

Two related traps I hit with the same statistic:

- **Comparing sharpness across resolutions.** It's per-pixel, so a bigger frame with the same
  edges scores *lower*. 1344×768 gave 133.6 and 1920×1088 gave 45.5 at native size — which would
  "prove" 1080p is three times worse. It isn't. Downscaling both first doesn't fix it either,
  because the downscale factors differ.
- **Cropping the same coordinates from two runs.** Change a LoRA or a step count and the
  composition changes, so you're comparing different content.

Practical upshot for anyone choosing: use v4 at 8 steps. Clean in every cell I tested and
slightly faster than the old 8-step LoRA.

Raw JSON, scripts and the frames: https://github.com/ddangkong/minimax-h3-notes
