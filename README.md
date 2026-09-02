# MiniMax H3 — measured notes from one 16GB card

First-hand measurements and working scripts from running MiniMax H3 locally on a single
RTX 5080 (16GB). **This is not a general guide.** Every number here came off one machine with
one kind of content, and the whole reason it exists is that general guides did not match what
the machine actually did.

If you only read one section, read [the silent `ref_images` failure](#the-silent-ref_images-failure).

```
RTX 5080 16GB · 31GB system RAM · ComfyUI 0.33.1 · torch 2.12.0+cu130 · Windows
```

---

## Contents

| | |
|---|---|
| [The silent `ref_images` failure](#the-silent-ref_images-failure) | A wrong input shape is ignored with no error, and the output still looks fine |
| [Character consistency across cuts](#character-consistency-across-cuts) | Three-view chroma-key sheets, scene-only prompts |
| [What the card actually does](#what-the-card-actually-does) | Frame ceiling, VRAM, resolution, timings |
| [Metric definitions](#metric-definitions) | What "sharpness" and "saturation" mean here, and two ways to measure them wrong |
| [Scripts](#scripts) | |
| [Limitations](#limitations) | Read this before trusting any number above |

---

## The silent `ref_images` failure

The reference-to-video node takes multiple reference images. The intuitive way to pass them is
a nested object. That is silently ignored:

```python
# WRONG — no error, no warning, node runs with zero references
"ref_images": {"ref_image_0": ["40", 0]}

# RIGHT — flat dot notation, as in the shipped workflow template
"ref_images.ref_image_0": ["40", 0]
```

ComfyUI's execution engine resolves a value into an image link only when it is literally
`[node_id, output_index]` at the position it inspects. **It does not descend into nested
dictionaries.** The inner `["40", 0]` is never substituted, so the node executes with no
references at all, and nothing raises.

Three reasons this is hard to catch:

1. **The output still looks right.** If the prompt describes the character's appearance — the
   natural thing to do — you get that character with zero references attached, and conclude the
   reference is working.
2. **The error message points the wrong way.** Passing `ref_image_0` at the top level gives
   `unexpected keyword argument 'ref_image_0'. Did you mean 'ref_images'?`, which reads as an
   instruction to nest it. The node source reinforces the misreading — it iterates
   `(ref_images or {}).values()`. The correct form is a third shape neither one suggests.
3. **The fix is in the workflow file, not the code.** The shipped template declares the input
   name as `"name": "ref_images.ref_image_0"`:
   ```
   ComfyUI\python_embeded\Lib\site-packages\comfyui_workflow_templates_json\templates\
     video_minimax_h3_r2v.json
   ```

**When a node's API is ambiguous, read the official working workflow before reading the source.**
The source tells you what the function accepts. The workflow tells you what the engine sends.

### Proving the references are actually doing something

A working run and a broken run look similar, so remove the confound — empty the appearance out
of the prompt entirely:

| Condition | Prompt | References | Result |
|---|---|---|---|
| A | Scene only | 3 images | Matches the character sheet |
| B | Scene only | none | An arbitrary person |

If A and B look the same, references are being ignored no matter what the code appears to say.
Secondary signal: reference processing adds roughly a minute per cut, so a suspiciously fast run
is a hint. See [`scripts/ref_ablation.py`](scripts/ref_ablation.py).

---

## Character consistency across cuts

![reference sheet and three independently generated cuts](images/h3-ember-consistency.jpg)

Three steps, and the third is the one people skip:

| Step | What | Why |
|---|---|---|
| 1 | Flux generates three sheets on flat green — front face, side face, full body. Same seed. | The identity has to exist as pixels before it can be referenced |
| 2 | All three go in as separate reference slots | One view carries only the information in that view |
| 3 | The video prompt describes **scene and action only** — no appearance | Anything you state overrides the reference |

About five minutes per cut at 768×1344, 124 frames, 8 steps.

**Why flat green.** A reference transfers everything in it. A sheet shot against a sky drags that
sky into every cut, which defeats writing a different scene each time. Keep the sheet lighting
neutral too — a colour cast in the reference tints every cut.

**Why three views.** Each carries different information, and the gap shows up as a specific
failure:

| View | Carries | Failure without it |
|---|---|---|
| Front face | Features, eye colour | Identity drifts between cuts |
| Side face | Nose and jaw line, back of head | Profile shots get an invented face |
| Full body | Leg length, shoulder-to-waist ratio, lower costume | **Proportions collapse** |

**The rule that explains most failures:** what the prompt states wins; the reference only fills
what the prompt leaves empty. Three separate bugs traced back to this one rule — an East Asian
reference producing a Western character (ethnicity unstated, so the model used its default), a
new reference still producing the old character (the costume description was still in the
prompt), and consistency surviving with references disabled (the prompt described everything).

**If you change the reference, change the prompt's description of the person too** — or better,
remove appearance from the prompt entirely and let the reference own it.

![second character, same pipeline](images/h3-tide-consistency.jpg)

Getting an extreme facial close-up out of Flux mostly does not work; repeated attempts still
framed the upper body. Detecting the subject against the green and cropping the head is more
reliable than arguing with the prompt.

See [`scripts/flux_charsheet_green.py`](scripts/flux_charsheet_green.py) and
[`scripts/make_two.py`](scripts/make_two.py).

---

## What the card actually does

### Frame ceiling

Published guidance puts a 16GB laptop GPU at
[5 seconds at 960×540](https://minimax-h3.wiki/local/minimax-h3-hardware-requirements/).
Measured on a 16GB desktop card at **1344×768** — about 2.7× those pixels:

| Frames | Length | Time | Peak VRAM |
|---|---|---|---|
| 124 | 5.2s | 4.8 min | 15.35 GB |
| 209 | 8.7s | 9.5 min | 15.02 GB |
| 260 | 10.8s | 13.8 min | 11.50 GB |
| 311 | 13.0s | 19.1 min | 13.55 GB |
| **362** | **15.1s** | 25.3 min | 15.33 GB |
| 430 | 17.9s | **OOM** | — |

Time scales close to linearly with frame count, unlike resolution. Quality does not decay along
the clip — at 13s into a 15s generation, faces, fingers and clothing knots are intact.

Frame counts snap to a 17k+5 lattice. Ask for 424 and you get 430. Valid: 124, 209, 260, 311,
362, 379, 396, 413, 430.

### VRAM is not the bottleneck — system RAM is

Peak VRAM stayed between 11.5 and 15.4 GB **regardless of resolution or frame count**. Dynamic
VRAM streaming offloads harder as the job grows, so VRAM stays roughly pinned while PCIe traffic
rises — which is also why time explodes above 1080p while VRAM does not move.

The DiT and text encoder stream from system RAM. Two settings mattered more than any GPU tuning:

| Setting | Why |
|---|---|
| `--disable-pinned-memory` | Otherwise a large share of system RAM is page-locked. Locked pages cannot swap, so the OOM killer takes the process. Measured 29.8GB → 7.5GB host RAM |
| WSL2 memory cap | WSL2 claims up to half of host RAM by default and does not return it after freeing it inside the VM |

If a run dies without a CUDA OOM message, look at host RAM first.

### More steps does not fix a face

Twenty steps cost three times the time and produced the same mangled face. The model works on a
latent grid downsampled ~32× from the output:

| Output | Token grid |
|---|---|
| 768 × 448 | 24 × 14 = 336 |
| 1344 × 768 | 42 × 24 = 1008 |

A face occupying 20–30 px of a 768×448 output is **under one token**. It was not drawn badly —
there was no unit to draw it with. Steps refine what the grid can represent; they cannot create
grid. The fix is resolution or framing, not sampling.

### Timings across resolutions

| Resolution / frames | Time |
|---|---|
| 1280 × 720 / 73f | 95s |
| 1344 × 768 / 124f | 290s |
| 1536 × 864 / 97f | 285s |
| 1920 × 1088 / 73f | 668s |
| 1920 × 1088 / 124f | 783s |

1920×1088 exceeds the nominal max pixel count and is still visibly better — separated hair
strands, iris gradients, background depth — at 2.7× the time. Iterate at 1344×768, render finals
at 1920×1088.

### Chained generation converges rather than drifting

Feeding the last frame of one segment in as the first frame of the next. The usual warning is
that drift compounds by the third to fifth segment. Measured saturation over twelve segments:

| Segment | Saturation | vs seg 1 |
|---|---|---|
| 1 | 42.7 | baseline |
| 2 | 59.7 | +39.8% |
| 3 | 71.2 | +66.6% |
| 6 | 72.8 | +70.5% |
| 12 | 73.7 | +72.5% |

Two thirds of the change happens by segment 3; the next nine segments add six points between
them. This is a one-time text-to-video → image-to-video conditioning difference followed by a
slow crawl, not compounding drift.

The real constraint on chaining is different: **only what is visible in the handoff frame
survives.** Chain from a close-up of a hand and you lose the character, the other subjects and
the art style at once. Let your harness choose which earlier frame to inherit from.

---

## Metric definitions

Any writeup quoting a sharpness number owes you its definition, because the word covers several
incompatible statistics. Here:

```python
sharpness  = mean over sampled frames of cv2.Laplacian(grey, cv2.CV_64F).var()
saturation = mean over sampled frames of cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[:,:,1].mean()
```

Frame stride 5 throughout. See [`scripts/measure_metrics.py`](scripts/measure_metrics.py) and
[`results/metrics_verified.json`](results/metrics_verified.json).

### Two ways to measure this wrong

**Comparing sharpness across resolutions.** Laplacian variance is per-pixel, so a larger frame
carrying the same edges scores *lower*. Measured at native size, 1344×768 returns 133.6 and
1920×1088 returns 45.5 — which would "prove" the higher resolution is three times worse. It is
not. Downscaling both to a common size first does not fix it either, because the downscale
factors differ. **This statistic cannot compare two resolutions in either direction.** Use it
within one resolution and judge resolution by eye.

**Cropping a region to compare.** Changing a LoRA or a step count changes the generation, and
therefore the composition. Cropping identical coordinates compares different content. Compare
full frames.

**And: an unrecorded method is an unverifiable number.** The sharpness and saturation figures in
the original notes were written down without their definitions. Recomputing them from the same
files reproduced the ratios — the A/B sharpness ratio came out 1.64 against a recorded 1.68, and
the saturation curve matched within a few points — but absolute values differ by a constant
factor and one ordering swapped. The conclusions held; the fact that this could not be known in
advance is the point.

---

## Scripts

Paths are hardcoded for the author's machine. Change the constants at the top of each file.

| Script | What |
|---|---|
| [`bench5.py`](scripts/bench5.py) | A/B harness — builds the graph, submits, watches VRAM, records timings |
| [`measure_metrics.py`](scripts/measure_metrics.py) | Sharpness / saturation with the definitions above |
| [`lora_n.py`](scripts/lora_n.py) | Raising n on the LoRA question — 2 content types × 2 seeds × steps |
| [`ref_ablation.py`](scripts/ref_ablation.py) | Confirms references are actually being applied |
| [`flux_charsheet_green.py`](scripts/flux_charsheet_green.py) | Three-view chroma-key character sheets |
| [`make_two.py`](scripts/make_two.py) | End-to-end: two characters, three cuts each |

ComfyUI launch flags used throughout:

```
--disable-pinned-memory --disable-async-offload --disable-smart-memory --reserve-vram 1.5
```

---

## Limitations

Worth being blunt about, because the whole point of this repo is that other people's numbers
did not transfer:

- **One GPU, one model, largely one content type** (anime-style action). A different card or a
  different subject will give a different table.
- **Most cells are n=1 on a single seed.** The LoRA comparison is being re-run across two
  content types and two seeds; results will land in `results/`.
- **Sharpness is a blunt instrument.** It does not capture "the hands became featureless
  blobs", which was the deciding factor in one comparison. Look at the frames.
- Scripts assume Windows paths and a local ComfyUI at `127.0.0.1:8188`.

Corrections welcome — particularly from anyone whose numbers disagree, since that is the
interesting case.
