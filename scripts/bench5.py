# -*- coding: utf-8 -*-
"""Final config: every research finding applied, A/B'd against the old settings.

What changed and why:
  1344x768   H3's native canvas (BASE_SHORT_EDGE 768, MAX_PIXELS 768*1344 in
             nodes_minimax_h3.py). 768x448 was 33% of native, which put a head
             at under one token of the 32x-downsampled grid - the faces could
             not be drawn, not merely drawn badly.
  124 frames Documented training range is 124-362. 97 silently snaps to 107 and
             49 snaps to 56, both below that floor. 124 is the real lower bound.
  6 steps    Both turbo-LoRA authors report 4 breaks on large motion and 6 is
             where face/hand micro-detail returns. Above 8 is over-sharpening,
             not detail - consistent with the 20-step run that fixed nothing.
  v4-600 EMA larryvrh's LoRA, documented as removing v1's plastic look.
  shift 8    SigmaShift moves steps from composition toward detail as it drops
             from the 12.0 default. Untested for this, so it is A/B'd, not
             assumed.

VRAM_Debug is deliberately absent: the decode OOM it fixes never occurred here
(peaks fell as resolution rose, because dynamic VRAM offloads more aggressively
on bigger jobs). Adding an unload between sampler and decode would only cost a
reload. Sol-Attn is also skipped - its own author reports it is worse at low
resolution, and 5080 reports range from no speedup to 2x slower.
"""
import json
import time
import urllib.request
import urllib.error
import subprocess
import threading
import uuid
import os

SERVER = "127.0.0.1:8188"
OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 42
FPS = 24
W, H, FRAMES = 1344, 768, 124

NEG = ("photorealistic, realistic, 3d render, cgi, live action, video game, blurry, "
       "ugly, deformed face, extra limbs, distorted anatomy, watermark, text")

PROMPT = ("Japanese anime style, hand-drawn cel-shaded 2D animation, high budget action anime. "
          "A young martial artist girl with a long dark ponytail in a red training uniform throws "
          "a rapid punch combo toward the camera, then spins into a high roundhouse kick. Her face "
          "is clearly visible with sharp detailed eyes. Dust kicks up from the stone floor, her "
          "ponytail and sleeves trailing the motion. Low angle, speed lines, sakuga fight animation.")

LORA_OLD = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
LORA_NEW = "minimax_h3_turbo_v4_step600_ema.safetensors"


def graph(lora, steps, shift, tag):
    g = {
        "6":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                          "weight_dtype": "default"}},
        "7":  {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["6", 0], "strength_model": 1.0, "lora_name": lora}},
        "13": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                          "type": "minimax", "device": "default"}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["13", 0], "text": PROMPT}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["13", 0], "text": NEG}},
        "11": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "24": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "30": {"class_type": "EmptyMiniMaxH3LatentAV",
               "inputs": {"width": W, "height": H, "length": FRAMES}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
    }
    # shift=None keeps the model's built-in 12.0/3.0 default
    model_src = ["7", 0]
    if shift is not None:
        g["8"] = {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": ["7", 0], "shift_video": float(shift), "shift_audio": 3.0}}
        model_src = ["8", 0]
    g.update({
        "9":  {"class_type": "BasicScheduler",
               "inputs": {"model": model_src, "scheduler": "simple",
                          "steps": steps, "denoise": 1.0}},
        "16": {"class_type": "CFGGuider",
               "inputs": {"model": model_src, "positive": ["20", 0],
                          "negative": ["21", 0], "cfg": 1.0}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0],
                          "sigmas": ["9", 0], "latent_image": ["30", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "fps": float(FPS), "audio": ["23", 0]}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": f"r5/{tag}",
                          "format": "auto", "codec": "auto"}},
    })
    return g


# Change one variable at a time from the old baseline.
RUNS = [
    ("A_old8step",   LORA_OLD, 8, None),   # native res only
    ("B_new6step",   LORA_NEW, 6, None),   # + better LoRA, 6 steps
    ("C_new6_sh8",   LORA_NEW, 6, 8.0),    # + shift 8
]


def vram_mb():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return int(o.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def alive():
    try:
        urllib.request.urlopen(f"http://{SERVER}/system_stats", timeout=10)
        return True
    except Exception:
        return False


def run(name, g, timeout=3000):
    peak = [0]
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], vram_mb())
            stop.wait(1.0)

    threading.Thread(target=watch, daemon=True).start()
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"http://{SERVER}/prompt",
            data=json.dumps({"prompt": g, "client_id": str(uuid.uuid4())}).encode(),
            headers={"Content-Type": "application/json"})
        pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        stop.set()
        return {"name": name, "error": e.read().decode()[:600]}

    result = None
    while time.time() - t0 < timeout:
        time.sleep(3)
        if not alive():
            stop.set()
            return {"name": name, "error": "서버 사망 (OOM)", "peak_vram_mb": peak[0]}
        try:
            h = json.loads(urllib.request.urlopen(
                f"http://{SERVER}/history/{pid}", timeout=30).read())
        except Exception:
            continue
        if pid in h:
            result = h[pid]
            break
    stop.set()
    el = round(time.time() - t0, 1)
    if result is None:
        return {"name": name, "error": f"timeout {el}s", "peak_vram_mb": peak[0]}
    st = result.get("status", {})
    if st.get("status_str") == "error":
        return {"name": name, "error": str(st.get("messages"))[:600], "peak_vram_mb": peak[0]}
    files = [f.get("filename") for o in result.get("outputs", {}).values()
             for k in ("images", "video", "gifs") for f in (o.get(k) or [])]
    return {"name": name, "seconds": el, "peak_vram_mb": peak[0], "files": files}


def main():
    print(f"H3 네이티브 {W}x{H} / {FRAMES}f / cfg 1.0\n")
    results = []
    for tag, lora, steps, shift in RUNS:
        label = f"{tag} (steps={steps}, shift={shift or 'default 12'})"
        print(f">>> {label}", flush=True)
        r = run(tag, graph(lora, steps, shift, tag))
        results.append(r)
        if "error" in r:
            print(f"    실패: {r['error'][:200]}", flush=True)
        else:
            print(f"    {r['seconds']}s  peak {r['peak_vram_mb']/1024:.2f} GB", flush=True)
        with open(os.path.join(OUT, "results5.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n===== 요약 =====")
    for r in results:
        if "error" in r:
            print(f"  {r['name']:14} 실패 {r['error'][:120]}")
        else:
            print(f"  {r['name']:14} {r['seconds']:7.1f}s  peak {r['peak_vram_mb']/1024:5.2f} GB")


if __name__ == "__main__":
    main()
