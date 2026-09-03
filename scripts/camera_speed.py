# -*- coding: utf-8 -*-
"""Isolate what makes an 8-step turbo generation go soft mid-clip.

Symptom: the Flux still handed in as the first frame is sharp, the first second
of video holds, and from roughly the midpoint the trunks smear and the ferns
turn to green mush.

Four cells, one variable changed at a time, same prompt and same seed
throughout. Changing two things at once tells you nothing about which one did
the work.

  A  baseline                     as generated
  B  + MiniMaxH3SigmaShift(12)    the shift node was never in the graph
  C  + 16 steps                   does the distilled model just need more
  D  + slow camera language       remove fast / rushing / tearing past / rapid

Measure with scripts/measure_sharpness.py, which reads full frames at native
size — see the metric caveats in the README before trusting any number here.
"""
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid

SERVER = "127.0.0.1:8188"
OUTDIR = r"D:\h3\output\qtest"
COMFY_OUT = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\qtest"

W, H = 1536, 640
FRAMES = 124
SEED = 4243
STILL = "forest2/giants.png"          # relative to ComfyUI's input folder

DIT = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

SCENE = ("Hyper detailed digital art, ultra sharp rendering, 8K, tack sharp focus. "
         "Ultra wide panoramic cinemascope composition. The interior of an old growth "
         "forest, two enormous buttressed trunks close on the left, smaller and younger "
         "trees of different species scattered behind at odd angles, ferns and deadfall "
         "on the ground, the forest thinning into brightness on the right")

NEG = ("blurry, soft focus, out of focus, low resolution, low detail, "
       "static, still image, frozen, motionless")

CAM_FAST = ("The camera flies fast and continuously forward, the nearest trunks rushing "
            "toward the lens and streaming past both edges of frame. Constant rapid "
            "forward motion")
CAM_SLOW = ("The camera drifts forward slowly and steadily, the nearest trunks easing "
            "past the edges of frame. Slow deliberate forward motion")

# (name, steps, shift or None, camera)
VARIANTS = [
    ("A_baseline",  8, None, CAM_FAST),
    ("B_shift12",   8, 12.0, CAM_FAST),
    ("C_16step",   16, 12.0, CAM_FAST),
    ("D_slowcam",   8, 12.0, CAM_SLOW),
]


def post(g, timeout=2400):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "http://%s/prompt" % SERVER,
            data=json.dumps({"prompt": g, "client_id": str(uuid.uuid4())}).encode(),
            headers={"Content-Type": "application/json"})
        pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:500]}
    while time.time() - t0 < timeout:
        time.sleep(2)
        try:
            h = json.loads(urllib.request.urlopen(
                "http://%s/history/%s" % (SERVER, pid), timeout=30).read())
        except Exception:
            continue
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                return {"error": str(st.get("messages"))[:500]}
            files = []
            for o in h[pid].get("outputs", {}).values():
                for k in ("images", "video", "gifs"):
                    files += [f.get("filename") for f in (o.get(k) or [])]
            return {"seconds": round(time.time() - t0, 1), "files": files}
    return {"error": "timeout"}


def graph(prompt, name, steps, shift):
    g = {
        "6":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": DIT, "weight_dtype": "default"}},
        "7":  {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["6", 0], "strength_model": 1.0, "lora_name": LORA}},
        "13": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                          "type": "minimax", "device": "default"}},
        # The still is the first frame. MiniMaxH3ImageToVideo emits BOTH the
        # positive conditioning and the AV latent, so there is no separate
        # CLIPTextEncode for the positive and no empty-latent node.
        "40": {"class_type": "LoadImage", "inputs": {"image": STILL}},
        "20": {"class_type": "MiniMaxH3ImageToVideo",
               "inputs": {"clip": ["13", 0], "vae": ["11", 0], "prompt": prompt,
                          "width": W, "height": H, "length": FRAMES,
                          "first_frame": ["40", 0]}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["13", 0], "text": NEG}},
        "11": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "24": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9":  {"class_type": "BasicScheduler",
               "inputs": {"model": ["7", 0], "scheduler": "simple",
                          "steps": steps, "denoise": 1.0}},
        "16": {"class_type": "CFGGuider",
               "inputs": {"model": ["7", 0], "positive": ["20", 0],
                          "negative": ["21", 0], "cfg": 1.0}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0],
                          "sigmas": ["9", 0], "latent_image": ["20", 1]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "fps": 24.0, "audio": ["23", 0]}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": "qtest/" + name,
                          "format": "auto", "codec": "auto"}},
    }
    if shift is not None:
        g["8"] = {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": ["7", 0], "shift_video": shift, "shift_audio": 3.0}}
        g["9"]["inputs"]["model"] = ["8", 0]
        g["16"]["inputs"]["model"] = ["8", 0]
    return g


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for name, steps, shift, cam in VARIANTS:
        dst = os.path.join(OUTDIR, name + ".mp4")
        if os.path.exists(dst):
            print(">>> %s (exists)" % name, flush=True)
            continue
        print(">>> %-12s steps=%d shift=%s cam=%s"
              % (name, steps, shift, "slow" if cam is CAM_SLOW else "fast"), flush=True)
        g = graph("%s. %s" % (SCENE, cam), name, steps, shift)
        # The first frame has gone missing silently before. Assert the link.
        assert g["20"]["inputs"]["first_frame"] == ["40", 0]
        assert g["14"]["inputs"]["latent_image"] == ["20", 1]
        if shift is not None:
            assert g["9"]["inputs"]["model"] == ["8", 0]
            assert g["16"]["inputs"]["model"] == ["8", 0]
        r = post(g)
        if "error" in r:
            print("    failed: %s" % r["error"][:300], flush=True)
            continue
        print("    %ss" % r["seconds"], flush=True)
        src = os.path.join(COMFY_OUT, r["files"][0])
        for _ in range(10):
            try:
                os.replace(src, dst)
                break
            except PermissionError:
                time.sleep(1)      # ComfyUI can still hold the handle
        else:
            shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
