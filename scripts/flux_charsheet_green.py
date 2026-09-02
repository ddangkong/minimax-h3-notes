# -*- coding: utf-8 -*-
"""크로마키 캐릭터 시트 — 초록 단색 배경, 캐릭터 정보만 담는다.

앞서 만든 시트는 하늘·꽃밭 배경이 들어 있었다. Ref2V는 참조 이미지 전체를
보므로 그 배경까지 캐릭터 정보와 함께 가져간다. 컷마다 배경이 달라야 하는
영상에서는 그게 방해가 된다.

배경을 초록 단색으로 두면 참조에 남는 것은 인물뿐이다. 배경·조명·분위기는
각 컷의 프롬프트가 온전히 결정한다.

  1. 정면 얼굴 (초록 배경)
  2. 측면 얼굴 (초록 배경)
  3. 전신     (초록 배경, 얼굴 포함)

세 장 모두 같은 시드를 써서 동일 인물을 유지한다. 조명도 중립 균일광으로
두어, 특정 색조가 캐릭터에 물들지 않게 한다.
"""
import json
import time
import urllib.request
import urllib.error
import uuid
import os

SERVER = "127.0.0.1:8188"
OUTDIR = r"D:\h3\output\charsheet_green"
COMFY_OUT = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\csgreen"

CHAR = ("A teenage girl with long silver-white hair in a high ponytail, large luminous "
        "violet eyes, pale skin, wearing a white and gold sleeveless top, a short white "
        "skirt with gold trim, white thigh-high boots, and a translucent iridescent cape")

# 배경은 완전 단색. 조명도 중립으로 두어 색조가 캐릭터에 물들지 않게 한다.
STYLE = ("Japanese anime style character reference sheet, vibrant cel-shaded 2D animation, "
         "high budget studio anime, crisp clean anime line art, bright even neutral studio "
         "lighting with no coloured cast, richly saturated character colours. "
         "Plain flat solid chroma key green screen background, uniform pure green backdrop, "
         "completely empty background with no scenery, no props, no ground, no horizon, "
         "no shadows on the background. Sharp focus, highly detailed")

# 배경에 무엇도 들어오지 못하게 막는다.
NEG_BASE = ("photorealistic, 3d render, cgi, live action, blurry, dark, desaturated, "
            "muted colours, washed out, deformed face, distorted anatomy, extra limbs, "
            "extra fingers, watermark, text, multiple people, "
            "scenery, landscape, sky, clouds, grass, flowers, field, trees, room, wall, "
            "gradient background, patterned background, lens flare, sunlight, backlight, "
            "coloured lighting, shadow on background, vignette")

SHOTS = [
    ("face_front", 1024, 1024,
     "Extreme close-up of her face only, tight head crop, her head filling most of the "
     "frame from just above the hair to the base of the neck. Front view facing the "
     "camera directly, symmetrical, neutral calm expression, mouth closed, eyes open "
     "looking straight at the viewer, both eyes large and clearly visible. Skin, eyes "
     "and hair rendered in fine detail.",
     ", side view, profile, three-quarter view, looking away, full body, half body, "
     "upper body, torso, waist, legs, distant, small head, wide shot, cropped head"),

    ("face_side", 1024, 1024,
     "Extreme close-up of her face only in strict profile, tight head crop, her head "
     "filling most of the frame. Side view from her left, looking straight ahead to the "
     "side, showing the clean line of her nose, lips and chin, the ear, and the shape of "
     "the back of her head where the ponytail is tied. Neutral expression, mouth closed. "
     "Skin, eye and hair rendered in fine detail.",
     ", front view, facing the camera, looking at the viewer, three-quarter view, "
     "full body, half body, upper body, torso, waist, legs, distant, small head, "
     "wide shot, cropped head"),

    # 전신은 재생성하지 않는다 (이미 양호)
    ("body_full_SKIP", 896, 1408,
     "Full body character reference, complete figure from the top of her head down to the "
     "soles of her boots, both feet fully visible in frame, generous empty space above her "
     "head and below her feet, the whole body occupying only the middle portion of the "
     "tall frame. Standing straight facing the camera, arms relaxed at her sides, face "
     "clearly visible, neutral calm expression. Natural slender anime proportions, long "
     "well proportioned legs, correct anatomy.",
     ", close-up, portrait crop, headshot, bust shot, cropped legs, cropped feet, "
     "feet out of frame, cut off at the waist, cut off at the knees, zoomed in, "
     "short legs, stubby limbs, oversized head, sitting, flying, floating"),
]

SEED = 333


def graph(prompt, neg, w, h, tag):
    return {
        "1":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": "flux1-krea-dev_fp8_scaled.safetensors",
                          "weight_dtype": "default"}},
        "2":  {"class_type": "DualCLIPLoader",
               "inputs": {"clip_name1": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                          "clip_name2": "clip_l.safetensors",
                          "type": "flux", "device": "default"}},
        "3":  {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": neg}},
        "6":  {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": 3.5}},
        "7":  {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": w, "height": h, "batch_size": 1}},
        "8":  {"class_type": "KSampler",
               "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["5", 0],
                          "latent_image": ["7", 0], "seed": SEED, "steps": 28, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9":  {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0], "filename_prefix": "csgreen/" + tag}},
    }


def run(name, g, timeout=900):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"http://{SERVER}/prompt",
            data=json.dumps({"prompt": g, "client_id": str(uuid.uuid4())}).encode(),
            headers={"Content-Type": "application/json"})
        pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        return {"name": name, "error": e.read().decode()[:500]}
    while time.time() - t0 < timeout:
        time.sleep(2)
        try:
            h = json.loads(urllib.request.urlopen(
                f"http://{SERVER}/history/{pid}", timeout=30).read())
        except Exception:
            continue
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                return {"name": name, "error": str(st.get("messages"))[:500]}
            files = [f.get("filename") for o in h[pid].get("outputs", {}).values()
                     for f in (o.get("images") or [])]
            return {"name": name, "seconds": round(time.time() - t0, 1), "files": files}
    return {"name": name, "error": "timeout"}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("크로마키 캐릭터 시트 3종 / seed %d / 초록 배경\n" % SEED, flush=True)
    for tag, w, h, framing, extra_neg in SHOTS:
        if tag.endswith("_SKIP"):
            print(">>> %s  건너뜀" % tag, flush=True)
            continue
        prompt = "%s. %s %s" % (STYLE, CHAR, framing)
        neg = NEG_BASE + extra_neg
        print(">>> %s  (%dx%d)" % (tag, w, h), flush=True)
        r = run(tag, graph(prompt, neg, w, h, tag))
        if "error" in r:
            print("    실패: %s" % r["error"][:200], flush=True)
            continue
        print("    %ss" % r["seconds"], flush=True)
        for fn in r["files"]:
            src = os.path.join(COMFY_OUT, fn)
            dst = os.path.join(OUTDIR, tag + ".png")
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
                print("    -> %s" % dst, flush=True)


if __name__ == "__main__":
    main()
