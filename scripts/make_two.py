# -*- coding: utf-8 -*-
"""새 캐릭터 2명, 각 15초 영상 1편씩.

참조 전달 형식이 고쳐졌으므로(ref_images.ref_image_N 평면 키) 이제 캐릭터
시트가 실제로 반영된다. 두 캐릭터 모두:

  1. Flux로 크로마키 캐릭터 시트 3장 (정면얼굴/측면얼굴/전신)
  2. 그 3장을 참조로 Ref2V 3컷 x 5.2초 = 15.5초

프롬프트에는 캐릭터 생김새를 쓰지 않는다. 참조가 담당하고, 프롬프트는 장면과
동작만 말한다. 이렇게 해야 참조가 실제로 일하는지가 결과에 드러난다.
"""
import sys
import os
import json
import time
import uuid
import urllib.request
import urllib.error
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench5

SERVER = "127.0.0.1:8188"
FF = r"C:\Users\sonbw\ffmpeg\bin\ffmpeg.exe"
OUT = r"D:\h3\output\two_films"
CS_OUT = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\two_cs"
VID_OUT = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\two_vid"
COMFY_IN = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\input"

W, H = 768, 1344
FRAMES = 124

REF_DIT = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
REF_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

CS_STYLE = ("Japanese anime style character reference sheet, vibrant cel-shaded 2D animation, "
            "high budget studio anime, crisp clean anime line art, bright even neutral studio "
            "lighting, richly saturated character colours. Plain flat solid chroma key green "
            "screen background, uniform pure green backdrop, completely empty background with "
            "no scenery, no props, no ground, no shadows on the background. Sharp focus")

CS_NEG = ("photorealistic, 3d render, cgi, live action, blurry, dark, desaturated, "
          "deformed face, distorted anatomy, extra limbs, extra fingers, watermark, text, "
          "multiple people, scenery, landscape, sky, clouds, grass, room, wall, "
          "gradient background, lens flare, coloured lighting, shadow on background")

VID_LOOK = ("Japanese anime style, vibrant cel-shaded 2D animation, high budget studio anime, "
            "crisp anime line art, extremely vivid hyper saturated colours, iridescent "
            "prismatic highlights, strong backlight, anamorphic lens flare, high contrast, "
            "luminous ethereal atmosphere, sparkling light particles. Vertical composition")

VID_NEG = ("photorealistic, 3d render, cgi, live action, blurry, dark, desaturated, muted "
           "colours, washed out, deformed face, extra limbs, extra fingers, distorted anatomy, "
           "short legs, stubby limbs, oversized head, bad proportions, watermark, text, "
           "static, still image, horizontal composition, multiple people")

# 두 캐릭터. 생김새는 시트에만 쓰고 영상 프롬프트에는 쓰지 않는다.
CHARS = [
    {
        "key": "ember",
        "seed": 8801,
        "desc": ("A teenage girl with short messy crimson-red hair and bright amber eyes, "
                 "tanned skin, wearing a fitted black bodysuit with glowing orange circuit "
                 "lines, a torn asymmetric jacket, and heavy black boots"),
        "beats": [
            ("1. 각성", "She stands in the centre of a ruined neon city street at night, "
                        "head lowered, then snaps her head up as orange light ignites along "
                        "her body. Sparks drift around her. "
                        "Audio: a low electrical hum rising into a sharp crack."),
            ("2. 질주", "Ultra wide angle, camera racing alongside her. She sprints at full "
                        "speed down the ruined street, trailing orange light, debris and "
                        "rubble rushing past the camera in the foreground. "
                        "Audio: pounding footsteps, roaring wind, an urgent drum pulse."),
            ("3. 도약", "Low angle from the ground. She leaps high off a broken car and "
                        "twists in mid-air against the night sky, orange energy flaring out "
                        "behind her in a long arc. "
                        "Audio: an explosive impact, then a soaring musical swell."),
        ],
    },
    {
        "key": "tide",
        "seed": 9902,
        "desc": ("A teenage boy with medium-length deep blue hair and pale green eyes, "
                 "fair skin, wearing a flowing white and teal robe with silver trim, "
                 "loose sleeves, and bare feet"),
        "beats": [
            ("1. 부름", "He stands ankle-deep in a vast shallow sea at dawn, arms lowered, "
                        "the mirror-flat water reflecting a pink and gold sky. He slowly "
                        "raises one hand and the water begins to ripple outward. "
                        "Audio: gentle lapping water, a soft rising chime."),
            ("2. 융기", "Ultra wide angle, low angle just above the water. Towering walls of "
                        "water rise around him and curve overhead, droplets streaming past "
                        "the camera in the foreground, dawn light refracting through them. "
                        "Audio: a deep roar of moving water, an orchestral swell."),
            ("3. 정적", "Medium close-up. He closes his eyes and lowers his hand, and the "
                        "walls of water collapse into a fine glittering mist that drifts "
                        "around him, the sea returning to mirror stillness. "
                        "Audio: the roar falling away into soft rain, a calm sustained note."),
        ],
    },
]

CS_SHOTS = [
    ("face_front", 1024, 1024,
     "Close-up of the face, front view facing the camera directly, symmetrical, neutral calm "
     "expression, mouth closed, eyes open looking straight at the viewer, both eyes clearly "
     "visible, head filling most of the frame.",
     ", side view, profile, looking away, full body, legs, distant, small head"),
    ("face_side", 1024, 1024,
     "Close-up of the face in strict profile, side view, looking straight ahead to the side, "
     "showing the line of the nose, lips and chin and the shape of the back of the head, "
     "head filling most of the frame. Neutral expression, mouth closed.",
     ", front view, facing the camera, looking at the viewer, full body, legs, distant, small head"),
    ("body_full", 896, 1408,
     "Full body reference, complete figure from the top of the head down to the feet, both "
     "feet fully visible in frame, generous empty space above the head and below the feet, "
     "the whole body occupying only the middle portion of the tall frame. Standing straight "
     "facing the camera, arms relaxed at the sides, face clearly visible. Natural slender "
     "anime proportions, long well proportioned legs, correct anatomy.",
     ", close-up, headshot, bust shot, cropped legs, cropped feet, feet out of frame, "
     "cut off at the waist, zoomed in, short legs, stubby limbs, oversized head, sitting"),
]


def post(g, timeout=2400):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"http://{SERVER}/prompt",
            data=json.dumps({"prompt": g, "client_id": str(uuid.uuid4())}).encode(),
            headers={"Content-Type": "application/json"})
        pid = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:400]}
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
                return {"error": str(st.get("messages"))[:400]}
            files = []
            for o in h[pid].get("outputs", {}).values():
                for k in ("images", "video", "gifs"):
                    files += [f.get("filename") for f in (o.get(k) or [])]
            return {"seconds": round(time.time() - t0, 1), "files": files}
    return {"error": "timeout"}


def sheet_graph(prompt, neg, w, h, tag, seed):
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "flux1-krea-dev_fp8_scaled.safetensors",
                         "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                         "clip_name2": "clip_l.safetensors", "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": neg}},
        "6": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": 3.5}},
        "7": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["6", 0], "negative": ["5", 0],
                         "latent_image": ["7", 0], "seed": seed, "steps": 28, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage",
               "inputs": {"images": ["9", 0], "filename_prefix": "two_cs/" + tag}},
    }


def video_graph(prompt, refs, tag, seed):
    ref_in = {"clip": ["13", 0], "vae": ["11", 0], "audio_vae": ["24", 0],
              "prompt": prompt, "width": W, "height": H,
              "length": FRAMES, "ref_image_size": "match"}
    g = {
        "6": {"class_type": "UNETLoader",
              "inputs": {"unet_name": REF_DIT, "weight_dtype": "default"}},
        "7": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["6", 0], "strength_model": 1.0, "lora_name": REF_LORA}},
        "13": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                          "type": "minimax", "device": "default"}},
        "11": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "24": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["13", 0], "text": VID_NEG}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "BasicScheduler",
              "inputs": {"model": ["7", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0],
                          "sigmas": ["9", 0], "latent_image": ["41", 1]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "fps": 24.0, "audio": ["23", 0]}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": "two_vid/" + tag,
                          "format": "auto", "codec": "auto"}},
    }
    for i, fn in enumerate(refs):
        nid = str(50 + i)
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": fn}}
        # 평면 점 표기여야 실행 엔진이 링크로 해석한다.
        ref_in["ref_images.ref_image_%d" % i] = [nid, 0]
    g["41"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": ref_in}
    g["16"] = {"class_type": "CFGGuider",
               "inputs": {"model": ["7", 0], "positive": ["41", 0],
                          "negative": ["21", 0], "cfg": 1.0}}
    return g


def main():
    os.makedirs(OUT, exist_ok=True)
    for ch in CHARS:
        key, seed, desc = ch["key"], ch["seed"], ch["desc"]
        cdir = os.path.join(OUT, key)
        os.makedirs(cdir, exist_ok=True)
        print("\n" + "=" * 60)
        print("캐릭터 [%s]" % key, flush=True)
        print("=" * 60)

        # --- 1. 캐릭터 시트 ---
        refs = []
        for tag, w, h, framing, extra_neg in CS_SHOTS:
            name = "%s_%s" % (key, tag)
            dst_in = os.path.join(COMFY_IN, name + ".png")
            if os.path.exists(dst_in):
                print("  시트 %s (있음)" % tag, flush=True)
                refs.append(name + ".png")
                continue
            print("  시트 %s ..." % tag, flush=True)
            r = post(sheet_graph("%s. %s %s" % (CS_STYLE, desc, framing),
                                 CS_NEG + extra_neg, w, h, name, seed))
            if "error" in r:
                print("    실패: %s" % r["error"][:200], flush=True)
                continue
            src = os.path.join(CS_OUT, r["files"][0])
            if os.path.exists(src):
                import shutil
                shutil.copy(src, dst_in)
                shutil.move(src, os.path.join(cdir, tag + ".png"))
                refs.append(name + ".png")
                print("    %ss" % r["seconds"], flush=True)

        if len(refs) < 3:
            print("  시트 부족, 건너뜀", flush=True)
            continue

        # --- 2. 영상 3컷 ---
        clips = []
        for i, (title, action) in enumerate(ch["beats"]):
            n = i + 1
            tag = "%s_cut%d" % (key, n)
            dst = os.path.join(cdir, "cut%d.mp4" % n)
            if os.path.exists(dst):
                print("  %s (있음)" % title, flush=True)
                clips.append(dst)
                continue
            print("  %s ..." % title, flush=True)
            prompt = "%s. %s" % (VID_LOOK, action)
            r = post(video_graph(prompt, refs, tag, seed + n))
            if "error" in r:
                print("    실패: %s" % r["error"][:250], flush=True)
                continue
            print("    %ss" % r["seconds"], flush=True)
            src = os.path.join(VID_OUT, r["files"][0])
            if os.path.exists(src):
                os.rename(src, dst)
                clips.append(dst)

        # --- 3. 합본 ---
        if len(clips) >= 2:
            lst = os.path.join(cdir, "_list.txt")
            with open(lst, "w", encoding="utf-8") as f:
                for p in clips:
                    f.write("file '%s'\n" % p.replace("\\", "/"))
            joined = os.path.join(OUT, "%s_%dcuts.mp4" % (key, len(clips)))
            subprocess.run([FF, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                            "-c:v", "libx264", "-crf", "16", "-c:a", "aac", joined], check=False)
            if os.path.exists(joined):
                print("  합본: %s (%.1f초)" % (joined, len(clips) * FRAMES / 24.0), flush=True)


if __name__ == "__main__":
    main()
