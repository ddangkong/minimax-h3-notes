# -*- coding: utf-8 -*-
"""참조가 실제로 작동하는가 — 결정적 테스트.

지금까지 "Ref2V가 정체성을 옮긴다"고 말해왔지만 근거가 약했다. 프롬프트에도
같은 캐릭터를 묘사해뒀으니, 참조 없이 프롬프트만으로 나온 결과와 구분이 되지
않는다. 참조가 무시되고 있어도 똑같이 보였을 것이다.

프롬프트에서 캐릭터 묘사를 완전히 비우고 참조만 남긴다. 프롬프트는 장면만
말한다. 그러면:

  참조가 작동한다면 -> 은발 포니테일 소녀가 나온다
  참조가 무시된다면 -> 아무 사람이나 나온다 (혹은 사람이 없다)

이보다 명확한 판별은 없다. 세 조건을 같은 시드로 돌려 비교한다:

  A  참조 3장 + 캐릭터 묘사 없는 프롬프트   <- 참조만의 힘
  B  참조 없음 + 캐릭터 묘사 없는 프롬프트   <- 대조군
  C  참조 없음 + 캐릭터 묘사 있는 프롬프트   <- 프롬프트만의 힘
"""
import sys
import os
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench5
import bench_chain as bc

FF = bc.FF
OUTDIR = r"D:\h3\output\ref_test"
COMFY_OUT = r"D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\output\reftest"
FRAMES = 124
W, H = 768, 1344
SEED = 777

REF_DIT = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
REF_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

# 장면만 말한다. 인물의 생김새·머리색·의상을 일절 언급하지 않는다.
SCENE_ONLY = (
    "Japanese anime style, vibrant cel-shaded 2D animation, high budget studio anime, "
    "electric cobalt sky, strong backlight, high contrast, vivid saturated colours, "
    "crisp anime line art. Vertical composition. A girl stands in a vast summer meadow "
    "of tall grass and dandelion seed heads, looking up at the sky. "
    "Audio: soft wind through grass."
)

# 비교용: 캐릭터를 프롬프트로 상세히 묘사한 버전
CHAR = ("a teenage girl with long silver-white hair in a high ponytail, large luminous "
        "violet eyes, pale skin, wearing a white and gold sleeveless top, a short white "
        "skirt with gold trim, white thigh-high boots, and a translucent iridescent cape")
SCENE_WITH_CHAR = SCENE_ONLY.replace(
    "A girl stands in a vast summer meadow",
    "%s stands in a vast summer meadow" % CHAR)

NEG = ("photorealistic, 3d render, cgi, live action, blurry, dark, desaturated, "
       "deformed face, extra limbs, extra fingers, watermark, text, "
       "horizontal composition, multiple people")


def graph(prompt, use_refs, tag):
    g = {
        "6":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": REF_DIT, "weight_dtype": "default"}},
        "7":  {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["6", 0], "strength_model": 1.0, "lora_name": REF_LORA}},
        "13": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                          "type": "minimax", "device": "default"}},
        "11": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "24": {"class_type": "VAELoader",
               "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["13", 0], "text": NEG}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9":  {"class_type": "BasicScheduler",
               "inputs": {"model": ["7", 0], "scheduler": "simple",
                          "steps": 8, "denoise": 1.0}},
        "14": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0],
                          "sigmas": ["9", 0], "latent_image": ["41", 1]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo",
               "inputs": {"images": ["10", 0], "fps": 24.0, "audio": ["23", 0]}},
        "92": {"class_type": "SaveVideo",
               "inputs": {"video": ["91", 0], "filename_prefix": "reftest/" + tag,
                          "format": "auto", "codec": "auto"}},
    }
    ref_node = {"clip": ["13", 0], "vae": ["11", 0], "audio_vae": ["24", 0],
                "prompt": prompt, "width": W, "height": H,
                "length": FRAMES, "ref_image_size": "match"}
    if use_refs:
        g["40"] = {"class_type": "LoadImage", "inputs": {"image": "cs_face_front.png"}}
        g["42"] = {"class_type": "LoadImage", "inputs": {"image": "cs_face_side.png"}}
        g["43"] = {"class_type": "LoadImage", "inputs": {"image": "cs_body_full.png"}}
        ref_node["ref_images.ref_image_0"] = ["40", 0]
        ref_node["ref_images.ref_image_1"] = ["42", 0]
        ref_node["ref_images.ref_image_2"] = ["43", 0]
    g["41"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": ref_node}
    g["16"] = {"class_type": "CFGGuider",
               "inputs": {"model": ["7", 0], "positive": ["41", 0],
                          "negative": ["21", 0], "cfg": 1.0}}
    return g


RUNS = [
    ("A_ref_noChar",  SCENE_ONLY,      True,  "참조3장 + 캐릭터묘사 없음"),
    ("B_noref_noChar", SCENE_ONLY,     False, "참조없음 + 캐릭터묘사 없음"),
    ("C_noref_char",  SCENE_WITH_CHAR, False, "참조없음 + 캐릭터묘사 있음"),
]


def main():
    bench5.FRAMES = FRAMES
    os.makedirs(OUTDIR, exist_ok=True)
    print("참조 작동 검증 / 시드 %d 고정 / %dx%d\n" % (SEED, W, H), flush=True)
    for tag, prompt, use_refs, desc in RUNS:
        print(">>> %s  (%s)" % (tag, desc), flush=True)
        r = bench5.run(tag, graph(prompt, use_refs, tag), timeout=2400)
        if "error" in r:
            print("    실패: %s" % r["error"][:250], flush=True)
            continue
        print("    %ss" % r["seconds"], flush=True)
        src = os.path.join(COMFY_OUT, r["files"][0])
        dst = os.path.join(OUTDIR, tag + ".mp4")
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)


if __name__ == "__main__":
    main()
