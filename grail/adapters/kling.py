"""
Local drop-in replacement for grail/adapters/kling.py.

Backs generate_video with the local LTX-Video-0.9.8-13B-distilled checkpoint
(ckpt-distilled/, a HuggingFace `diffusers` LTXConditionPipeline snapshot),
loaded and run in-process via `diffusers` — no subprocess, no separate
LTX-Video repo checkout, no network calls, no API key.

Supports two quality tiers via the LTX_QUALITY env var:
  - "fast" (default): fewer denoising steps, quicker iteration.
  - "quality": more denoising steps, better detail/motion coherence.

The gen_2dhoi.py "mode" argument (std/pro, originally Kling's terminology)
is repurposed here: mode="pro" maps to LTX_QUALITY=quality, mode="std" maps
to "fast", unless LTX_QUALITY is explicitly set (which always wins).
"""

import os
import time

import torch
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR = os.environ.get("LTX_CKPT_DIR", os.path.join(_REPO_ROOT, "ckpt-distilled"))

LTX_HEIGHT = int(os.environ.get("LTX_HEIGHT", "480"))
LTX_WIDTH = int(os.environ.get("LTX_WIDTH", "704"))
LTX_FPS = int(os.environ.get("LTX_FPS", "24"))

_STEPS_BY_QUALITY = {"fast": 8, "quality": 30}
_GUIDANCE_BY_QUALITY = {"fast": 1.0, "quality": 3.0}

_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, distorted, low quality, worst quality, deformed, extra limbs, "
    "static image, watermark, text"
)

_pipeline = None


def _get_pipeline():
    """Lazily load and cache the LTXConditionPipeline singleton."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from diffusers import LTXConditionPipeline

    print(f"  [ltx-local] loading pipeline from {CKPT_DIR} ...")
    t0 = time.time()
    pipe = LTXConditionPipeline.from_pretrained(CKPT_DIR, torch_dtype=torch.bfloat16)
    # The 13B transformer alone is ~26GB in bf16 -- doesn't fit on a 24GB
    # card even with whole-submodule offload. Sequential (layer-level)
    # offload keeps peak VRAM tiny at the cost of throughput.
    pipe.enable_sequential_cpu_offload()
    print(f"  [ltx-local] pipeline ready in {time.time() - t0:.0f}s")
    _pipeline = pipe
    return _pipeline


def _resolve_quality(mode):
    env_override = os.environ.get("LTX_QUALITY")
    if env_override in _STEPS_BY_QUALITY:
        return env_override
    return "quality" if mode == "pro" else "fast"


def _nearest_valid_num_frames(duration, fps=LTX_FPS):
    """LTX-Video requires num_frames % 8 == 1 (e.g. 121, 161)."""
    target = max(9, int(float(duration) * fps))
    k = round((target - 1) / 8)
    return max(9, k * 8 + 1)


def generate_video(
    input_image,
    prompt,
    out_dir,
    name,
    duration=5,
    mode="std",
    model_name=None,
    image_tail_path=None,
):
    """
    Generate a video from a starting (first) frame + text prompt using the
    local LTX-Video checkpoint, hard-conditioned on that first frame.

    Mirrors the original Kling adapter's signature/return convention:
    returns the output video path on success, None on failure (caller in
    gen_2dhoi.py retries on None).

    Quality tier is chosen via `mode` ("std" -> fast, "pro" -> quality) or
    overridden globally by the LTX_QUALITY env var.

    If image_tail_path is given, the video is also conditioned on that image
    at (approximately) the last frame.
    """
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    from diffusers.utils import export_to_video

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{name}.mp4")

    quality = _resolve_quality(mode)
    num_inference_steps = _STEPS_BY_QUALITY[quality]
    guidance_scale = _GUIDANCE_BY_QUALITY[quality]
    num_frames = _nearest_valid_num_frames(duration)

    try:
        pipe = _get_pipeline()

        first_image = Image.open(input_image).convert("RGB").resize((LTX_WIDTH, LTX_HEIGHT))
        conditions = [LTXVideoCondition(image=first_image, frame_index=0)]

        if image_tail_path is not None:
            last_frame_idx = (num_frames - 1) // 8 * 8
            tail_image = (
                Image.open(image_tail_path).convert("RGB").resize((LTX_WIDTH, LTX_HEIGHT))
            )
            conditions.append(LTXVideoCondition(image=tail_image, frame_index=last_frame_idx))

        print(
            f"  [ltx-local] quality={quality} steps={num_inference_steps} "
            f"frames={num_frames} {LTX_WIDTH}x{LTX_HEIGHT} tail={image_tail_path is not None}"
        )
        t0 = time.time()
        generator = torch.Generator(device="cpu").manual_seed(42)
        output = pipe(
            conditions=conditions,
            prompt=prompt,
            negative_prompt=_DEFAULT_NEGATIVE_PROMPT,
            height=LTX_HEIGHT,
            width=LTX_WIDTH,
            num_frames=num_frames,
            frame_rate=LTX_FPS,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        frames = output.frames[0]
        export_to_video(frames, output_path, fps=LTX_FPS)
        print(f"  [ltx-local] done in {time.time() - t0:.0f}s -> {output_path}")
        return output_path

    except Exception as e:
        print(f"  [ltx-local] generation failed: {e}")
        return None
