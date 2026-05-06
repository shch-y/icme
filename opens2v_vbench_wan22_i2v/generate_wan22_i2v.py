import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _prompt_stem_from_manifest(row: Dict[str, Any]) -> str:
    stem = row.get("prompt_stem") or row.get("prompt") or "empty_prompt"
    stem = str(stem).strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def _write_video_mp4(frames_tchw: torch.Tensor, out_path: str, fps: int) -> None:
    """
    frames_tchw: uint8 tensor [T,C,H,W] in RGB.
    """
    import torchvision

    _ensure_dir(os.path.dirname(out_path))
    frames_thwc = frames_tchw.permute(0, 2, 3, 1).contiguous().cpu()
    torchvision.io.write_video(
        out_path,
        frames_thwc,
        fps=float(fps),
        video_codec="h264",
        options={"crf": "10"},
    )


@dataclass
class GenConfig:
    model: str
    device: str
    dtype: str
    height: int
    width: int
    num_frames: int
    fps: int
    num_inference_steps: int
    guidance_scale: float


def _load_wan22_i2v_pipeline(model: str, device: str, dtype: torch.dtype):
    """
    Best-effort loader.
    - If `diffusers` provides a native pipeline for Wan2.2 I2V, we use it.
    - Otherwise, we try a generic Image-to-Video pipeline entry point.
    """
    # This script supports diffusers-format checkpoints only.
    # For the official Wan2.2 weight folders (no `model_index.json`), use:
    # `opens2v_vbench_wan22_i2v/generate_wan22_official.py --task i2v-A14B --ckpt_dir /path/to/Wan2.2-I2V-A14B`
    if os.path.isdir(model) and not os.path.exists(os.path.join(model, "model_index.json")):
        raise OSError(
            f"No `model_index.json` found in {model}. This looks like an official Wan2.2 checkpoint folder, "
            "not a diffusers pipeline. Please use `generate_wan22_official.py` instead."
        )

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=dtype)
    pipe.to(device)
    if hasattr(pipe, "enable_model_cpu_offload"):
        # safe for single-GPU large models
        pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="Wan-AI/Wan2.2-I2V-A14B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--num_videos_per_prompt", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=1234)

    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=6.0)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    videos_dir = os.path.join(out_dir, "videos")
    images_dir = os.path.join(out_dir, "images")
    _ensure_dir(videos_dir)
    _ensure_dir(images_dir)

    rows = _load_manifest(args.manifest)
    if len(rows) == 0:
        raise SystemExit(f"Empty manifest: {args.manifest}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    cfg = GenConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    pipe = _load_wan22_i2v_pipeline(cfg.model, cfg.device, dtype)

    for row_idx, row in enumerate(rows):
        prompt = str(row.get("prompt") or row.get("cap") or "").strip()
        if not prompt:
            continue

        img_path = row.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue

        prompt_stem = _prompt_stem_from_manifest(row)
        # ensure the condition image exists in the final folder (VBench needs it there)
        final_img_path = os.path.join(images_dir, f"{prompt_stem}.jpg")
        if not os.path.exists(final_img_path):
            Image.open(img_path).convert("RGB").save(final_img_path, format="JPEG", quality=95, subsampling=0)

        image = Image.open(final_img_path).convert("RGB")

        for k in range(args.num_videos_per_prompt):
            out_path = os.path.join(videos_dir, f"{prompt_stem}-{k}.mp4")
            if os.path.exists(out_path):
                continue

            seed = args.base_seed + row_idx * 10_000 + k
            generator = torch.Generator(device=cfg.device).manual_seed(seed)

            # Support both I2V and T2V pipelines:
            # - I2V: expects `image=...`
            # - T2V: does not accept `image`, and may or may not accept `num_frames`
            with torch.inference_mode():
                try:
                    result = pipe(
                        prompt=prompt,
                        image=image,
                        generator=generator,
                        height=cfg.height,
                        width=cfg.width,
                        num_inference_steps=cfg.num_inference_steps,
                        guidance_scale=cfg.guidance_scale,
                        num_frames=cfg.num_frames,
                    )
                except TypeError:
                    try:
                        result = pipe(
                            prompt=prompt,
                            generator=generator,
                            height=cfg.height,
                            width=cfg.width,
                            num_inference_steps=cfg.num_inference_steps,
                            guidance_scale=cfg.guidance_scale,
                            num_frames=cfg.num_frames,
                        )
                    except TypeError:
                        result = pipe(
                            prompt=prompt,
                            generator=generator,
                            height=cfg.height,
                            width=cfg.width,
                            num_inference_steps=cfg.num_inference_steps,
                            guidance_scale=cfg.guidance_scale,
                        )

            frames = None
            # Common diffusers video outputs
            if hasattr(result, "frames") and result.frames is not None:
                frames = result.frames
            elif hasattr(result, "videos") and result.videos is not None:
                frames = result.videos

            if frames is None:
                raise RuntimeError("Pipeline output does not contain `frames` or `videos`.")

            # Normalize to uint8 tensor [T,C,H,W]
            if isinstance(frames, list):
                # list of PIL or numpy
                import numpy as np

                arr = []
                for fr in frames:
                    if isinstance(fr, Image.Image):
                        fr = np.array(fr.convert("RGB"))
                    arr.append(fr)
                frames = torch.from_numpy(np.stack(arr, axis=0))

            if isinstance(frames, torch.Tensor):
                # common shapes: [B,T,C,H,W] or [T,H,W,C] or [T,C,H,W]
                if frames.ndim == 5:
                    frames = frames[0]
                if frames.ndim == 4 and frames.shape[-1] == 3:
                    frames = frames.permute(0, 3, 1, 2)
                # if float in [0,1], convert to uint8
                if frames.dtype != torch.uint8:
                    frames = (frames.clamp(0, 1) * 255).to(torch.uint8)
            else:
                raise RuntimeError(f"Unsupported frames type: {type(frames)}")

            _write_video_mp4(frames, out_path, fps=cfg.fps)

        if (row_idx + 1) % 10 == 0:
            print(f"Generated videos for {row_idx + 1}/{len(rows)} prompts")

    print(f"Done. Videos at {videos_dir}")
    print(f"Images at {images_dir}")


if __name__ == "__main__":
    main()

