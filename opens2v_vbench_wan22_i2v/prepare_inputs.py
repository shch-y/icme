import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


def _safe_stem(text: str, max_len: int = 160) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\\/:\n\r\t]+", " ", text)
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff _.,;!?'\"()\-]+", "", text)
    text = text.strip().strip(".")
    if not text:
        text = "empty_prompt"
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_first_frame_jpg(video_path: str, out_path: str) -> Tuple[int, int]:
    """
    Decode first frame to JPG. Returns (width, height).
    Uses decord (already required by VBench) for robust decoding.
    """
    from decord import VideoReader
    import decord

    decord.bridge.set_bridge("native")
    vr = VideoReader(video_path, num_threads=1)
    frame0 = vr[0].asnumpy()  # HWC uint8
    img = Image.fromarray(frame0)
    _ensure_dir(os.path.dirname(out_path))
    img.save(out_path, format="JPEG", quality=95, subsampling=0)
    return img.size  # (W, H)


@dataclass
class Sample:
    source_video: str
    prompt: str
    image_path: str
    prompt_stem: str
    width: Optional[int] = None
    height: Optional[int] = None


def _iter_opens2v_from_modelscope() -> Iterable[Dict[str, Any]]:
    """
    Streams/iterates OpenS2V-5M samples via ModelScope MsDataset.

    We intentionally keep this function flexible: different ModelScope backends
    may expose different field names. We normalize downstream.
    """
    try:
        from modelscope.msdatasets import MsDataset
    except Exception as e:
        raise RuntimeError(
            "Failed to import ModelScope MsDataset. "
            "Please run inside your `icme` env and ensure deps installed, e.g. "
            "`pip install modelscope addict`."
        ) from e

    ds = MsDataset.load("AI-ModelScope/OpenS2V-5M")
    # ds can be either a Dataset-like object or a dict of splits.
    if isinstance(ds, dict):
        for _, split in ds.items():
            for item in split:
                yield item
    else:
        for item in ds:
            yield item


def _normalize_opens2v_item(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (video_path, caption).
    Tries common keys from OpenS2V-5M conversions.
    """
    video_path = (
        item.get("path")
        or item.get("video")
        or item.get("video_path")
        or item.get("mp4")
        or item.get("file")
    )
    caption = item.get("cap") or item.get("caption") or item.get("text") or item.get("prompt")
    if video_path is None or caption is None:
        raise KeyError(f"Cannot normalize item keys. Available keys: {sorted(item.keys())[:50]}")
    return str(video_path), str(caption)


def _sample_items(
    it: Iterable[Dict[str, Any]],
    max_samples: int,
    seed: int,
    shuffle_buffer: int = 10_000,
) -> List[Dict[str, Any]]:
    """
    Approximate uniform sampling without loading full dataset:
    reservoir sample after buffering.
    """
    rng = random.Random(seed)
    buf: List[Dict[str, Any]] = []
    out: List[Dict[str, Any]] = []

    for item in it:
        buf.append(item)
        if len(buf) >= shuffle_buffer:
            rng.shuffle(buf)
            while buf and len(out) < max_samples:
                out.append(buf.pop())
            if len(out) >= max_samples:
                break

    if len(out) < max_samples:
        rng.shuffle(buf)
        out.extend(buf[: max_samples - len(out)])

    return out[:max_samples]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True, help="output directory for inputs")
    ap.add_argument("--max_samples", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--prefer_local_index",
        type=str,
        default="/home/chenyidong/train/HiFloat4/datasets/OpenS2V-5M_to_mm.json",
        help="optional local json index to sample from (faster than MsDataset)",
    )
    ap.add_argument(
        "--path_prefix_from",
        type=str,
        default="/home/datasets/OpenS2V-5M",
        help="if local index paths use a different root, replace this prefix",
    )
    ap.add_argument(
        "--path_prefix_to",
        type=str,
        default="",
        help="replace local index root to this path (e.g. /home/dataset/OpenS2V-5M). "
        "If empty, no replacement is done.",
    )
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    images_dir = os.path.join(out_dir, "images")
    _ensure_dir(images_dir)

    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    items: List[Dict[str, Any]]
    if args.prefer_local_index and os.path.exists(args.prefer_local_index):
        with open(args.prefer_local_index, "r", encoding="utf-8") as f:
            local_items = json.load(f)
        rng = random.Random(args.seed)
        rng.shuffle(local_items)
        items = local_items[: args.max_samples]
    else:
        items = _sample_items(_iter_opens2v_from_modelscope(), args.max_samples, args.seed)

    seen_stems: Dict[str, int] = {}
    written = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for raw in items:
            try:
                video_path, cap = _normalize_opens2v_item(raw)
            except Exception:
                # fall back to common local index schema
                if "path" in raw and "cap" in raw:
                    video_path, cap = str(raw["path"]), str(raw["cap"])
                else:
                    continue

            if not os.path.exists(video_path):
                if args.path_prefix_to and video_path.startswith(args.path_prefix_from):
                    remapped = args.path_prefix_to + video_path[len(args.path_prefix_from) :]
                    if os.path.exists(remapped):
                        video_path = remapped
                if not os.path.exists(video_path):
                    # For this pipeline we need the video locally to decode first frame.
                    continue

            stem_base = _safe_stem(cap)
            dup = seen_stems.get(stem_base, 0)
            seen_stems[stem_base] = dup + 1
            prompt_stem = stem_base if dup == 0 else f"{stem_base} ({dup})"

            image_path = os.path.join(images_dir, f"{prompt_stem}.jpg")

            try:
                w, h = _extract_first_frame_jpg(video_path, image_path)
            except Exception:
                continue

            sample = Sample(
                source_video=video_path,
                prompt=cap,
                image_path=image_path,
                prompt_stem=prompt_stem,
                width=w,
                height=h,
            )
            mf.write(json.dumps(sample.__dict__, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.max_samples:
                break

    print(f"Wrote {written} samples to {manifest_path}")
    print(f"Images at {images_dir}")


if __name__ == "__main__":
    main()

