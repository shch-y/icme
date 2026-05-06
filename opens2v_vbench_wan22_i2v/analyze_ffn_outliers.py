import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
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


def _size_str(height: int, width: int) -> str:
    return f"{height}*{width}"


@dataclass
class DistStat:
    x_abs_samples: List[np.ndarray]
    w_abs_samples: List[np.ndarray]
    x_col_max: np.ndarray
    w_col_max: np.ndarray
    n_x_elems: int
    n_w_elems: int


class OutlierAnalyzer:
    def __init__(self, sample_cap_per_call: int = 16384, topk_plots: int = 10):
        self.sample_cap_per_call = int(sample_cap_per_call)
        self.topk_plots = int(topk_plots)
        self.stats: Dict[str, DistStat] = {}

    def _sample_abs(self, t: torch.Tensor) -> np.ndarray:
        x = t.detach().float().abs().reshape(-1)
        if x.numel() > self.sample_cap_per_call:
            idx = torch.randperm(x.numel(), device=x.device)[: self.sample_cap_per_call]
            x = x[idx]
        return x.cpu().numpy()

    def _sample_weight_abs(self, w: torch.Tensor) -> np.ndarray:
        x = w.detach().float().abs().reshape(-1)
        if x.numel() > self.sample_cap_per_call:
            idx = torch.randperm(x.numel(), device=x.device)[: self.sample_cap_per_call]
            x = x[idx]
        return x.cpu().numpy()

    def _make_hook(self, key: str):
        def hook(module: torch.nn.Module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return
            if x.ndim < 2:
                return

            x2 = x.detach().float().reshape(-1, x.shape[-1])
            x_col = x2.abs().amax(dim=0).cpu().numpy()
            w = module.weight.detach().float()
            w_col = w.abs().amax(dim=0).cpu().numpy()

            if key not in self.stats:
                self.stats[key] = DistStat(
                    x_abs_samples=[],
                    w_abs_samples=[],
                    x_col_max=x_col.copy(),
                    w_col_max=w_col.copy(),
                    n_x_elems=0,
                    n_w_elems=0,
                )
            st = self.stats[key]
            st.x_abs_samples.append(self._sample_abs(x))
            st.w_abs_samples.append(self._sample_weight_abs(w))
            st.x_col_max = np.maximum(st.x_col_max, x_col)
            st.w_col_max = np.maximum(st.w_col_max, w_col)
            st.n_x_elems += int(x.numel())
            st.n_w_elems += int(w.numel())

        return hook

    def register_model(self, model: torch.nn.Module, prefix: str):
        hooks = []
        for bi, blk in enumerate(getattr(model, "blocks", [])):
            if not hasattr(blk, "ffn"):
                continue
            for fi in (0, 2):
                try:
                    lin = blk.ffn[fi]
                except Exception:
                    continue
                if isinstance(lin, torch.nn.Linear):
                    key = f"{prefix}.block_{bi:02d}.ffn.{fi}"
                    hooks.append(lin.register_forward_pre_hook(self._make_hook(key)))
        return hooks

    def dump(self, out_dir: str):
        _ensure_dir(out_dir)
        out_json = {}
        thr = [2, 4, 6, 8, 10]
        cache_for_plot: Dict[str, Dict[str, np.ndarray]] = {}

        for key, st in self.stats.items():
            x_abs = np.concatenate(st.x_abs_samples) if st.x_abs_samples else np.array([], dtype=np.float32)
            w_abs = np.concatenate(st.w_abs_samples) if st.w_abs_samples else np.array([], dtype=np.float32)
            if x_abs.size == 0 or w_abs.size == 0:
                continue

            x_top128_idx = np.argsort(st.x_col_max)[-128:][::-1]
            w_top128_idx = np.argsort(st.w_col_max)[-128:][::-1]

            out_json[key] = {
                "n_x_elems_seen": st.n_x_elems,
                "n_w_elems_seen": st.n_w_elems,
                "x_abs_mean": float(x_abs.mean()),
                "x_abs_p99": float(np.quantile(x_abs, 0.99)),
                "w_abs_mean": float(w_abs.mean()),
                "w_abs_p99": float(np.quantile(w_abs, 0.99)),
                "x_outlier_ratio": {str(t): float((x_abs > t).mean()) for t in thr},
                "w_outlier_ratio": {str(t): float((w_abs > t).mean()) for t in thr},
                "x_top128_col_idx": x_top128_idx.tolist(),
                "x_top128_col_score": st.x_col_max[x_top128_idx].tolist(),
                "w_top128_col_idx": w_top128_idx.tolist(),
                "w_top128_col_score": st.w_col_max[w_top128_idx].tolist(),
            }
            cache_for_plot[key] = {
                "x_abs": x_abs,
                "w_abs": w_abs,
                "x_col_max_sorted": np.sort(st.x_col_max)[::-1][:512],
                "w_col_max_sorted": np.sort(st.w_col_max)[::-1][:512],
            }

        with open(os.path.join(out_dir, "ffn_outlier_stats.json"), "w", encoding="utf-8") as f:
            json.dump(out_json, f, indent=2, ensure_ascii=False)

        # Write a compact ranking summary for quick inspection.
        rank_rows = []
        for key, v in out_json.items():
            rank_rows.append(
                {
                    "layer": key,
                    "x_outlier_ratio@6": v["x_outlier_ratio"]["6"],
                    "x_outlier_ratio@8": v["x_outlier_ratio"]["8"],
                    "x_outlier_ratio@10": v["x_outlier_ratio"]["10"],
                    "w_outlier_ratio@6": v["w_outlier_ratio"]["6"],
                    "w_outlier_ratio@8": v["w_outlier_ratio"]["8"],
                    "w_outlier_ratio@10": v["w_outlier_ratio"]["10"],
                    "x_abs_p99": v["x_abs_p99"],
                    "w_abs_p99": v["w_abs_p99"],
                }
            )
        rank_rows.sort(key=lambda r: (r["x_outlier_ratio@10"], r["x_outlier_ratio@8"], r["x_outlier_ratio@6"]), reverse=True)
        with open(os.path.join(out_dir, "ffn_outlier_ranking.json"), "w", encoding="utf-8") as f:
            json.dump(rank_rows, f, indent=2, ensure_ascii=False)

        # Plot only top-K most outlier layers to avoid too many figures.
        topk = max(0, self.topk_plots)
        selected_layers = [r["layer"] for r in rank_rows[:topk]]
        for key in selected_layers:
            if key not in cache_for_plot:
                continue
            c = cache_for_plot[key]
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            axes[0, 0].hist(c["x_abs"], bins=120, log=True)
            axes[0, 0].set_title(f"{key} - |x| histogram")
            axes[0, 1].hist(c["w_abs"], bins=120, log=True)
            axes[0, 1].set_title(f"{key} - |weight| histogram")
            axes[1, 0].plot(c["x_col_max_sorted"])
            axes[1, 0].set_title(f"{key} - x column max(|x|) top512")
            axes[1, 1].plot(c["w_col_max_sorted"])
            axes[1, 1].set_title(f"{key} - w column max(|w|) top512")
            for ax in axes.ravel():
                ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{key.replace('.', '_')}.png"), dpi=160)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--wan_repo", default="/home/chenyidong/train/Wan2.2")
    ap.add_argument("--task", choices=["t2v-A14B", "i2v-A14B"], default="i2v-A14B")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frame_num", type=int, default=81)
    ap.add_argument("--sample_steps", type=int, default=1, help="建议=1，仅分析一轮迭代")
    ap.add_argument("--sample_solver", choices=["unipc", "dpm++"], default="unipc")
    ap.add_argument("--sample_guide_scale", type=float, default=None)
    ap.add_argument("--sample_shift", type=float, default=None)
    ap.add_argument("--base_seed", type=int, default=1234)
    ap.add_argument("--offload_model", type=str, default="True")
    ap.add_argument("--max_prompts", type=int, default=1)
    ap.add_argument("--out_dir", default="runs/opens2v_1024/ffn_outlier_analysis")
    ap.add_argument("--sample_cap_per_call", type=int, default=16384)
    ap.add_argument("--topk_plots", type=int, default=10, help="Only save plots for top-K most outlier layers.")
    args = ap.parse_args()

    rows = _load_manifest(args.manifest)
    if not rows:
        raise SystemExit(f"Empty manifest: {args.manifest}")

    sys.path.insert(0, os.path.abspath(args.wan_repo))
    import wan
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import str2bool

    args.offload_model = str2bool(args.offload_model)
    cfg = WAN_CONFIGS[args.task]
    size_key = _size_str(args.height, args.width)
    if size_key not in SIZE_CONFIGS:
        raise SystemExit(f"Unsupported size: {size_key}")

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    if args.task == "t2v-A14B":
        pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
            init_on_cpu=True,
            convert_model_dtype=True,
        )
    else:
        pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
            init_on_cpu=True,
            convert_model_dtype=True,
        )

    analyzer = OutlierAnalyzer(sample_cap_per_call=args.sample_cap_per_call, topk_plots=args.topk_plots)
    hooks = []
    hooks += analyzer.register_model(pipeline.low_noise_model, "low_noise_model")
    hooks += analyzer.register_model(pipeline.high_noise_model, "high_noise_model")

    try:
        n = 0
        for row in rows:
            if args.max_prompts and n >= args.max_prompts:
                break
            prompt = str(row.get("prompt") or row.get("cap") or "").strip()
            if not prompt:
                continue
            seed = args.base_seed + n
            if args.task == "t2v-A14B":
                _ = pipeline.generate(
                    prompt,
                    size=SIZE_CONFIGS[size_key],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=seed,
                    offload_model=args.offload_model,
                )
            else:
                img_path = row.get("image_path")
                if not img_path or not os.path.exists(img_path):
                    continue
                img = Image.open(img_path).convert("RGB")
                _ = pipeline.generate(
                    prompt,
                    img,
                    max_area=MAX_AREA_CONFIGS[size_key],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=seed,
                    offload_model=args.offload_model,
                )
            n += 1
    finally:
        for h in hooks:
            h.remove()

    analyzer.dump(args.out_dir)
    print(f"Saved analysis to: {args.out_dir}")
    print(f"JSON: {os.path.join(args.out_dir, 'ffn_outlier_stats.json')}")


if __name__ == "__main__":
    main()

