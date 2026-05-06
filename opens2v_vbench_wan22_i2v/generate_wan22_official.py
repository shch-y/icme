import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List

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


def _size_str(height: int, width: int) -> str:
    return f"{height}*{width}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True, help="e.g. /home/dataset/Wan2.2-T2V-A14B")
    ap.add_argument("--wan_repo", default="/home/chenyidong/train/Wan2.2", help="path to Wan2.2 repo")
    ap.add_argument("--task", choices=["t2v-A14B", "i2v-A14B"], default="t2v-A14B")
    ap.add_argument("--max_prompts", type=int, default=0, help="0 means no limit")
    ap.add_argument("--num_videos_per_prompt", type=int, default=1)
    ap.add_argument("--base_seed", type=int, default=1234)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frame_num", type=int, default=81)
    ap.add_argument("--sample_steps", type=int, default=None)
    ap.add_argument("--sample_guide_scale", type=float, default=None)
    ap.add_argument("--sample_shift", type=float, default=None)
    ap.add_argument("--sample_solver", choices=["unipc", "dpm++"], default="unipc")
    ap.add_argument("--offload_model", type=str, default="True")
    ap.add_argument(
        "--print_model",
        choices=["none", "summary"],
        default="summary",
        help='Print model structure. "summary" prints a compact overview; "none" disables printing.',
    )
    ap.add_argument(
        "--print_model_full",
        action="store_true",
        help="Also print full `repr(model)` for low/high noise models (very verbose).",
    )
    ap.add_argument(
        "--use_hifx4",
        action="store_true",
        help='Use HiFloat4 "hifx4" QLinear replacement for inference (replaces nn.Linear in Wan DiT blocks).',
    )
    ap.add_argument(
        "--hifx4_hadamard_rotate",
        action="store_true",
        help="(HiFloat4) Enable RoMeo-style Hadamard rotation: x <- x*Q before quantization, and W <- W*Q for weights.",
    )
    ap.add_argument(
        "--use_precision_aware_ffn0",
        action="store_true",
        help="Enable precision-aware wrapper for blocks.*.ffn[0] (default: off).",
    )
    ap.add_argument(
        "--use_precision_aware_ffn2",
        action="store_true",
        help="Enable the same precision-aware wrapper for blocks.*.ffn[2] (default: off).",
    )
    ap.add_argument(
        "--ffn_smoothquant",
        action="store_true",
        help=(
            "SmoothQuant-style per-channel smoothing on wrapped FFN linears (requires "
            "--use_precision_aware_ffn0 and/or --use_precision_aware_ffn2). "
            "Uses scales = act_scale^alpha / weight_scale^(1-alpha) like smoothquant/smooth.py, "
            "folded into weight after warmup and applied as x / scales at inference."
        ),
    )
    ap.add_argument(
        "--ffn_smoothquant_alpha",
        type=float,
        default=0.85,
        help="SmoothQuant alpha (same role as smooth_lm(..., alpha=...) in smoothquant_llama_demo.ipynb).",
    )
    ap.add_argument(
        "--hifx4_skip_first_n_blocks",
        type=int,
        default=0,
        help="When --use_hifx4 is enabled, skip quantization/replacement for the first N transformer blocks.",
    )
    ap.add_argument("--convert_model_dtype", action="store_true", default=True)
    ap.add_argument("--t5_cpu", action="store_true", default=False)
    args = ap.parse_args()

    rows = _load_manifest(args.manifest)
    if not rows:
        raise SystemExit(f"Empty manifest: {args.manifest}")

    out_dir = os.path.abspath(args.out_dir)
    videos_dir = os.path.join(out_dir, "videos")
    images_dir = os.path.join(out_dir, "images")
    _ensure_dir(videos_dir)
    _ensure_dir(images_dir)

    gen_py = os.path.join(args.wan_repo, "generate.py")
    if not os.path.exists(gen_py):
        raise SystemExit(f"generate.py not found at {gen_py}")

    # Run in-process to avoid re-loading checkpoints for each sample.
    sys.path.insert(0, os.path.abspath(args.wan_repo))
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    import wan
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import save_video, str2bool

    args.offload_model = str2bool(args.offload_model)
    if args.ffn_smoothquant and not (args.use_precision_aware_ffn0 or args.use_precision_aware_ffn2):
        logging.warning(
            "--ffn_smoothquant requires --use_precision_aware_ffn0 and/or --use_precision_aware_ffn2; disabling."
        )
        args.ffn_smoothquant = False
    # Note: When --use_hifx4 is enabled we will perform layer replacement on CPU first
    # to avoid extra GPU allocations during QLinear.transfer(), then rely on Wan's own
    # (offload/init_on_cpu) logic to move the active model to GPU for inference.

    if args.task not in WAN_CONFIGS:
        raise SystemExit(f"Unsupported task: {args.task}")

    cfg = WAN_CONFIGS[args.task]
    size_key = _size_str(args.height, args.width)
    if size_key not in SIZE_CONFIGS:
        raise SystemExit(f"Unsupported size {size_key}. Supported: {sorted(SIZE_CONFIGS.keys())}")

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        # cfg.sample_guide_scale can be float or tuple
        args.sample_guide_scale = cfg.sample_guide_scale

    logging.info(f"Creating {args.task} pipeline once (no repeated checkpoint loading).")
    if args.task == "t2v-A14B":
        pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=args.t5_cpu,
            init_on_cpu=True if args.use_hifx4 else True,
            convert_model_dtype=args.convert_model_dtype,
        )
    elif args.task == "i2v-A14B":
        pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=args.t5_cpu,
            init_on_cpu=True if args.use_hifx4 else True,
            convert_model_dtype=args.convert_model_dtype,
        )
    else:
        raise SystemExit(f"Only t2v-A14B/i2v-A14B supported here; got {args.task}")

    if args.print_model == "summary":
        print("\n" + "=" * 88)
        print("Wan2.2 pipeline structure")
        print("=" * 88)
        print(f"task: {args.task}")
        for name in ("low_noise_model", "high_noise_model", "vae", "text_encoder"):
            if hasattr(pipeline, name):
                obj = getattr(pipeline, name)
                print(f"{name}: {type(obj).__name__}")
        for name in ("low_noise_model", "high_noise_model"):
            if not hasattr(pipeline, name):
                continue
            model = getattr(pipeline, name)
            print(model)
            dim = getattr(model, "dim", None)
            ffn_dim = getattr(model, "ffn_dim", None)
            patch_size = getattr(model, "patch_size", None)
            num_layers = len(getattr(model, "blocks", [])) if hasattr(model, "blocks") else None
            num_heads = None
            try:
                if hasattr(model, "blocks") and len(model.blocks) > 0 and hasattr(model.blocks[0], "self_attn"):
                    num_heads = getattr(model.blocks[0].self_attn, "num_heads", None)
            except Exception:
                num_heads = None
            print(f"\n{name} config:")
            print(f"  dim={dim}, ffn_dim={ffn_dim}, num_layers={num_layers}, num_heads={num_heads}, patch_size={patch_size}")

        if args.print_model_full:
            print("\n" + "-" * 88)
            print("Full model repr (low_noise_model)")
            print("-" * 88)
            print(pipeline.low_noise_model)
            print("\n" + "-" * 88)
            print("Full model repr (high_noise_model)")
            print("-" * 88)
            print(pipeline.high_noise_model)
        print("=" * 88 + "\n")

    if args.use_hifx4:
        hif4_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HiFloat4", "hif4_gpu"))
        sys.path.insert(0, hif4_root)
        try:
            from quant_cy.utils.utils import replace_linear  # type: ignore
            from quant_cy.base.QTensor import quant_dequant_float  # type: ignore
            from quant_cy.base.QType import QType  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import HiFloat4 QLinear replacement. Expected import path: "
                f"{hif4_root}/quant_cy/utils/utils.py"
            ) from e

        if args.hifx4_hadamard_rotate:
            # Make RoMeo Hadamard utils importable for QLinear rotation.
            romeo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RoMeo-main"))
            sys.path.insert(0, romeo_root)

        # Make sure DiT weights are on CPU during replacement to avoid OOM.
        import torch

        if next(pipeline.low_noise_model.parameters()).device.type == "cuda":
            pipeline.low_noise_model.to("cpu")
        if next(pipeline.high_noise_model.parameters()).device.type == "cuda":
            pipeline.high_noise_model.to("cpu")
        torch.cuda.empty_cache()

        class PrecisionAwareFFNLinear(torch.nn.Module):
            """
            Precision-aware wrapper for an FFN `nn.Linear` (e.g. ffn[0] or ffn[2]).

            Warmup (first `warmup_steps` forwards):
            - Run in original dtype (bf16/fp16) with full-precision weights.
            - Record high-magnitude input columns for a small FP branch.

            After warmup:
            - Cache `ind` and `bfp = W[:, ind]` once.
            - Zero out columns `ind` in x and W for quant branch.
            - Compute: out = fp_linear(x[:, ind], bfp) + quant_linear(quant(x_zero), quant(W_zero))
            """

            def __init__(
                self,
                linear: torch.nn.Linear,
                *,
                warmup_steps: int = 3,
                threshold: float = 6.0,
                w_q: str = "hifx4",
                in_q: str = "hifx4",
                quant_force_fp32: bool = True,
                use_smoothquant: bool = False,
                smoothquant_alpha: float = 0.85,
            ) -> None:
                super().__init__()
                if not isinstance(linear, torch.nn.Linear):
                    raise TypeError(f"Expected nn.Linear, got {type(linear)}")
                self.in_features = linear.in_features
                self.out_features = linear.out_features
                self.warmup_steps = int(warmup_steps)
                self.threshold = float(threshold)
                self.use_smoothquant = bool(use_smoothquant)
                self.smoothquant_alpha = float(smoothquant_alpha)

                self.weight = torch.nn.Parameter(linear.weight.detach().clone())
                self.bias = torch.nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

                self._step = 0
                self._prepared = False

                # union mask across warmup steps (bool over input dims)
                self.register_buffer("_ind_mask", torch.zeros(self.in_features, dtype=torch.bool), persistent=True)
                self.register_buffer("_ind", torch.empty(0, dtype=torch.long), persistent=True)

                # SmoothQuant: per-input-channel max |x| during warmup (same role as act_scales in smoothquant)
                if self.use_smoothquant:
                    self.register_buffer("_act_col_max", torch.zeros(self.in_features, dtype=torch.float32))
                    self.register_buffer("_smooth_scales", torch.ones(self.in_features, dtype=torch.float32))
                else:
                    self._act_col_max = None  # type: ignore
                    self._smooth_scales = None  # type: ignore

                # cached fp and quant weights
                self.register_buffer("_bfp", torch.empty(0), persistent=True)  # [out, n_ind]
                self.register_buffer("_wq", torch.empty(0), persistent=True)  # [out, in]

                self._w_qtype = QType(w_q)
                self._in_qtype = QType(in_q)
                self._quant_force_fp32 = bool(quant_force_fp32)

            def _record_indices(self, x: torch.Tensor) -> None:
                # x: [..., in_features]
                # Pick the 128 columns with largest outliers (by max(|x|) per column),
                # then accumulate a union across warmup steps.
                x2 = x.reshape(-1, x.shape[-1])
                if self.use_smoothquant and self._act_col_max is not None:
                    with torch.no_grad():
                        am = x2.detach().float().abs().amax(dim=0).to(self._act_col_max.device)
                        torch.maximum(self._act_col_max, am, out=self._act_col_max)
                scores = x2.abs().amax(dim=0)
                k = min(256, int(scores.numel()))
                if k <= 0:
                    return
                topk_idx = torch.topk(scores, k=k, largest=True).indices
                cols = torch.zeros_like(self._ind_mask)
                cols[topk_idx.to(device=cols.device)] = True
                self._ind_mask |= cols

            def _block128x1_quant_dequant(self, mat: torch.Tensor, q: QType) -> torch.Tensor:
                # 128×1 blocks along the last dim (K). Each block has its own scale = max(|block|).
                # Per block: y = QDQ(x / s) * s, where QDQ is HiFloat4 quant_dequant_float.
                if mat.numel() == 0:
                    return mat
                blk = 128
                orig_shape = mat.shape
                k = orig_shape[-1]
                pad_k = (blk - (k % blk)) % blk
                m = torch.nn.functional.pad(mat, (0, pad_k), value=0.0) if pad_k else mat
                kp = m.shape[-1]
                nblk = kp // blk
                prefix = math.prod(m.shape[:-1]) if m.ndim > 0 else 1
                m2 = m.reshape(prefix, kp).view(prefix, nblk, blk).reshape(-1, blk)
                scales = m2.detach().float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
                normalized = (m2.float() / scales).to(dtype=m2.dtype)
                qd = quant_dequant_float(normalized, q, force_fp32=self._quant_force_fp32)
                out_flat = (qd.float() * scales).to(dtype=m2.dtype)
                out = out_flat.view(prefix, nblk, blk).reshape(prefix, kp)
                if pad_k:
                    out = out[..., :k]
                return out.reshape(orig_shape)

            def _prepare(self) -> None:
                if self._prepared:
                    return
                device = self.weight.device
                dtype = self.weight.dtype

                if self.use_smoothquant and self._act_col_max is not None:
                    # Same formula as smoothquant.smooth.smooth_ln_fcs / smooth_ln_fcs_llama_like for fc weights:
                    # scales = act_scales^alpha / weight_scales^(1-alpha)
                    act_scales = self._act_col_max.to(device=device, dtype=torch.float32).clamp(min=1e-5)
                    w_col = self.weight.detach().float().abs().max(dim=0).values.clamp(min=1e-5)
                    a = self.smoothquant_alpha
                    s = ((act_scales.pow(a)) / (w_col.pow(1.0 - a))).clamp(min=1e-5).to(dtype=dtype)
                    self._smooth_scales.copy_(s.to(device=self._smooth_scales.device, dtype=self._smooth_scales.dtype))
                    # Fold scales into weights: y = (x/s) @ (W * diag(s))^T + b
                    self.weight.data.mul_(s.unsqueeze(0).to(device=self.weight.device, dtype=self.weight.dtype))

                ind = torch.nonzero(self._ind_mask, as_tuple=False).flatten().to(device=device)
                self._ind = ind
                if ind.numel() == 0:
                    # no fp columns; quant-only
                    w_zero = self.weight
                    self._bfp = torch.empty((self.out_features, 0), device=device, dtype=dtype)
                else:
                    bfp = self.weight[:, ind].contiguous()
                    self._bfp = bfp
                    w_zero = self.weight.clone()
                    w_zero[:, ind] = 0

                # quantize weight once and cache dequantized weight for fast forward
                qp_w = self._w_qtype.dim(-1)
                self._wq = self._block128x1_quant_dequant(w_zero, qp_w).to(device=device)
                self._prepared = True

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self._step < self.warmup_steps:
                    self._record_indices(x)
                    self._step += 1
                    return torch.nn.functional.linear(x, self.weight, self.bias)

                if not self._prepared:
                    self._prepare()

                if self.use_smoothquant and self._smooth_scales is not None:
                    s = self._smooth_scales.to(device=x.device, dtype=x.dtype)
                    x_eff = x / s
                else:
                    x_eff = x

                # fp branch on selected columns
                if self._ind.numel() > 0:
                    afp = x_eff.index_select(dim=-1, index=self._ind)
                    out_fp = torch.nn.functional.linear(afp, self._bfp, None)
                    x_zero = x_eff.clone()
                    x_zero.index_fill_(dim=-1, index=self._ind, value=0)
                else:
                    out_fp = None
                    x_zero = x_eff

                # quant branch
                qp_in = self._in_qtype.dim(-1)
                x_q = self._block128x1_quant_dequant(x_zero, qp_in)
                out_q = torch.nn.functional.linear(x_q, self._wq, self.bias)

                if out_fp is None:
                    return out_q
                return out_q + out_fp

        def _block_idx_from_name(name: str):
            m = re.match(r"^blocks\.(\d+)\.", name)
            if m is None:
                return None
            return int(m.group(1))

        def _install_precision_aware_ffn(mod, ffn_idx: int) -> int:
            n = 0
            for bi, blk in enumerate(getattr(mod, "blocks", [])):
                if bi < args.hifx4_skip_first_n_blocks:
                    continue
                try:
                    ffn_lin = blk.ffn[ffn_idx]
                except Exception:
                    continue
                if isinstance(ffn_lin, torch.nn.Linear):
                    blk.ffn[ffn_idx] = PrecisionAwareFFNLinear(
                        ffn_lin,
                        warmup_steps=3,
                        threshold=6.0,
                        use_smoothquant=args.ffn_smoothquant,
                        smoothquant_alpha=args.ffn_smoothquant_alpha,
                    )
                    n += 1
            return n

        if args.use_precision_aware_ffn0:
            n_low = _install_precision_aware_ffn(pipeline.low_noise_model, 0)
            n_high = _install_precision_aware_ffn(pipeline.high_noise_model, 0)
            logging.info(f"Installed PrecisionAwareFFNLinear for ffn[0]: low_noise={n_low}, high_noise={n_high}.")

        if args.use_precision_aware_ffn2:
            n_low2 = _install_precision_aware_ffn(pipeline.low_noise_model, 2)
            n_high2 = _install_precision_aware_ffn(pipeline.high_noise_model, 2)
            logging.info(f"Installed PrecisionAwareFFNLinear for ffn[2]: low_noise={n_low2}, high_noise={n_high2}.")

        def _replace_selected_linear_only(mod) -> None:
            """
            Only replace selected Linear layers:
            - *.self_attn.{q,k,v,o}
            - *.cross_attn.{q,k,v,o}
            - text_embedding.{0,2}
            - time_embedding.{0,2}
            - time_projection.1
            """
            import torch.nn as nn

            keep_pat = re.compile(
                r"(^text_embedding\.(0|2)$)|(^time_embedding\.(0|2)$)|(^time_projection\.1$)"
                r"|(\.(self_attn|cross_attn)\.(q|k|v|o)$)"
            )
            exclude: List[str] = []
            total_linear = 0
            kept_linear = 0
            for n, m in mod.named_modules():
                if isinstance(m, nn.Linear):
                    total_linear += 1
                    blk_idx = _block_idx_from_name(n)
                    skip_by_block = blk_idx is not None and blk_idx < args.hifx4_skip_first_n_blocks
                    if keep_pat.search(n) is None or skip_by_block:
                        exclude.append(n)
                    else:
                        kept_linear += 1

            logging.info(
                f'Applying HiFloat4 to selected Linear layers: keeping {kept_linear}/{total_linear} (excluding {len(exclude)}).'
            )
            replace_linear(mod, "hifx4", in_Q="hifx4", quant_grad=False, exclude_layers=exclude)

        _replace_selected_linear_only(pipeline.low_noise_model)
        _replace_selected_linear_only(pipeline.high_noise_model)
        torch.cuda.empty_cache()

        if args.hifx4_hadamard_rotate:
            try:
                from quant_cy.layers.QLinear import QLinear  # type: ignore
            except Exception as e:
                raise RuntimeError("Failed to import HiFloat4 QLinear for rotation enablement.") from e

            def _enable_rotation(mod) -> int:
                n = 0
                for _, m in mod.named_modules():
                    if isinstance(m, QLinear):
                        m.enable_romeo_hadamard_rotation(rotate_weight=True)
                        n += 1
                return n

            n1 = _enable_rotation(pipeline.low_noise_model)
            n2 = _enable_rotation(pipeline.high_noise_model)
            logging.info(f"Enabled RoMeo Hadamard rotation for HiFloat4 QLinear layers: low_noise={n1}, high_noise={n2}.")

    for row_idx, row in enumerate(rows):
        if args.max_prompts and row_idx >= args.max_prompts:
            break
        prompt = str(row.get("prompt") or row.get("cap") or "").strip()
        if not prompt:
            continue

        prompt_stem = _prompt_stem_from_manifest(row)

        # VBench needs images in a dedicated folder keyed by <prompt_stem>.jpg
        in_img_path = row.get("image_path")

        if in_img_path and os.path.exists(in_img_path):
            final_img_path = os.path.join(images_dir, f"{prompt_stem}.jpg")
            if not os.path.exists(final_img_path):
                Image.open(in_img_path).convert("RGB").save(final_img_path, format="JPEG", quality=95, subsampling=0)
        else:
            final_img_path = None

        if args.task == "i2v-A14B" and final_img_path is None:
            raise SystemExit(
                f"--task i2v-A14B requires an input image, but `image_path` is missing for row {row_idx}. "
                f"Please ensure your manifest contains `image_path` pointing to "
                f"/home/chenyidong/train/runs/opens2v_1024/inputs/images/*.jpg"
            )

        for k in range(args.num_videos_per_prompt):
            out_path = os.path.join(videos_dir, f"{prompt_stem}-{k}.mp4")
            if os.path.exists(out_path):
                continue

            seed = args.base_seed + row_idx * 10_000 + k
            if args.task == "t2v-A14B":
                video = pipeline.generate(
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
                img = Image.open(final_img_path).convert("RGB")
                # import pdb; pdb.set_trace()
                video = pipeline.generate(
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

            save_video(
                tensor=video[None],
                save_file=out_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

        if (row_idx + 1) % 5 == 0:
            print(f"Generated videos for {row_idx + 1}/{len(rows)} prompts")

    print(f"Done. Videos at {videos_dir}")
    print(f"Images at {images_dir}")


if __name__ == "__main__":
    main()

