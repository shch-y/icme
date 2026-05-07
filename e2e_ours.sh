#!/usr/bin/env bash
# Like debug_smoothquant.sh, but wraps both ffn[0] and ffn[2] with PrecisionAwareFFNLinear + SmoothQuant.
# Conda env: icme (same as other debug scripts).

set -ex

type=fp4ffn02sq


rm -rf "runs/opens2v_1024/generated${type:?}"/*
srun -N 1 --gres=gpu:H100:1 -c 16 --pty python \
  opens2v_vbench_wan22_i2v/generate_wan22_official.py \
  --manifest runs/opens2v_1024/inputs/manifest1.jsonl \
  --out_dir "runs/opens2v_1024/generated${type}" \
  --ckpt_dir /home/dataset/Wan2.2-I2V-A14B \
  --wan_repo /home/shchy/diffusion/icme/Wan2.2 \
  --task i2v-A14B \
  --max_prompts 1 \
  --num_videos_per_prompt 1 \
  --height 480 --width 832 \
  --frame_num 77 \
  --base_seed 1234 \
  --use_hifx4 \
  --use_precision_aware_ffn0 \
  --use_precision_aware_ffn2 \
  --ffn_smoothquant \
  --ffn_smoothquant_alpha 0.85

rm -rf "runs/opens2v_1024/vbench_out${type}"/*
python VBench/evaluate_i2v.py \
  --videos_path "runs/opens2v_1024/generated${type}/videos" \
  --custom_image_folder "runs/opens2v_1024/generated${type}/images" \
  --mode custom_input \
  --ratio 16-9 \
  --dimension imaging_quality aesthetic_quality subject_consistency motion_smoothness  \
  --output_path "runs/opens2v_1024/vbench_out${type}"
python opens2v_vbench_wan22_i2v/summarize_vbench.py \
  --eval_json "runs/opens2v_1024/vbench_out${type}/*_eval_results.json"
