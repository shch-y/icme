#!/usr/bin/env bash
# Same layout as debug.sh: one-sample generation + VBench eval, with SmoothQuant-style
# smoothing on ffn.0 (see opens2v_vbench_wan22_i2v/generate_wan22_official.py --ffn_smoothquant).

set -ex

type=fp4ffn0sq

if [ -f "/home/chenyidong/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/home/chenyidong/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2
  exit 1
fi
conda activate icme

rm -rf "runs/opens2v_1024/generated${type:?}"/*
srun -N 1 --gres=gpu:H100:1 -p Long --pty /home/chenyidong/anaconda3/envs/icme/bin/python \
  opens2v_vbench_wan22_i2v/generate_wan22_official.py \
  --manifest runs/opens2v_1024/inputs/manifest.jsonl \
  --out_dir "runs/opens2v_1024/generated${type}" \
  --ckpt_dir /home/dataset/Wan2.2-I2V-A14B \
  --wan_repo /home/chenyidong/train/Wan2.2 \
  --task i2v-A14B \
  --max_prompts 1 \
  --num_videos_per_prompt 1 \
  --height 480 --width 832 \
  --frame_num 81 \
  --base_seed 1234 \
  --use_hifx4 \
  --use_precision_aware_ffn0 \
  --ffn_smoothquant \
  --ffn_smoothquant_alpha 0.85

rm -rf "runs/opens2v_1024/vbench_out${type}"/*
python VBench/evaluate_i2v.py \
  --videos_path "runs/opens2v_1024/generated${type}/videos" \
  --custom_image_folder "runs/opens2v_1024/generated${type}/images" \
  --mode custom_input \
  --ratio 16-9 \
  --dimension i2v_subject \
  --output_path "runs/opens2v_1024/vbench_out${type}"
python opens2v_vbench_wan22_i2v/summarize_vbench.py \
  --eval_json "runs/opens2v_1024/vbench_out${type}/*_eval_results.json"
