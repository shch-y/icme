## OpenS2V-5M + VBench 评测 Wan2.2 I2V

本目录提供一个**可直接跑通**的流水线：

- **准备 OpenS2V-5M I2V 输入**：从视频抽首帧作为条件图，使用 caption 作为 prompt
- **用 `Wan-AI/Wan2.2-I2V-A14B` 生成视频**
- **用 VBench-I2V 指标评测**（`i2v_subject` / `i2v_background` / `camera_motion` + 质量维度）

### 0) 环境

你已经给出运行方式：

```bash
conda activate icme
srun -N 1 --pty --gres=gpu:4090:1 -p Long
```

建议从仓库根目录运行（`/home/chenyidong/train`）。

### 1) 准备输入（从 OpenS2V-5M 抽样 + 抽首帧）

你的本地数据在：

- 视频：`/home/chenyidong/train/HiFloat4/datasets/total_part2`
- prompt/index：`/home/chenyidong/train/HiFloat4/datasets/OpenS2V-5M_to_mm.json`

示例：抽样 1024 条（会自动从 index 里读取 caption 作为 prompt，并把 `/home/datasets/OpenS2V-5M/...` 重映射到你本地 `HiFloat4/datasets/...`），输出到 `runs/opens2v_1024/inputs/`

```bash
python opens2v_vbench_wan22_i2v/prepare_inputs.py \
  --output_dir runs/opens2v_1024/inputs \
  --max_samples 1024 \
  --seed 42 \
  --prefer_local_index /home/chenyidong/train/HiFloat4/datasets/OpenS2V-5M_to_mm.json \
  --path_prefix_from /home/datasets/OpenS2V-5M \
  --path_prefix_to /home/chenyidong/train/HiFloat4/datasets/total_part2
```

会生成：

- `runs/opens2v_1024/inputs/images/*.jpg`：条件图（首帧）
- `runs/opens2v_1024/inputs/manifest.jsonl`：每条样本的 `prompt`、`image_path`、源视频路径等

### 2) 生成视频（Wan2.2 I2V）

```bash
# 这里会直接使用你本地的 OpenS2V 子集（`/home/chenyidong/train/HiFloat4/datasets/total_part2`）：
# `manifest.jsonl` 里的 `image_path/source_video` 都来自该目录抽帧得到
srun -N 1 --pty --gres=gpu:H100:1 -p Long python  opens2v_vbench_wan22_i2v/generate_wan22_i2v.py \
  --manifest runs/opens2v_1024/inputs/manifest.jsonl \
  --out_dir runs/opens2v_1024/generated \
  --model /home/dataset/Wan2.2-I2V-A14B \
  --num_videos_per_prompt 5 \
  --fps 16 \
  --num_frames 81 \
  --height 480 \
  --width 832 \
  --base_seed 1234

# 如果要使用 HiFloat4 的 hifx4 量化 Linear 做推理（参考 replace_linear(model, "hifx4", in_Q="hifx4", quant_grad=False)）
# 把生成脚本换成官方权重推理封装，并加上 `--use_hifx4`：
#


srun -N 1 --pty --gres=gpu:H100:1 -p Long /home/chenyidong/anaconda3/envs/icme/bin/python opens2v_vbench_wan22_i2v/generate_wan22_official.py   --manifest runs/opens2v_1024/inputs/manifest.jsonl   --out_dir runs/opens2v_1024/generatedfp4   --ckpt_dir /home/dataset/Wan2.2-I2V-A14B   --wan_repo /home/chenyidong/train/Wan2.2   --task i2v-A14B   --num_videos_per_prompt  5   --height 480 --width 832   --frame_num 81   --base_seed 1234    --use_hifx4

srun -N 1 --pty --gres=gpu:H100:1 -p Long /home/chenyidong/anaconda3/envs/icme/bin/python opens2v_vbench_wan22_i2v/generate_wan22_official.py   --manifest runs/opens2v_1024/inputs/manifest.jsonl   --out_dir runs/opens2v_1024/generated   --ckpt_dir /home/dataset/Wan2.2-I2V-A14B   --wan_repo /home/chenyidong/train/Wan2.2   --task i2v-A14B   --num_videos_per_prompt  5   --height 480 --width 832   --frame_num 81   --base_seed 1234   

srun -N 1 --pty --gres=gpu:H100:1 -p Long /home/chenyidong/anaconda3/envs/icme/bin/python opens2v_vbench_wan22_i2v/generate_wan22_official.py   --manifest xpq/manifest.jsonl   --out_dir xpq/generated   --ckpt_dir /home/dataset/Wan2.2-I2V-A14B   --wan_repo /home/chenyidong/train/Wan2.2   --task i2v-A14B   --num_videos_per_prompt  5   --height 480 --width 832   --frame_num 81   --base_seed 1234   

```

输出结构（满足 VBench custom_input 规则）：

- `runs/opens2v_1024/generated/videos/<prompt_stem>-0.mp4 ... -4.mp4`
- `runs/opens2v_1024/generated/images/<prompt_stem>.jpg`

### 3) 跑 VBench-I2V 评测

```bash

type=fp4

rm -rf  runs/opens2v_1024/vbench_out${type}/*
python VBench/evaluate_i2v.py \
  --videos_path runs/opens2v_1024/generated${type}/videos \
  --custom_image_folder runs/opens2v_1024/generated/images \
  --mode custom_input \
  --ratio 16-9 \
  --dimension i2v_subject   \
  --output_path runs/opens2v_1024/vbench_out${type}
python opens2v_vbench_wan22_i2v/summarize_vbench.py \
  --eval_json "runs/opens2v_1024/vbench_out${type}/*_eval_results.json"


 python VBench/evaluate_i2v.py \
  --videos_path runs/opens2v_1024/generated/videos \
  --custom_image_folder runs/opens2v_1024/generated/images \
  --mode custom_input \
  --ratio 16-9 \
  --dimension i2v_subject   \
  --output_path runs/opens2v_1024/vbench_out
```

如果你也要跑质量维度（更慢），把 `--dimension` 换成：

```bash
--dimension subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality i2v_subject i2v_background camera_motion
```

### 4) 汇总结果

```bash
python opens2v_vbench_wan22_i2v/summarize_vbench.py \
  --eval_json "runs/opens2v_1024/vbench_out/*_eval_results.json"

python opens2v_vbench_wan22_i2v/summarize_vbench.py \
  --eval_json "runs/opens2v_1024/vbench_outfp4/*_eval_results.json"


```

