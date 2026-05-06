type=fp4

rm -rf  runs/opens2v_1024/vbench_out${type}/*
python VBench/evaluate_i2v.py \
  --videos_path runs/opens2v_1024/generated${type}/videos \
  --custom_image_folder runs/opens2v_1024/generated${type}/images \
  --mode custom_input \
  --ratio 16-9 \
  --dimension i2v_subject   \
  --output_path runs/opens2v_1024/vbench_out${type}
python opens2v_vbench_wan22_i2v/summarize_vbench.py \
  --eval_json "runs/opens2v_1024/vbench_out${type}/*_eval_results.json"


type=fp4ffn0

rm -rf  runs/opens2v_1024/vbench_out${type}/*
python VBench/evaluate_i2v.py \
  --videos_path runs/opens2v_1024/generated${type}/videos \
  --custom_image_folder runs/opens2v_1024/generated${type}/images \
  --mode custom_input \
  --ratio 16-9 \
  --dimension i2v_subject   \
  --output_path runs/opens2v_1024/vbench_out${type}
python opens2v_vbench_wan22_i2v/summarize_vbench.py \
  --eval_json "runs/opens2v_1024/vbench_out${type}/*_eval_results.json"


type=fp4ffn0tmp

rm -rf  runs/opens2v_1024/vbench_out${type}/*
python VBench/evaluate_i2v.py \
  --videos_path runs/opens2v_1024/generated${type}/videos \
  --custom_image_folder runs/opens2v_1024/generated${type}/images \
  --mode custom_input \
  --ratio 16-9 \
  --dimension i2v_subject   \
  --output_path runs/opens2v_1024/vbench_out${type}
python opens2v_vbench_wan22_i2v/summarize_vbench.py \
  --eval_json "runs/opens2v_1024/vbench_out${type}/*_eval_results.json"