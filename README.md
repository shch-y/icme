1. uv sync应该已经配置好了
uv sync -v 2>&1 | tee build.log

2. 手动装一下HiFloat4
source .venv/bin/activate
cd HiFloat4/hi4_gpu
bash build.sh

3. 应该能跑了
bash debug_smoothquant_ffn02.sh