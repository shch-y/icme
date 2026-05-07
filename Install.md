### 拉一下submodule
git submodule update --init --recursive

### 编译flash_attn
cuda@12.8开始完整支持sm120，但是cuda@12.8不支持本集群默认的gcc@14.2，需要降级到类似gcc@12.2；我用conda在自己目录下装了个gcc@12.2
```bash
spack load cuda@12.8
export CXX=/home/shchy/miniconda3/bin/g++
export CC=/home/shchy/miniconda3/bin/gcc
```
### uv sync应该已经配置好了
uv sync -v 2>&1 | tee build.log

### 手动装一下HiFloat4
如果在登陆节点上编译，需要设置一下arch
```bash
export TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"
source .venv/bin/activate
cd HiFloat4/hi4_gpu
bash build.sh
```

### 替换一下manifest.jsonl的路径
sed 's#/home/chenyidong/train#/home/shchy/diffusion/icme#g' runs/opens2v_1024/inputs/manifest.jsonl > manifest1.jsonl 


### 应该能跑了
bash debug_smoothquant_ffn02.sh