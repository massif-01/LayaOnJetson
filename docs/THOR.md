# Thor 安装和运行

## 已验证环境

现场为 JetPack 7.1 / L4T R38.4、Linux ARM64、Thor SM110、CUDA 13.0、Python 3.12。使用本机编译的 PyTorch 2.11.0，Laya 0.4.7，transformers 4.57.6。精度采用 FP16 autocast，PyTorch 保存的主权重为 FP32。SDPA 实测选中了 efficient attention。

源码和环境记录见 requirements/thor-validated.json。原始 vLLM FA2 构建针对 sm80，包含 PTX，Thor 成功通过 PTX JIT 执行。设备上存在扩展文件只是定位材料，实际调用结果是验收依据。[CUDA 的 PTX 兼容说明](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)

requirements/thor-revalidated-pip.txt 保存仓库脚本重新安装后的完整 Python 包版本，用于恢复时核对依赖。该次新环境的 Python 为 3.12.16，历史环境为 3.12.15。Torch 仍需先安装目标机器匹配的 wheel，包版本清单没有包含 CUDA、系统 TensorRT 或 FA2 二进制。

## 环境准备

```bash
uname -m
head -n 1 /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack nvidia-l4t-core
nvidia-smi --query-gpu=name,compute_cap,driver_version,utilization.gpu --format=csv
conda env list
```

按目标 Python、CUDA、C++ ABI 选择 PyTorch wheel，保留 wheel 散列作为本地部署记录。原始 Thor wheel 为自定义构建，散列在 requirements/thor-validated.json。PyTorch 官方还提供 CUDA 13.0 的 ARM64 wheel，可作为重新部署候选，运行 GPU 证明后记录其差异。[PyTorch wheel 源](https://download.pytorch.org/whl/cu130/torch/) · [JetPack 7.1 组件](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.1)

```bash
bash scripts/create_env.sh laya-jetson-fp16 3.12 /absolute/path/to/torch.whl sdpa
conda activate laya-jetson-fp16
mkdir -p results
python scripts/check_gpu.py > results/environment.json
python scripts/download_model.py models/multilingual
python scripts/serve.py --backend sdpa --model-dir models/multilingual --proof-only
python scripts/serve.py --backend sdpa --model-dir models/multilingual --port 18047
```

启动前会验证模型六个文件的 SHA-256，显式配置 Laya 的 FP16、CUDA、预加载和 multilingual。服务继承上游的 HTTP 请求处理、tokenization、Noul 概率和响应格式。

## 复用已有 vLLM FA2

先定位已安装 vLLM 的目录，找到同目录的 flash_attn_interface.py 和 _vllm_fa2_C.abi3.so。将这两个文件复制到新环境的专用 vendor 目录。下面的占位路径替换为本机实际路径。

```bash
mkdir -p vendor/vllm_fa2
cp /absolute/path/to/vllm/vllm_flash_attn/flash_attn_interface.py vendor/vllm_fa2/
cp /absolute/path/to/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so vendor/vllm_fa2/
conda activate laya-jetson-fp16
python scripts/serve.py --backend fa2 --model-dir models/multilingual \
  --fa2-source vllm --vllm-fa2-dir vendor/vllm_fa2 --proof-only
python scripts/serve.py --backend fa2 --model-dir models/multilingual \
  --fa2-source vllm --vllm-fa2-dir vendor/vllm_fa2 --port 18048
```

适配器以隔离的 Python 模块名加载这两个文件，Python 依赖来自当前 Laya 环境。成功 warm-up 会统计编码器 8 层全局和 14 层滑窗调用，并报告 CUDA 和 FP16。两个决策头层继续走 PyTorch。

## 独立 flash_attn 构建

脚本支持 `--fa2-source standalone`。在目标环境安装匹配的 `flash_attn` 后可使用下列入口。这个入口留给独立 FA2 构建复现，原始 Thor 实测采用上面的 vLLM 扩展。

```bash
python scripts/serve.py --backend fa2 --model-dir models/multilingual \
  --fa2-source standalone --proof-only
python scripts/serve.py --backend fa2 --model-dir models/multilingual \
  --fa2-source standalone --port 18048
```

安装 wheel 时核对 Python 标签、Torch 版本、CUDA 版本、架构和 C++ ABI。构建参考 FlashAttention 对应版本的 setup.py，最终以本机全局、滑窗及完整 Laya 请求验证为准。[FlashAttention](https://github.com/Dao-AILab/flash-attention)

## HTTP 验收

在另一个终端激活相同环境并进入仓库根目录。

```bash
curl -fsS http://127.0.0.1:18048/health
curl -fsS http://127.0.0.1:18048/jetson/status
mkdir -p results
python scripts/bench_noul.py http://127.0.0.1:18048 results/fa2-short.jsonl \
  --device cuda --questions 1 5 --warmup 5 --iterations 30
python scripts/bench_noul.py http://127.0.0.1:18048 results/fa2-long.jsonl \
  --device cuda --questions 1 5 --warmup 5 --iterations 30 \
  --state-file fixtures/recordings-long.txt
```

SDPA 用 18047 端口和不同输出文件名测同一批请求。比较命令见 docs/RESULTS.md。每次生成全新结果文件，防止覆盖既有记录。
