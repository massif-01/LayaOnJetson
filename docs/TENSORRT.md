# TensorRT FP16

## 范围

原始 Thor 测试采用 TensorRT 10.13.3.9。模型导出为 ONNX opset 18，动态 batch 和序列长度，marker 数固定为 2。当前构建 profile 为 batch 1–5、序列长度 8–1024。Octop 的 Noul 每问有两个选项，完整场景可能超过 5 问，接入前需用实际 payload 验证 profile。

本仓库沿用 Laya 的模型和概率解码，在 forward 入口调用 TensorRT。运行时代码为 engine 创建专用 CUDA stream，等待输出完成后再交给 Laya 解码，并拒绝非有限输出。上游 HTTP 服务使用单工作线程和 gate 调度请求。

## 独立 Conda 环境

TensorRT 创建单独环境。系统的 CUDA、TensorRT 运行库来自该设备 JetPack。系统 Python 绑定与 Conda 的 Python 主次版本应相同，Thor 7.1 对应 Python 3.12。

```bash
bash scripts/create_env.sh laya-jetson-trt 3.12 /absolute/path/to/torch.whl trt
conda activate laya-jetson-trt
python scripts/check_gpu.py
trt_binding="$(dpkg -L python3-libnvinfer | sed -n 's@/tensorrt/__init__.py$@@p' | head -n 1)"
test -n "$trt_binding"
export PYTHONPATH="$trt_binding"
python -c 'import tensorrt; print(tensorrt.__version__)'
```

也可以把与当前 JetPack 匹配的 TensorRT Python wheel 安装进专用环境。使用该安装方式时根据实际路径设置运行库搜索路径。

## 导出、构建、启动

在仓库根目录，激活 TRT 环境后执行。

```bash
python scripts/download_model.py models/multilingual
python scripts/export_trt.py --model-dir models/multilingual --output artifacts/laya-noul.onnx
bash scripts/build_trt.sh artifacts/laya-noul.onnx artifacts/laya-noul.engine
python scripts/serve.py --backend trt --model-dir models/multilingual \
  --engine artifacts/laya-noul.engine --proof-only
python scripts/serve.py --backend trt --model-dir models/multilingual \
  --engine artifacts/laya-noul.engine --port 18049
```

构建脚本支持通过 `TRTEXEC` 指定 CLI 路径。现有命令采用 TensorRT 10.x 的 `--fp16` 混合精度策略，各层精度由 engine 编译器决定。TensorRT 11.x 更改为显式强类型图，先按相应迁移指南转换模型。[TensorRT CLI 迁移说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html)

engine 在目标 GPU 和 TensorRT 版本上构建。模型权重、Python 源码和 ONNX 可以用作下一台机器的复现输入，engine 重新生成。[TensorRT 兼容规则](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/version-compatibility.html#hardware-compatibility)

## 测量与概率回归

```bash
mkdir -p results
python scripts/bench_noul.py http://127.0.0.1:18049 results/trt-short.jsonl \
  --device cuda --questions 1 5 --warmup 5 --iterations 30
python scripts/bench_noul.py http://127.0.0.1:18049 results/trt-long.jsonl \
  --device cuda --questions 1 5 --warmup 5 --iterations 30 \
  --state-file fixtures/recordings-long.txt
python scripts/summarize.py results/sdpa-short.jsonl results/trt-short.jsonl
python scripts/summarize.py results/sdpa-long.jsonl results/trt-long.jsonl
```

原始 TensorRT 对照最大的概率差为 0.0304，一条 Noul 从 0.6357 变为 0.6053。该样本在 0.60 阈值下筛选一致，靠近阈值的真实任务仍需逐项验证。扩大 profile 或改变 TensorRT 精度策略后，重新测同一批请求。
