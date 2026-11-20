# LayaOnJetson

在 NVIDIA Jetson 上通过独立 Conda 环境运行 Laya，并把决策服务接到 Octop。

仓库包含 Jetson Thor 的源码与逐次测量记录，以及用户提供的 Jetson AGX Orin 部署回执。Orin 回执已脱敏，涵盖 FP16 + FA2 和 Octop 真实请求验收；按用户要求不收录同事电脑上的文件和链接。AI 接手时从 [AGENTS.md](AGENTS.md) 开始，Claude Code 也可从 [CLAUDE.md](CLAUDE.md) 进入。

| 路径 | 内容 | 当前证据 |
| --- | --- | --- |
| PyTorch FP16 + SDPA | Laya 原生推理，独立 Conda 环境 | Thor 实测、Orin 现场回执 |
| PyTorch FP16 + FA2 | ModernBERT 全局及滑窗 attention 适配，保留 Laya 决策头 | Thor 实测、Orin 现场回执，均复用各自设备的 vLLM FA2 扩展 |
| TensorRT FP16 | 同一权重导出 ONNX，在目标设备构建 engine，独立 Conda 环境 | Thor 实测，当前 profile 为 1–5 问 |
| Octop 接入 | 配置 endpoint、模型、鉴权、网络和真实请求验收 | Orin 回执报告已接通；Thor 原始测试为独立服务 |

## 从哪里开始

- [Thor 完整安装和启动](docs/THOR.md)
- [Orin 环境、重建步骤和现场回执](docs/ORIN.md)
- [TensorRT 导出、构建和启动](docs/TENSORRT.md)
- [Octop 接通与回滚](docs/OCTOP.md)
- [实测数据和验证边界](docs/RESULTS.md)
- [常见问题和备份恢复](docs/OPERATIONS.md)
- [交给 Codex / Claude Code 的指令](docs/AI-PROMPT.md)
- [依赖来源和版权](NOTICE.md)

## 最短启动路径

在 Thor 上进入本仓库。先准备与本机 JetPack、Python、CUDA 匹配的 ARM64 PyTorch wheel。原始实测 wheel 的版本和散列在 [环境记录](requirements/thor-validated.json)。使用其他构建时，通过实际 CUDA 运算和下列自测确认兼容性。

```bash
bash scripts/create_env.sh laya-jetson-fp16 3.12 /absolute/path/to/torch.whl sdpa
conda activate laya-jetson-fp16
python scripts/check_gpu.py
python scripts/download_model.py models/multilingual
python scripts/serve.py --backend sdpa --model-dir models/multilingual --port 18047
```

独立终端中验证请求。

```bash
conda activate laya-jetson-fp16
mkdir -p results
python scripts/bench_noul.py http://127.0.0.1:18047 results/sdpa-short.jsonl \
  --device cuda --questions 1 5 --warmup 5 --iterations 30
python scripts/bench_noul.py http://127.0.0.1:18047 results/sdpa-long.jsonl \
  --device cuda --questions 1 5 --warmup 5 --iterations 30 \
  --state-file fixtures/recordings-long.txt
```

## 已测结果

2026-11-13 北京时间，Thor 的本机 HTTP 往返时延。每格为 p50 / p95，每组预热 5 次、正式测量 30 次。

| 合成请求 | PyTorch SDPA | vLLM FA2 | TensorRT |
| --- | --- | --- | --- |
| 短输入，1 问 | 21.61 / 25.42 ms | 22.16 / 22.66 ms | 7.77 / 8.09 ms |
| 短输入，5 问 | 22.51 / 23.19 ms | 25.73 / 26.35 ms | 9.93 / 10.57 ms |
| 长输入，1 问 | 24.34 / 30.14 ms | 22.95 / 26.82 ms | 13.98 / 14.61 ms |
| 长输入，5 问 | 79.85 / 81.28 ms | 68.24 / 70.59 ms | 48.59 / 49.12 ms |

FA2 相对 SDPA 的最大概率偏差为 0.0014，TensorRT 为 0.0304。12 个不同的合成判断在 0.60 阈值下筛选一致。完整记录与解释见 [RESULTS.md](docs/RESULTS.md)。

Orin 回执中，长输入 5–20 问使用 FA2 的 p50 比 SDPA 下降约 13%–14%，短输入及长输入单问由 SDPA 更快。最大概率差为 0.0081，在 0.60 阈值下无筛选翻转。另一个编码器对照的最低余弦相似度为 0.996285，未达到该项额外检查的 0.999 阈值，详见 [Orin 数值检查](docs/ORIN.md#数值检查)。两台设备的输入与采样不同，耗时表各自记录。

模型、Conda 归档、CUDA 库、FA2 扩展和 TensorRT engine 由部署机器获取或构建。仓库提供固定模型 revision、SHA-256 和源码入口，运行时文件写入 Git 忽略的目录。
