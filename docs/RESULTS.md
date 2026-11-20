# Thor 实测记录

本页保留 Thor 原始测试。Orin 的单独输入、采样次数和用户提供的结果见 [ORIN.md](ORIN.md#http-性能)，两组耗时用于各自后端比较。

## 运行条件

2026-11-13 北京时间。JetPack 7.1、Thor SM110、CUDA 13.0、自定义 PyTorch 2.11.0、Laya 0.4.7、transformers 4.57.6。模型 revision 及文件散列见 requirements/model.json。

计时范围为本机 Python HTTP 客户端发出请求到读完响应。服务常驻并预加载。每组预热 5 次，再测 30 次。p95 为排序后第 ceil(0.95 × n) 项。共两种 state，问题取合成候选目录前 1 或 5 项。

| 输入 | state tokens | 单个 HTTP 请求中各问题合计 input tokens |
| --- | --- | --- |
| 短输入，1 问 | 7 | 48 |
| 短输入，5 问 | 7 | 225 |
| 长输入，1 问 | 469 | 510 |
| 长输入，5 问 | 469 | 2535 |

多问题的 input_tokens 是各编码行的合计，长输入 5 问包含重复的 state。完整长输入在 fixtures/recordings-long.txt，候选目录及序列化逻辑在 scripts/bench_noul.py。

## 时延

| 请求 | SDPA p50 / p95 | FA2 p50 / p95 | TensorRT p50 / p95 |
| --- | --- | --- | --- |
| 短输入，1 问 | 21.61 / 25.42 ms | 22.16 / 22.66 ms | 7.77 / 8.09 ms |
| 短输入，5 问 | 22.51 / 23.19 ms | 25.73 / 26.35 ms | 9.93 / 10.57 ms |
| 长输入，1 问 | 24.34 / 30.14 ms | 22.95 / 26.82 ms | 13.98 / 14.61 ms |
| 长输入，5 问 | 79.85 / 81.28 ms | 68.24 / 70.59 ms | 48.59 / 49.12 ms |

三个后端分别 140 条 HTTP 请求成功，包含预热。所有输入都未截断。SDPA 单独启动的早期测试中，预加载后的第一条请求约 514 ms，后续进入热态。本表来自后面的三路径对照，模型加载、engine 构建及首轮 JIT 时间分别处理。

FA2 使用 vLLM 内置扩展，22 层编码器采用 FA2，决策头继续 PyTorch。SDPA 原始验证观察到 24 次 efficient attention 调用。TensorRT 用本机生成的混合 FP16 engine，构建约 51 秒、体积约 619 MiB。服务进程共用同一 GPU，已有生成模型常驻，记录中的测试前后生成服务健康检查通过。

## 概率比较

FA2 相对 SDPA 最大概率差为 0.0014，TensorRT 为 0.0304。最大 TRT 差来自长输入 5 问中的一项，概率从 0.6357 降到 0.6053。

每个候选在热态重复 30 次。两个 state 各有 1 问和 5 问，总共 12 个不同的 payload/question 组合，360 次重复概率比较。0.60 阈值下这些样本的筛选结果一致。数据用于数值与接口对照，未提供任务正确答案标签，因此准确率、概率校准质量以及真实 Octop 成功率仍需对应现场任务验证。

## 重算

```bash
python scripts/summarize.py \
  evidence/thor-2026-11-13/sdpa-short-ab.jsonl \
  evidence/thor-2026-11-13/fa2-short-ab.jsonl \
  evidence/thor-2026-11-13/trt-short-ab.jsonl
python scripts/summarize.py \
  evidence/thor-2026-11-13/sdpa-long-ab.jsonl \
  evidence/thor-2026-11-13/fa2-long-ab.jsonl \
  evidence/thor-2026-11-13/trt-long-ab.jsonl
```

## 原始代码与仓库版

evidence/thor-2026-11-13/source/ 保存测量时的三个脚本。GPU 证明中的本地模型目录已替换成占位符，其余提交内容为合成输入对应的记录。

当前 scripts/serve.py 将模型路径、FA2 来源和监听端口做成参数，增加 GPU 证明和诊断路由。TRT runtime 采用专用 stream 和完成同步，修正原型里依赖默认 stream 的缓冲区生命周期。scripts/create_env.sh 对 Conda 管理的 setuptools 设置小于 82 的版本范围，以匹配原始自定义 Torch wheel。

历史表格保留原运行结果。仓库版脚本的重新验证记录见 evidence/repository-validation.json，现场新机器的性能由实际复测产生。

2026-11-20 的仓库版验证通过了新建 PyTorch 环境、三个后端实际执行、84 次短长 HTTP 请求、鉴权、ONNX 导出、TRT 本机构建、三个 profile 角点及超出范围后恢复请求。FA2 最大概率差仍为 0.0014，新构建 TRT 为 0.0306，12 个不同判断在 0.60 阈值下没有筛选翻转。原始 JSONL 和执行证明保存在 evidence/thor-2026-11-20/。SDPA 和 FA2 自测与 TRT 构建有时间重叠，这批耗时用于运行验证。
