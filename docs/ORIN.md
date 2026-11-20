# Jetson AGX Orin 部署记录与重建

本页依据用户于 2026-11-20 提供的现场部署回执整理。原部署和测试日期未单独给出。按用户要求，仅收录脱敏后的版本、结果与通用命令，不收录同事电脑上的文件和链接。本轮整理没有连接设备或重新运行推理。

## 环境

| 项目 | 现场回执 |
| --- | --- |
| GPU | Jetson AGX Orin 64 GB，SM87 |
| 系统 | L4T 36.4.7，JetPack 6 系列，未提供精确 JetPack 小版本 |
| Conda | laya-orin-fp16，Python 3.12.12 |
| PyTorch | 2.9.0，CUDA 12.6 构建，cuDNN 9.17，C++11 ABI |
| Laya / Transformers | 0.4.7 / 4.57.6 |
| FA2 来源 | 本机 vLLM 0.11.1rc6.dev52+g6c317a656.d20251103.cu126 的 _vllm_fa2_C.abi3.so |
| 模型 | multilingual，revision 1720e3e3357cfe1e281542e223f8273b0890ca34 |
| Octop | 1.0.3b1，另一台宿主机上的容器 |
| HTTP | 端口 58190，/v1/systemone，模型 multilingual，Bearer 鉴权 |
| 服务 | 用户级 laya-orin.service，回执报告已配置开机自启，受控服务重启后复测通过 |

模型来自已有缓存，回执报告权重 SHA-256 与指定 revision 一致。新机器按 [固定模型清单](../requirements/model.json) 核对全部六个文件。机器可用内存、Torch wheel 散列、FA2 二进制散列和完整包清单未在回执正文中提供，仓库未推定这些值。

## 重建环境

以下命令是基于仓库入口整理的重建步骤。原现场回执没有提供完整安装命令和适配脚本，本仓库版本尚未在该 Orin 独立复测。执行者在目标机完成 GPU、接口、数值及 Octop 验收后，记录本机结果。

先核对当前设备和服务，保持既有服务正常运行。

```bash
uname -m
head -n 1 /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack nvidia-l4t-core
nvidia-smi
conda env list
```

在新机器准备适配 Orin SM87、Python 3.12、CUDA 12.6 和 C++11 ABI 的 Torch 2.9.0 wheel。用新环境运行 SDPA 基线。

```bash
bash scripts/create_env.sh laya-orin-fp16 3.12.12 /absolute/path/to/orin-torch.whl sdpa
conda activate laya-orin-fp16
mkdir -p results
python -m pip check
python scripts/check_gpu.py > results/orin-environment.json
python scripts/download_model.py models/multilingual
python scripts/serve.py --backend sdpa --model-dir models/multilingual --proof-only
```

核对实际输出中的 Torch 版本、CUDA、SM87 和 C++ ABI。已有模型可以先复制到 models/multilingual，再用 download_model.py 的 --verify-only 核对。Thor 的 CUDA 13 / SM110 wheel 和 engine 不能作为该机安装输入。

现场回执使用过 Conda 克隆。如果目标机已有兼容的 GPU 环境，可以先在本机选定来源，再用 `conda create -n laya-orin-fp16 --clone SOURCE_ENV` 创建专用环境，在 Torch 版本约束下安装 requirements/common.txt。该路线仍需核对依赖和实际 GPU 运算。原克隆事务曾报告 libstdc++ 缓存文件大小校验差异，随后完成，pip check、CUDA 运算和实际推理通过；新机器若出现校验错误，先定位受影响的缓存包并重新获取，再完成环境事务和验收。

## 启用 FlashAttention 2

复用目标 Orin 自身的 vLLM 包装层和扩展。将本机相同发行构建中的文件放入 Git 忽略的 vendor/。下面是占位路径。

```bash
mkdir -p vendor/vllm_fa2
cp /absolute/path/to/orin/vllm_flash_attn/flash_attn_interface.py vendor/vllm_fa2/
cp /absolute/path/to/orin/vllm_flash_attn/_vllm_fa2_C.abi3.so vendor/vllm_fa2/
sha256sum vendor/vllm_fa2/flash_attn_interface.py vendor/vllm_fa2/_vllm_fa2_C.abi3.so
python scripts/serve.py --backend fa2 --model-dir models/multilingual \
  --fa2-source vllm --vllm-fa2-dir vendor/vllm_fa2 --proof-only
```

Thor 和 Orin 回执使用的 vLLM 版本不同，先核对本机包装层的扩展导入、可用性标志和 flash_attn_varlen_func 签名与 scripts/fa2_backend.py 的接口。本仓库 loader 的直接实测来自 Thor；不能把上述命令写成 Orin 仓库版已验收。遇到接口差异时对照目标包装层调整 loader，保留真正的 FA2 调用和数值验证，避免静默转回 SDPA。

现场调用记录报告每轮 8 次双向全局 FA2、14 次双向滑窗 FA2，Q/K/V 为 cuda:0 上的 FP16，head dimension 为 64。仓库版运行时应检查相同的编码器调用计数、GPU 状态和 CPU 回退计数。

先在回环地址前台启动并验证。

```bash
python scripts/serve.py --backend fa2 --model-dir models/multilingual \
  --fa2-source vllm --vllm-fa2-dir vendor/vllm_fa2 --port 58190
```

在另一个终端发出短长合成请求。以下样本来自本仓库，与回执中的 21 / 754 token 输入不同。

```bash
conda activate laya-orin-fp16
mkdir -p results
python scripts/bench_noul.py http://127.0.0.1:58190 results/orin-fa2-short.jsonl \
  --device cuda --questions 1 5 10 20 --warmup 5 --iterations 20
python scripts/bench_noul.py http://127.0.0.1:58190 results/orin-fa2-long.jsonl \
  --device cuda --questions 1 5 10 20 --warmup 5 --iterations 20 \
  --state-file fixtures/recordings-long.txt
```

SDPA 用不同端口和输出文件，对同一批输入按相同参数测量，再用 scripts/summarize.py 比较。两种后端顺序运行，记录当前负载、功耗模式和时钟设置；这些系统条件未在原回执正文中提供。

## HTTP 性能

以下数值来自用户提供的回执。单并发，每组预热 5 次、正式测 20 次，单位 ms。

| 正文 tokens | Noul 数 | SDPA p50 / p95 | FA2 p50 / p95 | 最大概率差 |
| --- | --- | --- | --- | --- |
| 21 | 1 | 62.94 / 65.30 | 74.40 / 75.08 | 0.0000 |
| 21 | 5 | 65.95 / 67.35 | 77.51 / 78.56 | 0.0005 |
| 21 | 10 | 69.34 / 70.63 | 81.19 / 82.59 | 0.0009 |
| 21 | 20 | 75.12 / 76.48 | 87.01 / 88.25 | 0.0008 |
| 754 | 1 | 68.88 / 70.25 | 76.30 / 77.27 | 0.0000 |
| 754 | 5 | 169.11 / 170.48 | 147.21 / 148.60 | 0.0075 |
| 754 | 10 | 299.62 / 301.27 | 258.64 / 260.13 | 0.0076 |
| 754 | 20 | 567.80 / 568.32 | 488.59 / 489.49 | 0.0081 |

每个后端共 160 次正式请求、0 失败、0 截断。长输入 5–20 问时 FA2 的 p50 下降约 13%–14%；短输入及长输入单问由 SDPA 更快。两套后端结束时 /health 报告模型驻留 CUDA，CPU 回退为 0。

CUDA 峰值分配量为 SDPA 1,873 MiB、FA2 1,817 MiB；峰值保留量分别为 2,342 MiB、2,178 MiB。这些是 CUDA allocator 统计，未提供整机或进程总内存峰值。

Orin 和 Thor 测试的正文、问题数量、软件版本及采样次数不同。当前表格支持同一 Orin 测试内比较后端，不支持直接算两块板的性能倍率。

## 数值检查

变长序列的全局和滑窗 FA2 分别与 FP32 显式掩码参考对照，回执报告两项最大绝对误差均为 0.000537。该回执没有给出原始张量或完整输入。

整个编码器跨 FP16 后端比较时，最低余弦相似度为 0.996285，未达到该项额外检查设定的 0.999。这项检查没有通过，回执未提供根因定位结果。局部 attention 的小误差与端到端 Noul 在给定阈值下未翻转，不能替代编码器这项检查的结论。

1,440 对重复概率比较中，平均绝对差 0.00058，最大绝对差 0.0081。按 Octop 使用的 0.60 阈值，0 次跨线。这里没有带任务正确答案的准确率测试，重复概率比较次数也不等于独立任务数。

## Octop 接入结果

回执报告，服务监听目标 Orin 的接口，Octop 保存在另一台宿主机上的 endpoint 为 `http://ORIN_HOST:58190/v1/systemone`，model 为 multilingual，启用 Bearer 鉴权。ORIN_HOST 是脱敏占位符，实际配置时替换为从 Octop 网络空间可达的地址。

Laya 受控重启后，Octop 用已保存配置执行决策测试，返回 ok=true，概率 0.9139。随后新会话触发评分，记录包含 20 个 Noul，相关 HTTP 返回 200。回执未单独提供 HTTP 调用计数。Octop 发出决策事件，主 Agent 成功调用 current_time、回复工具返回的上海时间并结束该轮。

新机器按 [Octop 指南](OCTOP.md) 重新核对接口、保存回滚材料并验收。跨主机启动前在本机读取 LAYA_API_KEY，并显式指定目标接口。

```bash
read -r -s -p 'Laya key: ' LAYA_API_KEY
printf '\n'
export LAYA_API_KEY
export ORIN_BIND_ADDRESS='REPLACE_WITH_ORIN_INTERFACE_ADDRESS'
python scripts/serve.py --backend fa2 --model-dir models/multilingual \
  --fa2-source vllm --vllm-fa2-dir vendor/vllm_fa2 \
  --host "$ORIN_BIND_ADDRESS" --port 58190
```

密钥通过部署机器的本地配置管理，凭据与 Octop 配置备份保存在仓库外。

## 服务启停与恢复

以下启停命令来自现场回执，在服务所属用户会话中执行。

```bash
systemctl --user start laya-orin.service
systemctl --user stop laya-orin.service
systemctl --user restart laya-orin.service
systemctl --user status laya-orin.service
journalctl --user -u laya-orin.service -n 50
systemctl --user is-enabled laya-orin.service
```

原回执报告已配置开机自启，验证的是服务受控重启，未提供整机重启验收。新装单元参照 [维护指南](OPERATIONS.md)，替换环境、仓库、模型、FA2 路径、监听接口和端口，按现场需求配置用户级 systemd 与 linger。

恢复 Octop 时使用部署前保留的本机配置与凭据，按 [Octop 回滚流程](OCTOP.md) 回写并回读验证。原现场的本地恢复脚本没有收入仓库，新机器应建立自己的回滚材料。

结构化环境见 [orin-reported.json](../requirements/orin-reported.json)，结构化结果见 [orin-deployment-report.json](../evidence/orin-deployment-report.json)。本页未包含 Orin TensorRT 部署结果。
