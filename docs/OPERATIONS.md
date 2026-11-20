# 运行维护和备份

## 服务生命周期

前台运行时用 Ctrl-C 正常退出。长期运行可创建用户级 systemd 单元，使用环境里 Python 的绝对路径、仓库的绝对路径和最终服务参数。

下面单元中的占位路径替换成本机值后，保存到用户配置目录 `~/.config/systemd/user/laya-jetson.service`。FA2 的目录指向已核验的扩展来源。选择 TRT 时把 ExecStart 改为 TRT 环境及 engine 参数，并设置 TensorRT Python 绑定路径。

```ini
[Unit]
Description=Laya Jetson decision service
After=network.target

[Service]
Type=simple
WorkingDirectory=/ABSOLUTE/PATH/LayaOnJetson
ExecStart=/ABSOLUTE/CONDA/envs/laya-jetson-fp16/bin/python scripts/serve.py --backend fa2 --model-dir models/multilingual --fa2-source vllm --vllm-fa2-dir vendor/vllm_fa2 --port 18048
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user start laya-jetson.service
systemctl --user status laya-jetson.service
journalctl --user -u laya-jetson.service -n 50
systemctl --user restart laya-jetson.service
systemctl --user stop laya-jetson.service
```

需要登录启动时按现场要求 enable 单元。用户退出后的运行方式取决于用户级 systemd 与 linger 配置，部署时记录现场决定。

## 故障定位

| 现象 | 检查与处理 |
| --- | --- |
| torch.cuda.is_available 为 false | 核对当前 Python、wheel 架构、JetPack、CUDA 和动态库，用 check_gpu.py 做实际 FP16 运算 |
| FA2 导入出现 undefined symbol | 比较 Torch 版本、C++ ABI、CUDA 和扩展编译版本，再在专用环境安装匹配构建 |
| FA2 缺少 kernel image | 检查 cubin/PTX 架构，在目标架构重新编译或换匹配的扩展 |
| 安装 FA2 后仍使用 SDPA | 使用本仓库的 FA2 启动参数，检查 /jetson/status 的实际调用计数 |
| TRT 输出异常或概率变化大 | 固定输入与 SDPA 比较，核对模型散列、mask、LayerNorm 和具体层精度，重建后回归 |
| 请求超过 TRT profile | 检查问题数、batch、序列长度和 marker 数，按实际需求重建 profile |
| /health 正常但 Octop 未调用 | 从 Octop 进程网络空间测 endpoint，核对保存设置和实际候选是否触发评分 |
| pip 与 Conda 的 setuptools 记录冲突 | 建环境时固定与 Torch 相容的 Conda setuptools。既有备份按实际安装版本审计后打包 |

## Conda 备份

在目标机器用独立打包工具环境运行 conda-pack，分别封装 PyTorch 和 TRT 环境。同时归档模型、启动脚本、FA2 扩展来源、ONNX、该机 engine、依赖清单及系统组件版本。所有文件保存在用户指定的备份位置。

```bash
conda create -n laya-pack python=3.12 pip
conda run -n laya-pack python -m pip install conda-pack
conda run -n laya-pack conda-pack -n laya-jetson-fp16 -o laya-fp16-env.tar.gz
conda run -n laya-pack conda-pack -n laya-jetson-trt -o laya-trt-env.tar.gz
sha256sum laya-fp16-env.tar.gz laya-trt-env.tar.gz
```

原始 Thor 两个完整备份均在新目录解包、执行 conda-unpack、验证依赖和 CUDA 后，成功返回 SDPA、FA2、TRT 的 HTTP 推理。归档依赖宿主机的匹配 JetPack、CUDA 和系统 TensorRT。跨设备时沿用源码和固定模型，在目标机器建立新环境并重建 engine。[Conda-pack 恢复说明](https://conda.github.io/conda-pack/)
