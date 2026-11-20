# AI 接手指令

目标是在用户指定的 Jetson 上恢复 Laya 工作成果，并按当前任务接入 Octop。先读 README、目标设备对应的 docs/THOR.md 或 docs/ORIN.md、docs/TENSORRT.md、docs/OCTOP.md 和 docs/RESULTS.md。

## 证据和范围

- Thor 留存源码与逐次记录，JetPack 7.1、CUDA 13.0、SM110。
- Orin 回执来自用户转贴，L4T 36.4.7、CUDA 12.6 构建、SM87、PyTorch 2.9.0，报告 FA2 和 Octop 真实请求验收通过。按用户要求不收录或索取同事电脑上的文件。仓库版脚本尚未在该 Orin 独立复测，重建步骤与原现场回执分开标记。
- Orin 编码器余弦相似度额外检查未达到 0.999 阈值，必须保留该限制，不能归纳为所有数值检查通过。其 Noul 对照在 0.60 阈值下未翻转，不能据此推断准确率一致。
- 模型固定在 requirements/model.json 的 revision 和六个文件散列。不同机器采用各自兼容的 CUDA PyTorch、FA2 和 TensorRT 构建。
- SDPA 和 FA2 使用同一个专用 PyTorch Conda 环境。TensorRT 使用另一个专用 Conda 环境。
- 服务验收包含 CUDA FP16、实际 attention 或 engine 执行、短长请求、完整答案和回退检查。Octop 接入任务还必须验证真实 Agent 轮次确实发出了评分请求。

## 工作方法

1. 查看设备树、L4T、JetPack、CUDA、GPU 架构、内存和当前服务负载。记录现场版本。
2. 用 conda create 建专用环境。先安装目标机器匹配的 CUDA PyTorch，再以 Torch 约束安装 Laya。脚本见 scripts/create_env.sh。
3. 下载或核对固定模型，先运行 SDPA。通过 scripts/serve.py --proof-only 输出真实 GPU 和算子证据。
4. FA2 按设备对应文档适配。独立 flash_attn 和 vLLM 扩展的函数签名不同，Thor 与 Orin 回执使用的 vLLM 版本也不同。先核对当前包装层的导入方式和接口，再验证双向全局、滑窗和 padding 语义。
5. TensorRT 按 docs/TENSORRT.md 在本机生成 engine。当前 profile 覆盖顺序布局的两个选项、1–5 行、序列长度 8–1024。超出 profile 时调整并重建，再重跑验证。
6. Octop 配置前核对实际版本和 API，保存本地回滚材料。根据 Octop 所在网络空间确定 endpoint，验证设置测试和一次真实任务。
7. 在环境完成、GPU 验证、每组实验和 Octop 接通节点同步进度。使用已有授权完成任务。

## 隐私

仓库提交范围为通用源码、合成测试和脱敏结果。SSH 信息、令牌、真实业务 payload、私有路径、录音及 Octop 本地配置放在仓库外。对所有待提交文件和 Git 历史检查凭据与隐私。内网地址在文档中用示例或占位符表示。

## 修改后的验证

- bash -n scripts/*.sh
- python -m compileall -q scripts
- scripts/summarize.py 重算 evidence/thor-2026-11-13 下的表格
- 修改 attention 时，验证全局、滑窗、不同长度 padding 和完整服务请求
- 修改 TensorRT 时，验证 profile 边界、输出有限性、完整服务请求和概率差
- git diff --check

写清本次实际验证的设备、依赖、请求样本和限制。每台新机器的验证结果保存在本地 results/，由用户选定并脱敏后再提交证据。
