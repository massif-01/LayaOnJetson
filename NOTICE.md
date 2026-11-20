# 来源和依赖

本仓库整理了实际部署脚本、适配代码、合成请求及脱敏测试记录。第三方模型、库和二进制按各自许可获取和使用。

- Laya 来自 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)，使用 0.4.7。服务路由、模型装载及概率解码复用该项目。原始使用环境的源码散列保存在 requirements/thor-validated.json。
- multilingual 权重来自 [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual)，模型卡声明 Apache-2.0，revision 固定在 requirements/model.json。
- ModernBERT 编码器和 RoPE 调用来自 [Hugging Face Transformers](https://github.com/huggingface/transformers/tree/v4.57.6)，使用 4.57.6，Apache-2.0。
- vLLM 的 FlashAttention 包装层来自 [vLLM](https://github.com/vllm-project/vllm/tree/v0.21.0/vllm/vllm_flash_attn)，保留其 Apache-2.0 及 Tri Dao 版权声明。FlashAttention 内核按其 [BSD-3-Clause 许可](https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE) 使用。
- PyTorch、CUDA、TensorRT 分别按对应软件发行包和 NVIDIA 许可安装。
- TensorRT 方案研究参考了 [Laya Jetson 转换记录](https://github.com/suharvest/laya-on-rk3588-jetson) 和 [laya-cuda-bench](https://github.com/bhushankinge/laya-cuda-bench)。本仓库记录的 Thor 测试数据由本次现场部署生成。

Git 仓库收录的是我们编写的适配和执行入口。运行时加载的第三方源码和二进制位于部署机器的 vendor/ 或环境目录，随本地备份保留对应来源和许可文件。
