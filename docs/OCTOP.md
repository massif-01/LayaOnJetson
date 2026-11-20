# 接通 Octop

## 现场信息

先记录 Octop 版本、octop-harness 版本、启动方式和网络空间。当前仓库只保存通用操作流程，Thor 的历史测量来自独立 HTTP 服务。另一台 Orin 的成功接入由用户确认，详细回执后补。

Octop 的决策客户端通过 HTTP 请求配置的 endpoint。目标为本服务的 `/v1/systemone`，模型填 `multilingual`。内置 Laya 安装器的已查版本默认 CPU，独立 GPU 服务使用本页的 endpoint 配置。[Octop Laya 安装实现](https://github.com/TencentCloud/Octop/blob/4022e24e/src/octop/infra/agents/settings/laya_runtime.py) · [Harness HTTP 客户端](https://github.com/TencentCloud/octop-harness/blob/65b54561f39327b8d6a144a377b00230002cebad/src/octop_harness/decision/client.py)

## 服务地址

| Octop 所在位置 | Laya 监听及 endpoint |
| --- | --- |
| 同一个宿主机 | 监听 127.0.0.1，用 http://127.0.0.1:18048/v1/systemone |
| Docker bridge 容器 | Laya 监听宿主机容器可达的地址，从容器内验证。可配置 host-gateway 别名，endpoint 使用该别名 |
| 另一台机器 | 监听指定的 LAN 接口，endpoint 使用该设备的主机名和端口 |

服务支持 `--host`、`--port`。上游 Laya 通过 `LAYA_API_KEY` 启用 Bearer 鉴权，该值保存在部署机器的本地运行配置中。访问 `/health` 详细设备信息和 `/jetson/status` 时使用同一密钥。

## 配置

推荐使用 Octop 管理界面的决策模型设置。

1. 保存当前 provider、endpoint、model、skills、tools、team、threshold 和 tool_min_count，作为本地回滚记录。有凭据时按当前 Octop 的凭据管理方式保留恢复途径。
2. provider 选择 laya，endpoint 填从 Octop 进程可达的完整 `/v1/systemone` 地址，model 填 multilingual。启用决策功能。
3. 核对技能、工具和团队筛选开关。选择符合目标 Agent 的类别。
4. 使用 Octop 自身的连接测试，确认返回了有效概率。
5. 新会话执行目标任务，在 Laya 日志中核对 POST 请求、响应状态和时刻，并确认主 Agent 正常完成工具调用。

## API 操作参考

先从当前安装的 OpenAPI 或源码确认路由前缀、请求体和鉴权。已查版本提供 GET /settings/decision、PUT /settings/decision、POST /settings/decision/test。写入设置会重建运行中的 Agent 中间件。[Octop 决策设置 API](https://github.com/TencentCloud/Octop/blob/4022e24e/src/octop/api/routers/decision.py)

下面的变量在当前终端设置，令牌从用户本机获取。保存文件位于 Git 忽略的 runtime/。

```bash
export OCTOP_BASE_URL='http://localhost:REPLACE_WITH_OCTOP_PORT'
read -r -s -p 'Octop token: ' OCTOP_TOKEN
printf '\n'
export OCTOP_TOKEN
mkdir -p runtime
umask 077
curl -fsS -H "Authorization: Bearer $OCTOP_TOKEN" \
  "$OCTOP_BASE_URL/settings/decision" > runtime/octop-before.json
```

先用真实 endpoint 测试。若 Octop 有 API 前缀，在 BASE_URL 中包含该前缀。

```bash
cat > runtime/decision-test.json <<'JSON'
{
  "provider": "laya",
  "endpoint": "http://127.0.0.1:18048/v1/systemone",
  "model": "multilingual"
}
JSON
curl -fsS -X POST -H "Authorization: Bearer $OCTOP_TOKEN" \
  -H 'Content-Type: application/json' --data-binary @runtime/decision-test.json \
  "$OCTOP_BASE_URL/settings/decision/test"
```

如 Laya 启用了密钥，在本地请求文件中加上 `api_key`。写入设置时按当前版本的字段准备 `runtime/decision-next.json`，填入 provider、enabled、endpoint、model，并显式保留原有筛选开关、threshold 和 tool_min_count。如该版本使用 services 列表，同步保存完整列表内对应服务，保留其他服务项。

```bash
curl -fsS -X PUT -H "Authorization: Bearer $OCTOP_TOKEN" \
  -H 'Content-Type: application/json' --data-binary @runtime/decision-next.json \
  "$OCTOP_BASE_URL/settings/decision"
curl -fsS -H "Authorization: Bearer $OCTOP_TOKEN" \
  "$OCTOP_BASE_URL/settings/decision" > runtime/octop-after.json
```

回滚时按相同 API 请求体重建原设置并回读。当前实现切换某个服务 endpoint 时可能清除该服务的原密钥，API 返回的 api_key_set 只表示是否存在密钥，因此旧配置的凭据恢复需要使用本机原有凭据来源。

## 验收记录

记录最终保存值、从 Octop 所在网络空间测试的结果、新会话的一次请求时间线、Laya 收到评分请求的日志以及 Agent 工具动作。只读目录任务可使用用户批准的合成目录或测试文件。日志和真实请求留在本机，再选择脱敏信息整理回执。

候选不足时中间件可能跳过评分。已有工作区 MD 文件数量与 skill metadata 数量分开核对。为验收准备能触发评分的最小候选场景，再看服务日志。连接测试和真实 Agent 请求各保留一份记录。
