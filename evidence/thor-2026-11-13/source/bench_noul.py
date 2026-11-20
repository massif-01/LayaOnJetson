#!/usr/bin/env python3
"""Replay fixed synthetic Octop-style Noul batches over the Laya HTTP API."""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


STATE = "看一下最新的录音文件是什么。"
CATALOG = [
    ("skill", "录音检索", "列出录音文件，按最后修改时间判断哪一条录音最新。"),
    ("skill", "会议纪要", "从已有转写文本中提取讨论事项与待办。"),
    ("skill", "设备管理", "查看设备注册信息、在线状态与固件版本。"),
    ("skill", "知识库维护", "整理知识文档，更新标签和索引。"),
    ("skill", "邮件撰写", "起草邮件与附件说明。"),
    ("skill", "表格分析", "汇总电子表格数据，计算统计指标。"),
    ("skill", "内容翻译", "按指定语言翻译已有文本。"),
    ("skill", "周报整理", "根据项目更新编制每周进度。"),
    ("skill", "合同核对", "对比合同版本中的条款变更。"),
    ("skill", "日程安排", "查询可用时间并整理会议议程。"),
    ("tool", "录音索引查询", "查询录音文件名、保存位置和修改时间。"),
    ("tool", "订单检索", "按订单号查询履约和付款状态。"),
    ("tool", "转写任务查询", "按任务编号读取语音转写进度。"),
    ("tool", "报销单查询", "按申请人和月份查询报销记录。"),
    ("tool", "网络检查", "检查服务连通性和 DNS 解析结果。"),
    ("tool", "设备温度读取", "读取设备最新温度采样。"),
    ("tool", "文本摘要生成", "将输入文本缩写为要点列表。"),
    ("tool", "图片元数据读取", "读取图片尺寸与拍摄时间。"),
    ("tool", "联系人查询", "查询通讯录中的联系人。"),
    ("tool", "资源额度查询", "查询项目的资源额度和剩余用量。"),
]


def request_body(count, state=STATE):
    questions = {}
    for kind, label, description in CATALOG[:count]:
        category = {"skill": "技能", "tool": "工具"}[kind]
        questions[f"{kind}:{label}"] = {
            "type": "noul",
            "instructions": f"这轮需要{category}「{label}」。 {description}",
            "criteria": {"true": "适用", "false": "不适用"},
        }
    return json.dumps(
        {"state": state, "questions": questions, "model": "multilingual"},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def exchange(url, key, timeout, body=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    started = time.perf_counter()
    with urlopen(Request(url, data=body, headers=headers), timeout=timeout) as response:
        raw = response.read()
        timing = response.headers.get("X-Inference-Time-Ms")
    http_ms = (time.perf_counter() - started) * 1000
    return json.loads(raw), http_ms, float(timing) if timing is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Service base URL, e.g. http://127.0.0.1:18047")
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", required=True, help="Required reported device, e.g. cpu or cuda")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--questions", type=int, nargs="+", default=(1, 5, 10, 20))
    parser.add_argument("--state-file", type=Path, help="Use this UTF-8 text as the synthetic state")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup nonnegative")
    if any(not 1 <= count <= len(CATALOG) for count in args.questions):
        parser.error(f"--questions values must be in 1..{len(CATALOG)}")
    state = args.state_file.read_text(encoding="utf-8").strip() if args.state_file else STATE
    base_url = args.base_url.rstrip("/")
    key = os.environ.get("LAYA_API_KEY", "")
    timeout = 60

    def health():
        report, _, _ = exchange(f"{base_url}/health", key, timeout)
        device = report.get("checkpoint_devices", {}).get("multilingual", "")
        if device.split(":")[0] != args.device:
            raise RuntimeError(f"multilingual is on {device!r}; expected {args.device!r}")
        return report

    with args.output.open("x", encoding="utf-8") as output:
        def record(row):
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), file=output, flush=True)

        before = health()
        record({"phase": "health_before", "health": before})
        for count in args.questions:
            body = request_body(count, state)
            fingerprint = hashlib.sha256(body).hexdigest()
            expected_ids = set(json.loads(body)["questions"])
            durations = []
            for index in range(args.warmup + args.iterations):
                payload, http_ms, inference_ms = exchange(
                    f"{base_url}/v1/systemone", key, timeout, body
                )
                probabilities = {qid: answer["noul"] for qid, answer in payload["answers"].items()}
                if probabilities.keys() != expected_ids or any(
                    not 0 <= probability <= 1 for probability in probabilities.values()
                ):
                    raise RuntimeError(f"Invalid decision response for Q={count}")
                if inference_ms is None:
                    raise RuntimeError("Laya did not return X-Inference-Time-Ms")
                warm = index >= args.warmup
                record({
                    "phase": "measured" if warm else "warmup",
                    "questions": count,
                    "request_sha256": fingerprint,
                    "iteration": index + 1,
                    "model": payload.get("model"),
                    "usage": payload.get("usage"),
                    "http_ms": round(http_ms, 3),
                    "server_ms": inference_ms,
                    "probabilities": probabilities,
                })
                if warm:
                    durations.append(http_ms)
            p95 = sorted(durations)[math.ceil(0.95 * len(durations)) - 1]
            print(f"Q={count:2} n={len(durations)} HTTP p50={statistics.median(durations):.1f} ms "
                  f"p95={p95:.1f} ms", flush=True)
        after = health()
        record({"phase": "health_after", "health": after})
        if after.get("cpu_fallbacks") != before.get("cpu_fallbacks"):
            raise RuntimeError("Laya reported CPU fallback during the run")


if __name__ == "__main__":
    main()
