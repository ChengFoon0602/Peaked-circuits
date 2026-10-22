# -*- coding: utf-8 -*-
"""bq_diag.py — 分清 401（token 无效）与 403（接口被限制）

`platform_smoke.py --stage 0` 收到 403 "Access denied for this endpoint" 时，
有两种可能：token 本身不被接受，或 token 有效但这个端点对当前账号/套餐不开放。
两者处理方式完全不同，所以这里直接发裸请求逐条对照。

关键点：不走 SDK 的自动重试（SDK 会重试 5 次，每次都要等），
并且**绝不打印 token**（只打长度与前 4 位）。

用法
----
    python experiments/platform/bq_diag.py                 # 读环境变量 BLUEQUBIT_API_TOKEN
    python experiments/platform/bq_diag.py --token xxx     # 或者显式传
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

BASE = "https://app.bluequbit.io/api/v1"


def probe(session, method, path, params=None, label=""):
    """发一次请求，返回 (status, body_text)。不重试。"""
    url = BASE + path
    try:
        r = session.request(method, url, params=params, timeout=30)
    except Exception as e:
        print(f"  {label:28s} 连接失败：{type(e).__name__}: {e}")
        return None, ""
    body = r.text
    try:                       # 能解析就压缩成一行，省得刷屏
        body = json.dumps(r.json(), ensure_ascii=False)[:400]
    except Exception:
        body = body[:400].replace("\n", " ")
    print(f"  {label:28s} HTTP {r.status_code}   {body}")
    return r.status_code, r.text


def main():
    ap = argparse.ArgumentParser(description="BlueQubit 接口诊断")
    ap.add_argument("--token", type=str, default=None)
    args = ap.parse_args()

    token = args.token or os.environ.get("BLUEQUBIT_API_TOKEN")
    if not token:
        sys.exit("缺少 token：设置 BLUEQUBIT_API_TOKEN，或用 --token 传入。")
    token = token.strip()

    print(f"token 长度 = {len(token)}  前4位 = {token[:4]}  后4位 = {token[-4:]}")
    if token != token.strip() or any(c in token for c in ('"', "'", " ")):
        print("⚠️  token 里含空格或引号，很可能是设置时粘错了。")

    s = requests.Session()
    s.headers.update({
        # SDK 源码里就是这么发鉴权头的（backend_connection.py）
        "Authorization": f"SDK {token}",
        "User-Agent": "BlueQubit SDK diag (raw probe)",
    })

    print("\n[1] 用已存在的接口验 token（/jobs）")
    st_jobs, _ = probe(s, "GET", "/jobs", {"limit": 1}, "GET /jobs")

    print("\n[2] 目标接口（/peaked-circuits/file）")
    probe(s, "GET", "/peaked-circuits/file", {"difficulty": 40}, "difficulty=40")
    probe(s, "GET", "/peaked-circuits/file", {"difficulty": 1},  "difficulty=1 (无效值)")

    print("\n[3] 可能是相邻路径（探一下有没有别的入口）")
    for p in ("/peaked-circuits", "/peaked-circuits/list", "/peaked-circuits/difficulties"):
        probe(s, "GET", p, None, f"GET {p}")

    print("\n—— 怎么读这个结果 ——")
    print("  /jobs 200 + 目标 403  → token 有效，是端点的权限问题（见下）")
    print("  /jobs 401             → token 本身无效，重新复制")
    print("  difficulty=1 与 40 报错不同 → 是参数/难度问题，不是端点权限")


if __name__ == "__main__":
    main()
