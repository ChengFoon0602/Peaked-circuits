# -*- coding: utf-8 -*-
"""exp_platform_pps.py — 用平台的 pauli-path 设备逐站点取 <Z_i>

背景：`mps.cpu` 在 49 比特 / 12109 门上，最大键维 16 的边缘已经全平，而键维 32
受 15 分钟/作业上限卡死。`pauli-path`（UI 里叫 `pps.cpu`，支持到 192 比特、免费）
是唯一还站得住的免费路径，而且它正是论文 4.4 节的方法，与本地 PPS 实验同源。

**平台限制（实测）**：`pauli-path` 只接受**一个**可观测量——

    BQSDKUsageError: Only a single observable is supported on 'pauli-path' device.
                     Received 'pauli_sum' with 12 observables.

所以拿 n 个 <Z_i> 必须提交 n 个作业（每个作业只算一个站点）。1000 作业/天够用，
但每个作业都要排队 + 收缩，时间上按站点数线性计。

用法
----
    :: 先测单站点，确认能跑通、看耗时
    python exp_platform_pps.py --qasm autotest_results/p3_qasm/n12_d8_t015_s0_G88.qasm ^
        --truth 100100001011 --sites 0

    :: 12 比特全跑（12 个作业）
    python exp_platform_pps.py --qasm ... --truth 100100001011 --sites all

    :: 49 比特先取前 6 个站点（6 个作业）
    python exp_platform_pps.py --circuit heavy_hex_49x4020 --sites 0-5

注意
----
* 站点编号是 QASM 里的比特编号（qiskit 约定）；拼出的串再与真值两种位序都比。
* 阈值越小项数越多、越慢；先用 1e-2 探路。
* 只允许免费设备；`pauli-path.gpu` 会被拒绝。
* csv 按站点合并写，支持分批累加（`--sites 0-5` 之后再跑 `--sites 6-11`）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH_DIR = os.path.join(HERE, "autotest_results", "p3_kremer_cpu")
OUT_DIR = os.path.join(HERE, "autotest_results", "p3_platform_pps")
QASM_DIR = "D:/桌面文件/大三/peaked_qasm"

DEVICE_ALIASES = {
    # 同一个设备在三种地方三种写法：
    #   定价页 / UI      -> "pps.cpu"（PPS = Pauli Path Simulation）
    #   官方文档表格      -> "pauli-path.cpu"（SDK 会拒绝）
    #   SDK DEVICE_TYPES -> "pauli-path"（唯一能过 validate_device 的）
    "pps": "pauli-path",
    "pps.cpu": "pauli-path",
    "pauli-path.cpu": "pauli-path",
}

FREE_DEVICES = {"pauli-path", "mps.cpu", "cpu"}


def load_truth(name):
    path = os.path.join(TRUTH_DIR, f"peaked_circuit_{name}.json")
    if not os.path.isfile(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("bitstring"), d.get("n_qubits")


def load_qasm(path):
    from qiskit import qasm2, qasm3
    with open(path, encoding="utf-8") as f:
        head = f.readline()
    if "3.0" in head or "3;" in head:
        return qasm3.load(path), "3.0"
    return qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS), "2.0"


def score_R(ans, guess):
    n = len(ans)
    return (n - sum(a != b for a, b in zip(ans, guess))) / n


def parse_sites(spec, n):
    """'all' / '0' / '0,3,7' / '0-5' → 站点编号列表。"""
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [s for s in out if s < 0 or s >= n]
    if bad:
        sys.exit(f"站点编号越界：{bad}（电路只有 0..{n - 1}）")
    return out


def single_site_sum(site, n):
    """构造**一个**可观测量：第 site 位是 Z，其余是 I。"""
    s = ["I"] * n
    s[site] = "Z"
    return [("".join(s), 1.0)]


def main():
    ap = argparse.ArgumentParser(description="平台 pauli-path 逐站点取 <Z_i>")
    ap.add_argument("--circuit", type=str, default="heavy_hex_49x4020")
    ap.add_argument("--qasm", type=str, default=None)
    ap.add_argument("--truth", type=str, default=None)
    ap.add_argument("--sites", type=str, default="0",
                    help="all / 0 / 0,3,7 / 0-5（默认只跑 0 号站点做探测）")
    ap.add_argument("--threshold", type=float, default=1e-2)
    ap.add_argument("--device", type=str, default="pauli-path")
    ap.add_argument("--token", type=str, default=None)
    ap.add_argument("--timeout", type=float, default=880)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.device = DEVICE_ALIASES.get(args.device.lower(), args.device.lower())
    if args.device not in FREE_DEVICES:
        sys.exit(f"拒绝在 {args.device} 上跑：只允许 {sorted(FREE_DEVICES)}。")

    name = args.circuit
    qasm = args.qasm or os.path.join(QASM_DIR, f"peaked_circuit_{name}.qasm")
    if not os.path.isfile(qasm):
        sys.exit(f"找不到 {qasm}")
    if args.qasm:
        name = os.path.basename(qasm).rsplit(".", 1)[0]

    truth, _ = (args.truth, None) if args.truth else load_truth(name)
    qc, ver = load_qasm(qasm)
    n = qc.num_qubits
    sites = parse_sites(args.sites, n)
    print(f"电路      : {name}  ({n} 比特 / {sum(qc.count_ops().values())} 门, QASM {ver})")
    print(f"设备      : {args.device}   阈值 = {args.threshold:g}")
    print(f"真值      : {truth or '✗ 无（拒绝打分）'}")
    print(f"站点      : {len(sites)} 个 → 需要 {len(sites)} 个作业 "
          f"（{sites[:6]}{'...' if len(sites) > 6 else ''}）")
    if not truth and not args.dry_run:
        sys.exit("没有真值就没法判断。")
    if args.dry_run:
        print(f"  示例可观测量 : Z 在第 {sites[0]} 位，其余 I")
        print("\n（--dry-run：未提交任何作业）")
        return

    token = args.token or os.environ.get("BLUEQUBIT_API_TOKEN")
    if not token:
        sys.exit("缺少 token：设置 BLUEQUBIT_API_TOKEN，或用 --token 传入。")
    import bluequbit
    bq = bluequbit.init(token)
    os.makedirs(OUT_DIR, exist_ok=True)

    vals = {}
    for k, site in enumerate(sites, 1):
        obs = single_site_sum(site, n)
        t0 = time.time()
        try:
            res = bq.run(qc, device=args.device, pauli_sum=obs,
                         options={"pauli_path_truncation_threshold": args.threshold},
                         timeout=args.timeout)
            ev = getattr(res, "expectation_value", None)
            v = float(np.real(ev)) if ev is not None else float("nan")
        except Exception as e:
            print(f"[{k}/{len(sites)}] site {site:2d}  失败：{type(e).__name__}: "
                  f"{str(e)[:200]}")
            continue
        dt = time.time() - t0
        vals[site] = v
        print(f"[{k}/{len(sites)}] site {site:2d}  <Z> = {v:+.6f}   "
              f"|Z| {abs(v):.3f}   {dt:.1f}s", flush=True)

    if not vals:
        sys.exit("一个站点都没成功，先看上面的报错。")

    filled = [s for s in range(n) if s in vals]
    bits = "".join("1" if vals[s] < 0 else "0" for s in filled)
    print(f"\n已取站点 {len(filled)}/{n}，|<Z>| 最大 {max(abs(v) for v in vals.values()):.4f}")
    if len(filled) == n:
        R = score_R(truth, bits)
        R_rev = score_R(truth, bits[::-1])
        print(f"边缘串  = {bits}")
        print(f"R = {R:.3f}   R(反转) = {R_rev:.3f}   "
              f"{'★ 完全恢复' if max(R, R_rev) == 1.0 else '未恢复'}")
    else:
        print("（站点没取全，不打分；已取值见 csv，可分批续跑）")

    path = os.path.join(OUT_DIR, f"pps_{name}_thr{args.threshold:g}.csv")
    merged = {}
    if os.path.isfile(path):                      # 合并写，支持分批累加
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if str(r.get("site", "")).isdigit():
                    merged[int(r["site"])] = r
    for s, v in vals.items():
        merged[s] = {"site": s, "expectation_Z": f"{v:.8f}",
                     "bit": "1" if v < 0 else "0"}
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["site", "expectation_Z", "bit"])
        w.writeheader()
        for s in sorted(merged):
            w.writerow(merged[s])
    print(f"写出 {path}（累计 {len(merged)} 个站点）")


if __name__ == "__main__":
    main()
