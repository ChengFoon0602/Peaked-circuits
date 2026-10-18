# -*- coding: utf-8 -*-
"""exp_platform_pps.py — 用平台的 pauli-path 设备直接取逐位 <Z_i>

动机：`mps.cpu` 在 49 比特 / 12109 门上，最大键维 16 的边缘已经全平，而
键维 32 又受 15 分钟/作业上限卡死。`pauli-path` 支持到 192 比特且免费，
如果它能在可接受的阈值下给出有偏的 <Z_i>，那就是 49 比特上唯一还站得住的
免费攻击路径——而且它正是论文 4.4 节的方法，与本地 PPS 实验同一个问题。

与本地 PPS 的对应关系：

| 本地 `exp_pps.py` | 平台 pauli-path |
|---|---|
| 反向传播 Pauli 和 | 平台实现 |
| `standard` / `relative` 两种截断（我们手搓） | `pauli_path_truncation_threshold`（只有一种，语义未公开） |
| 输出逐位 <Z_i> | 同样输出 expectation_value |

用法
----
    python exp_platform_pps.py --dry-run
    python exp_platform_pps.py --circuit heavy_hex_49x4020 --threshold 1e-2

注意
----
* pauli_sum 传 49 个**单站点**的和（每个是 `[( "Z..I..", 1.0 )]`），
  平台会返回 49 个 JobResult，各带一个 expectation_value。
* 49 比特的 Pauli 串很长，阈值越小项数越多、越慢；先用大阈值探路。
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
    # 同一个设备在不同地方有三种写法：
    #   平台定价页 / UI  -> "pps.cpu"（PPS = Pauli Path Simulation）
    #   官方文档表格      -> "pauli-path.cpu"（这个 SDK 反而会拒绝）
    #   SDK DEVICE_TYPES -> "pauli-path"（唯一能通过 validate_device 的）
    "pps": "pauli-path",
    "pps.cpu": "pauli-path",
    "pauli-path.cpu": "pauli-path",
    "pauli-path.gpu": "pauli-path.gpu",
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


def make_single_site_sums(n):
    """49 个单站点 Pauli 和，每个形如 [[('Z..I..', 1.0)]]。

    外层列表 = 多个 Pauli 和；每个和是「一个 (串, 系数) 元组」组成的列表。
    """
    out = []
    for i in range(n):
        s = ["I"] * n
        s[i] = "Z"
        out.append([("".join(s), 1.0)])
    return out


def main():
    ap = argparse.ArgumentParser(description="平台 pauli-path 取逐位 <Z_i>")
    ap.add_argument("--circuit", type=str, default="heavy_hex_49x4020")
    ap.add_argument("--qasm", type=str, default=None)
    ap.add_argument("--truth", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=1e-2,
                    help="pauli_path_truncation_threshold")
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

    truth, n_truth = (args.truth, None) if args.truth else load_truth(name)
    qc, ver = load_qasm(qasm)
    n = qc.num_qubits
    print(f"电路      : {name}  ({n} 比特 / {sum(qc.count_ops().values())} 门, QASM {ver})")
    print(f"设备      : {args.device}   阈值 = {args.threshold:g}")
    print(f"真值      : {truth or '✗ 无（拒绝打分）'}")
    if not truth and not args.dry_run:
        sys.exit("没有真值就没法判断。")

    sums = make_single_site_sums(n)
    print(f"Pauli 和  : {len(sums)} 个单站点 Z（共 {len(sums) * 1} 项）")
    print(f"  示例    : {sums[0][0][0][:20]}... 与 {sums[-1][0][0][-20:]}...")

    if args.dry_run:
        print("\n（--dry-run：未提交任何作业）")
        return

    token = args.token or os.environ.get("BLUEQUBIT_API_TOKEN")
    if not token:
        sys.exit("缺少 token：设置 BLUEQUBIT_API_TOKEN，或用 --token 传入。")
    import bluequbit
    bq = bluequbit.init(token)

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("\n提交中（可能是 49 个作业，也可能平台合并成一次）...", flush=True)
    try:
        res = bq.run(qc, device=args.device, pauli_sum=sums,
                     options={"pauli_path_truncation_threshold": args.threshold},
                     timeout=args.timeout)
    except Exception as e:
        print(f"失败：{type(e).__name__}: {str(e)[:400]}")
        sys.exit(1)
    elapsed = time.time() - t0

    res_list = res if isinstance(res, list) else [res]
    print(f"返回 {len(res_list)} 个结果，用时 {elapsed:.1f}s")
    vals = []
    for i, r in enumerate(res_list):
        ev = getattr(r, "expectation_value", None)
        vals.append(np.real(ev) if ev is not None else np.nan)
    vals = np.asarray(vals, dtype=float)
    print("  <Z_i> 前 8 个 :", np.round(vals[:8], 4))
    print(f"  |<Z_i>| 最大 : {np.nanmax(np.abs(vals)):.4f}   "
          f"最小 : {np.nanmin(np.abs(vals)):.4f}")

    # Z=+1 对应 |0>，所以 <Z><0 记 1
    bits = "".join("1" if v < 0 else "0" for v in vals)
    R = score_R(truth, bits)
    R_rev = score_R(truth, bits[::-1])
    print(f"  边缘串 = {bits}")
    print(f"  R = {R:.3f}   R(反转) = {R_rev:.3f}   "
          f"{'★ 完全恢复' if max(R, R_rev) == 1.0 else '未恢复'}")

    path = os.path.join(OUT_DIR, f"pps_{name}_thr{args.threshold:g}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["site", "expectation_Z", "bit"])
        for i, (v, b) in enumerate(zip(vals, bits)):
            w.writerow([i, f"{v:.8f}", b])
        w.writerow([])
        w.writerow(["summary", f"R={R:.4f}", f"R_rev={R_rev:.4f}"])
        w.writerow(["summary", f"max|Z|={np.nanmax(np.abs(vals)):.4f}", f"elapsed={elapsed:.1f}s"])
    print(f"\n写出 {path}")


if __name__ == "__main__":
    main()
