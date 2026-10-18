# -*- coding: utf-8 -*-
"""exp_platform_mps.py — 用平台的 mps.cpu 复现我们的 chi_break 实验（平台版 L2）

本地 L2（`exp_mps_break.py`）给出的是：n=12/16 时，恢复正确比特串所需的
MPS 键维 chi_break 随门数上升。这里换一个完全独立的对象做同样的测量——
**平台的 MPS 实现 + 49 比特公开电路**，看它需要多大 bond dimension 才能
把峰值串恢复出来。

与本地实验的差别（写材料时要讲清）：

| | 本地 L2 | 本脚本 |
|---|---|---|
| 电路 | 自己训练，n=12/16，G=88..660 | 公开电路，49 比特，G=12109/15265 |
| MPS 实现 | 自写（中心规范 + 逐门 SVD） | 平台（quimb 系） |
| 布局 | 可做 RCM 重排对照 | 电路自带 heavy-hex 布局，不可改 |
| 真值 | 训练时已知 | 用我们已逐位复现的恢复串 |

真值来源：`autotest_results/p3_kremer_cpu/<circuit>.json` 的 `bitstring` 字段
（`heavy_hex_49x4020` / `49x5072` 已与 Kremer 作者输出逐位一致）。
P9（56 比特）没有真值，脚本会拒绝在没有真值的情况下打分。

用法
----
    python exp_platform_mps.py --dry-run                 # 只打印计划，不上传
    python exp_platform_mps.py --circuit heavy_hex_49x4020 --chis 2,4,8
    python exp_platform_mps.py --circuit heavy_hex_49x5072 --chis 2,4,8,16 --shots 2000

注意
----
* `mps.cpu` 免费但有 **15 分钟/作业** 上限、1000 作业/天。
* 只允许 `mps.cpu`（免费）；`mps.gpu` / `gpu` / `quantum` 会被拒绝。
* 账户余额 $0 不影响 mps.cpu，但会让任何计费设备立即失败。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH_DIR = os.path.join(HERE, "autotest_results", "p3_kremer_cpu")
OUT_DIR = os.path.join(HERE, "autotest_results", "p3_platform_mps")
QASM_DIR = "D:/桌面文件/大三/peaked_qasm"

FREE_DEVICES = {"mps.cpu", "cpu", "pauli-path"}


def load_truth(name):
    """从 Kremer 复现的 json 里取真值比特串；没有就返回 None。"""
    path = os.path.join(TRUTH_DIR, f"peaked_circuit_{name}.json")
    if not os.path.isfile(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    # bitstring 是按「平台/QASM 约定」记录的那一份（Kremer 的 pred_bitstring 反转后）
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


def run_one(bq, qc, chi, shots, timeout):
    """提交一次 mps.cpu 作业，返回 (counts, elapsed)。"""
    t0 = time.time()
    res = bq.run(qc, device="mps.cpu", shots=shots,
                 job_name=f"mps_chi{chi}",
                 options={"mps_bond_dimension": chi},
                 timeout=timeout)
    counts = res.get_counts()
    return counts, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="平台 mps.cpu 的 chi 扫描")
    ap.add_argument("--circuit", type=str, default="heavy_hex_49x4020",
                    help="电路名（不含 peaked_circuit_ 前缀与 .qasm 后缀）")
    ap.add_argument("--chis", type=str, default="2,4,8",
                    help="bond dimension 列表，逗号分隔")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=900,
                    help="单作业等待上限（秒）；mps.cpu 平台侧上限 15 min")
    ap.add_argument("--device", type=str, default="mps.cpu")
    ap.add_argument("--token", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不提交作业")
    args = ap.parse_args()

    if args.device not in FREE_DEVICES:
        sys.exit(f"拒绝在 {args.device} 上跑：本脚本只允许 {sorted(FREE_DEVICES)}。")

    name = args.circuit
    qasm = os.path.join(QASM_DIR, f"peaked_circuit_{name}.qasm")
    if not os.path.isfile(qasm):
        cands = [os.path.basename(p) for p in glob.glob(os.path.join(QASM_DIR, "*.qasm"))]
        sys.exit(f"找不到 {qasm}\n可用的有：{cands}")

    truth, n_truth = load_truth(name)
    qc, ver = load_qasm(qasm)
    n = qc.num_qubits
    nops = sum(qc.count_ops().values())
    chis = [int(x) for x in args.chis.split(",") if x.strip()]

    print(f"电路      : {name}  ({n} 比特 / {nops} 门, QASM {ver})")
    print(f"设备      : {args.device}   shots={args.shots}")
    print(f"chi 序列  : {chis}")
    print(f"真值      : {truth if truth else '✗ 无（拒绝打分）'}")
    if truth and n_truth and int(n_truth) != n:
        sys.exit(f"真值长度 {n_truth} 与电路比特数 {n} 不符，先查电路文件。")
    if not args.dry_run and not truth:
        sys.exit("没有真值就没法判断恢复是否成功。可先用本地 Kremer 流程求出真值。")

    if args.dry_run:
        for chi in chis:
            print(f"  [计划] options={{'mps_bond_dimension': {chi}}}  shots={args.shots}")
        print("\n（--dry-run：未提交任何作业）")
        return

    token = args.token or os.environ.get("BLUEQUBIT_API_TOKEN")
    if not token:
        sys.exit("缺少 token：设置 BLUEQUBIT_API_TOKEN，或用 --token 传入。")
    import bluequbit
    bq = bluequbit.init(token)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    print()
    for chi in chis:
        print(f"--- chi = {chi} ---", flush=True)
        try:
            counts, elapsed = run_one(bq, qc, chi, args.shots, args.timeout)
        except Exception as e:
            print(f"  失败：{type(e).__name__}: {str(e)[:200]}")
            rows.append({"circuit": name, "chi": chi, "ok": False,
                         "argmax": "", "R": "", "R_rev": "", "peak_prob": "",
                         "elapsed_s": "", "note": str(e)[:120]})
            continue
        total = sum(counts.values())
        top = max(counts, key=counts.get)
        peak = counts[top] / total
        R = score_R(truth, top)
        R_rev = score_R(truth, top[::-1])     # 位序兜底：两种都比一次
        ok = max(R, R_rev) == 1.0
        print(f"  argmax = {top}")
        print(f"  峰概率 = {peak:.4f}   R = {R:.3f}   R(反转) = {R_rev:.3f}   "
              f"{'★ 完全恢复' if ok else ''}   用时 {elapsed:.1f}s", flush=True)
        rows.append({"circuit": name, "chi": chi, "ok": ok, "argmax": top,
                     "R": R, "R_rev": R_rev, "peak_prob": peak,
                     "elapsed_s": elapsed, "note": ""})

    path = os.path.join(OUT_DIR, f"chi_sweep_{name}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n写出 {path}")

    hit = [r for r in rows if r["ok"]]
    if hit:
        print(f"chi_break（平台，{name}）= {min(r['chi'] for r in hit)}")
    else:
        print("本次 chi 序列内没有完全恢复；继续加大 chi 或提高 shots。")


if __name__ == "__main__":
    main()
