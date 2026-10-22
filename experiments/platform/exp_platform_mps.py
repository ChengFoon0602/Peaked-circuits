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
    python experiments/platform/exp_platform_mps.py --dry-run                 # 只打印计划，不上传
    python experiments/platform/exp_platform_mps.py --circuit heavy_hex_49x4020 --chis 2,4,8
    python experiments/platform/exp_platform_mps.py --circuit heavy_hex_49x5072 --chis 2,4,8,16 --shots 2000

注意
----
* `mps.cpu` 免费但有 **15 分钟/作业** 上限、1000 作业/天。
* **采样比收缩还贵**（49 比特实测）：1000 shots 全作业 28 s，20000 shots 变成 281 s，
  即采样约 **14 ms/shot**。而平台的 "estimated runtime" 只算收缩、**不计 shots**
  （两次估计都是 58.4 s）。所以调大 shots 前先算这笔账。
* 低键维下输出在样本层面就是均匀的（20000 shots 出 20000 个不同串），此时
  argmax 无意义；必须看逐位边缘表决。`--no-save-counts` 可关掉原始样本落盘。
* **免费档对键维另有一道资金闸**：`heavy_hex_49x4020` 上 chi=16 可跑，chi=32 直接
  `NOT_ENOUGH_FUNDS`（"Your circuit is too large for the free tier — try a smaller
  bond dimension. We recommend ... at least $10.00"）。与 15 分钟时限无关。
* 平台自报的 `estimated runtime` 既不计 shots 也不准（chi=8 两次都报 58.4 s，实测
  28.3 s vs 280.6 s），不能用来排计划。
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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # experiments/<group>/ → 仓库根
TRUTH_DIR = os.path.join(ROOT, "autotest_results", "p3_kremer_cpu")
OUT_DIR = os.path.join(ROOT, "autotest_results", "p3_platform_mps")
# 公开 QASM 数据集（Quantum Advantage Tracker 下载）：优先环境变量，默认仓库内 data/qasm/
QASM_DIR = os.environ.get("PEAKED_QASM_DIR", os.path.join(ROOT, "data", "qasm"))

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


def marginal_vote(counts, n):
    """逐位多数表决——Kremer 的 `extract_bitstring` 用的就是这种读数。

    低键维 MPS 的输出在**样本层面**几乎是均匀的（1000 shots 里 1000 个不同的串，
    峰概率恰好 1/shots），此时 argmax 纯属噪声。但单比特边缘仍带有偏差：
    逐位统计 P(bit=1)，超过 0.5 记 1，拼成候选串。

    这也是本地 L2 的读数方式（`bitstring_from_marginals`），两条线才对得上。
    """
    ones = np.zeros(n)
    total = 0
    for s, c in counts.items():
        if len(s) != n:
            continue
        total += c
        for i, ch in enumerate(s):
            if ch == "1":
                ones[i] += c
    frac = ones / max(total, 1)
    bits = "".join("1" if f > 0.5 else "0" for f in frac)
    # 偏离 0.5 最小的那位决定这个候选串可信不可信
    return bits, frac, float(np.min(np.abs(frac - 0.5))), total


def run_one(bq, qc, chi, shots, timeout):
    """提交一次 mps.cpu 作业，返回 (counts, elapsed)。"""
    t0 = time.time()
    res = bq.run(qc, device="mps.cpu", shots=shots,
                 job_name=f"mps_chi{chi}",
                 options={"mps_bond_dimension": chi},
                 timeout=timeout)
    counts = res.get_counts()
    return counts, time.time() - t0


def run_observables(bq, qc, n, chi, timeout):
    """把 n 个单站点 <Z_i> 作为可观测量提交，返回逐站点期望值。

    pauli-path 明确只允许一个可观测量；mps.cpu 是否允许一次传多个尚未验证，
    报错就说明只能退回逐站点提交。
    """
    sums = []
    for s in range(n):
        pauli = "".join("Z" if i == s else "I" for i in range(n))
        sums.append([(pauli, 1.0)])
    t0 = time.time()
    res = bq.run(qc, device="mps.cpu", pauli_sum=sums,
                 options={"mps_bond_dimension": chi}, timeout=timeout)
    return flatten_expectations(res), time.time() - t0


def flatten_expectations(res):
    """把返回结果摊平成 (n,) 的 float 数组。

    实测有两种形状：
      * 一个 JobResult，`expectation_value` 是长度 n 的数组（mps.cpu 一次算完全部）
      * n 个 JobResult，各自 `expectation_value` 是标量
    早期版本只处理了后者，于是在前者上把数组塞进 float() 而报
    "truth value of an array with more than one element is ambiguous"。
    """
    res_list = res if isinstance(res, list) else [res]
    vals = []
    for r in res_list:
        ev = getattr(r, "expectation_value", None)
        if ev is None:
            continue
        vals.extend(np.asarray(ev, dtype=float).ravel().tolist())
    return np.asarray(vals, dtype=float)


def main():
    ap = argparse.ArgumentParser(description="平台 mps.cpu 的 chi 扫描")
    ap.add_argument("--circuit", type=str, default="heavy_hex_49x4020",
                    help="电路名（不含 peaked_circuit_ 前缀与 .qasm 后缀）")
    ap.add_argument("--qasm", type=str, default=None,
                    help="直接指定 QASM 文件，覆盖 --circuit（可用于我们自己的 12 比特电路）")
    ap.add_argument("--truth", type=str, default=None,
                    help="直接指定真值比特串（平台位序），覆盖从 json 读取")
    ap.add_argument("--chis", type=str, default="2,4,8",
                    help="bond dimension 列表，逗号分隔")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=900,
                    help="单作业等待上限（秒）；mps.cpu 平台侧上限 15 min")
    ap.add_argument("--device", type=str, default="mps.cpu")
    ap.add_argument("--token", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不提交作业")
    ap.add_argument("--no-save-counts", action="store_true",
                    help="默认会把原始 counts 落盘（gzip json），加此项则不存")
    ap.add_argument("--repeat", type=int, default=1,
                    help="同一 chi 重复提交几次，用来检验平台是否可复现（默认 1）")
    ap.add_argument("--job-id", type=str, default=None,
                    help="不提交新作业，取回该 job_id 的结果重新分析（崩溃后救数据用）")
    ap.add_argument("--observables", action="store_true",
                    help="不用采样，改为提交 n 个单站点 <Z_i> 可观测量直接取边缘"
                         "（无采样噪声；能否一次传多个取决于平台）")
    args = ap.parse_args()

    if args.device not in FREE_DEVICES:
        sys.exit(f"拒绝在 {args.device} 上跑：本脚本只允许 {sorted(FREE_DEVICES)}。")

    name = args.circuit
    if args.qasm:
        qasm = args.qasm
        name = os.path.basename(qasm).rsplit(".", 1)[0]
    else:
        qasm = os.path.join(QASM_DIR, f"peaked_circuit_{name}.qasm")
    if not os.path.isfile(qasm):
        cands = [os.path.basename(p) for p in glob.glob(os.path.join(QASM_DIR, "*.qasm"))]
        sys.exit(f"找不到 {qasm}\n当前 QASM 目录：{QASM_DIR}\n可用的有：{cands}\n"
                 "把 .qasm 放进去，或用环境变量 PEAKED_QASM_DIR 指向存放目录。")

    truth, n_truth = (args.truth, None) if args.truth else load_truth(name)
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

    if args.job_id:
        print(f"--- 取回作业 {args.job_id} ---", flush=True)
        vals = flatten_expectations(bq.get(args.job_id))
        print(f"  取到 {vals.size} 个期望值（应为 {n}）")
        if vals.size != n:
            print("  ⚠️  数量不符，下面按前 n 个打分，请人工核对。")
        vals = vals[:n]
        bits = "".join("1" if v < 0 else "0" for v in vals)
        mm = float(np.nanmin(np.abs(vals)))
        R = score_R(truth, bits)
        R_rev = score_R(truth, bits[::-1])
        ok = max(R, R_rev) == 1.0
        print(f"  <Z> 前 6 个 = {np.round(vals[:6], 4)}")
        print(f"  最小 |<Z>| = {mm:.4f}   最大 |<Z>| = {float(np.nanmax(np.abs(vals))):.4f}")
        print(f"  边缘串 = {bits}")
        print(f"  R = {R:.3f}   R(反转) = {R_rev:.3f}   {'★ 完全恢复' if ok else '未恢复'}")
        path = os.path.join(OUT_DIR, f"observables_{name}_job{args.job_id}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["site", "expectation_Z", "bit"])
            for i, (v, b) in enumerate(zip(vals, bits)):
                w.writerow([i, f"{v:.10f}", b])
        print(f"  写出 {path}")
        return

    if args.observables:
        for chi in chis:
            print(f"--- chi = {chi}（可观测量路线，无采样噪声）---", flush=True)
            try:
                runs, elapsed = [], 0.0
                for rep in range(max(1, args.repeat)):
                    v_rep, el = run_observables(bq, qc, n, chi, args.timeout)
                    runs.append(v_rep)
                    elapsed += el
                    if rep > 0:
                        d = np.max(np.abs(v_rep - runs[0]))
                        b0 = "".join("1" if x < 0 else "0" for x in runs[0])
                        b1 = "".join("1" if x < 0 else "0" for x in v_rep)
                        same = sum(x == y for x, y in zip(b0, b1))
                        print(f"    重复 {rep}: 与第 1 次的最大 |Δ<Z>| = {d:.6f}，"
                              f"符号相同的位 {same}/{len(b0)}", flush=True)
                vals = np.mean(runs, axis=0) if len(runs) > 1 else runs[0]
            except Exception as e:
                msg = str(e)
                print(f"  失败：{type(e).__name__}: {msg[:300]}")
                if "too large for the free tier" in msg or "NOT_ENOUGH_FUNDS" in msg:
                    print("  → 免费档对**键维**有资金闸：这条电路 chi=16 可以，chi=32 被拒，"
                          "并建议余额 ≥ $10。")
                    print("     与 15 分钟时限无关，纯账户/规模限制；换更小的 chi 或充值才能继续。")
                elif "single observable" in msg:
                    print("  → 该设备只接受一个可观测量，改用采样路线（去掉 --observables），"
                          "或用 exp_platform_pps.py 逐站点提交。")
                continue
            raw = os.path.join(OUT_DIR, f"observables_{name}_chi{chi}.csv")
            with open(raw, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["site", "expectation_Z"])
                for i, v in enumerate(vals):
                    w.writerow([i, f"{v:.10f}"])
            bits = "".join("1" if v < 0 else "0" for v in vals)
            mm = float(np.nanmin(np.abs(vals)))
            R = score_R(truth, bits)
            R_rev = score_R(truth, bits[::-1])
            ok = max(R, R_rev) == 1.0
            print(f"  <Z> 前 6 个 = {np.round(vals[:6], 4)}")
            print(f"  边缘串 = {bits}")
            print(f"  最小 |<Z>| = {mm:.4f}   R = {R:.3f}   R(反转) = {R_rev:.3f}   "
                  f"{'★ 完全恢复' if ok else '未恢复'}   用时 {elapsed:.1f}s", flush=True)
            rows.append({"circuit": name, "chi": chi, "ok": ok, "argmax": "",
                         "R_argmax": "", "R_argmax_rev": "",
                         "vote": bits, "R_vote": R, "R_vote_rev": R_rev,
                         "min_margin": mm / 2, "peak_prob": "", "n_distinct": "",
                         "shots": "", "elapsed_s": elapsed, "note": "observables"})
        if not rows:
            sys.exit("全部失败，没有可写的结果。")
        # 合并写：按 chi 保留最新一条，分批跑不会互相覆盖
        path = os.path.join(OUT_DIR, f"chi_sweep_{name}.csv")
        merged = {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    merged[str(r["chi"])] = {k: r.get(k, "") for k in rows[0]}
        for r in rows:
            merged[str(r["chi"])] = {k: r[k] for k in rows[0]}
        ordered = [merged[k] for k in sorted(merged, key=int)]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(ordered)
        print(f"\n写出 {path}（累计 {len(ordered)} 个 chi）")
        return

    for chi in chis:
        print(f"--- chi = {chi} ---", flush=True)
        try:
            counts, elapsed = run_one(bq, qc, chi, args.shots, args.timeout)
        except Exception as e:
            print(f"  失败：{type(e).__name__}: {str(e)[:200]}")
            rows.append({"circuit": name, "chi": chi, "ok": False,
                         "argmax": "", "R_argmax": "", "R_argmax_rev": "",
                         "vote": "", "R_vote": "", "R_vote_rev": "",
                         "min_margin": "", "peak_prob": "", "n_distinct": "",
                         "shots": "", "elapsed_s": "", "note": str(e)[:120]})
            continue
        total = sum(counts.values())
        top = max(counts, key=counts.get)
        peak = counts[top] / total
        R = score_R(truth, top)
        R_rev = score_R(truth, top[::-1])     # 位序兜底：两种都比一次

        if not args.no_save_counts:
            import gzip
            cpath = os.path.join(OUT_DIR, f"counts_{name}_chi{chi}.json.gz")
            with gzip.open(cpath, "wt", encoding="utf-8") as f:
                json.dump(counts, f)
            print(f"  （原始 counts 已存 {os.path.basename(cpath)}）")

        vote, frac, min_margin, _ = marginal_vote(counts, n)
        Rm = score_R(truth, vote)
        Rm_rev = score_R(truth, vote[::-1])
        best = max(R, R_rev, Rm, Rm_rev)
        ok = best == 1.0
        print(f"  argmax  = {top}")
        print(f"    峰概率 {peak:.4f}（1/shots = 1/{1/max(peak,1e-12):.0f}）  "
              f"R = {R:.3f}  R(反转) = {R_rev:.3f}")
        print(f"  边缘表决 = {vote}")
        print(f"    R = {Rm:.3f}  R(反转) = {Rm_rev:.3f}   最小偏离量 {min_margin:.4f}  "
              f"不同串 {len(counts)}/{total}")
        print(f"  → {'★ 完全恢复' if ok else '未恢复'}   用时 {elapsed:.1f}s", flush=True)
        rows.append({"circuit": name, "chi": chi, "ok": ok,
                     "argmax": top, "R_argmax": R, "R_argmax_rev": R_rev,
                     "vote": vote, "R_vote": Rm, "R_vote_rev": Rm_rev,
                     "min_margin": min_margin, "peak_prob": peak,
                     "n_distinct": len(counts), "shots": total,
                     "elapsed_s": elapsed, "note": ""})

    # 合并写：同一个 (circuit, chi) 只保留最新一条，避免后来的小批量把前面的覆盖掉
    path = os.path.join(OUT_DIR, f"chi_sweep_{name}.csv")
    merged = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                merged[(r["circuit"], str(r["chi"]))] = r
    for r in rows:
        merged[(r["circuit"], str(r["chi"]))] = {k: r[k] for k in rows[0]}
    ordered = [merged[k] for k in sorted(merged, key=lambda k: int(k[1]))]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(ordered)
    print(f"\n写出 {path}（累计 {len(ordered)} 行）")

    hit = [r for r in rows if r["ok"]]
    if hit:
        print(f"chi_break（平台，{name}）= {min(r['chi'] for r in hit)}")
    else:
        print("本次 chi 序列内没有完全恢复；继续加大 chi 或提高 shots。")


if __name__ == "__main__":
    main()
