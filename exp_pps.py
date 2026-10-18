# -*- coding: utf-8 -*-
"""exp_pps.py — 第三篇 §4.4：Pauli 路径（Pauli sum）攻击 + relative truncation 对照

论文的攻击 #4 是 Pauli Path Simulation（PPS）：把可观测量按 Pauli 基反向穿过电路展开，
每穿过一个门就展开成至多 16 项。项数指数增长，所以必须截断。论文的方法学贡献是
**relative truncation**：标准做法丢 |c_P| <= Δ，他们改成丢 |c_P| <= ||O_k||·Δ，
其中 ||O_k|| 是当前 Pauli 和的 Frobenius 范数——避免大电路里系数整体"熄火"后
一刀切掉全部尾部项。

本脚本
------
1. 把 <0|U† Z_i U|0> 反向传播（逐门 undo），得到 Z_i 的 Pauli 和近似；
2. 用 <0|P|0>（只留全 I/Z 的 Pauli 串）取边缘 <Z_i>；
3. 扫 Δ 与两种截断，比较「相同保留项数下的精度」（Pareto 曲线）与「R vs 门数」。

正确性门槛：Δ = 0（不截断）时，得到的 <Z_i> 必须与精确态矢量逐位相等。

用法
----
    python exp_pps.py verify                       # Δ=0 vs 精确态矢量
    python exp_pps.py sweep --ns 12 --depths 220   # 扫 Δ 与两种截断
    python exp_pps.py report
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import time

import numpy as np

from exp_mps_break import (CIRC_DIR, bitstring_from_marginals, gate_pattern,
                           load_circuits, norm_gate, score_R, sv_gates_to_state,
                           sv_marginals)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "autotest_results", "p3_pps")

# Pauli 编码：0=I, 1=X, 2=Z, 3=Y；每个比特占 2 bit，site q 占 bit 2q..2q+1
I2 = np.eye(2, dtype=complex)
PAULI = {
    0: I2,
    1: np.array([[0, 1], [1, 0]], dtype=complex),          # X
    2: np.array([[1, 0], [0, -1]], dtype=complex),         # Z
    3: np.array([[0, -1j], [1j, 0]], dtype=complex),       # Y
}
PAIR = {cp: np.kron(PAULI[cp & 3], PAULI[(cp >> 2) & 3]) for cp in range(16)}


def gate_map(u):
    """16x16 复矩阵：M[cp_new, cp_old] = Tr(P_new† U† P_old U)/4。"""
    u = np.asarray(u, dtype=complex)
    m = np.empty((16, 16), dtype=complex)
    for cold in range(16):
        w = u.conj().T @ PAIR[cold] @ u
        for cnew in range(16):
            m[cnew, cold] = np.trace(PAIR[cnew].conj().T @ w) / 4.0
    return m


def propagate(n, gates, site, scheme="standard", delta=0.0, hard_cap=200000):
    """反向把 Z_site 传播过电路，返回 (⟨Z_site⟩ 的估计, 诊断)。

    scheme: 'standard' -> 丢 |c| <= delta
            'relative' -> 丢 |c| <= delta * ||O_k||_F
    """
    t0 = time.time()
    keys = np.array([2 << (2 * site)], dtype=np.uint64)   # 只在该 site 放一个 Z
    coeff = np.array([1.0 + 0j])

    term_counts, capped = [], False
    for i0, j0, u0 in reversed(gates):
        i, j, u = norm_gate(i0, j0, u0)
        m = gate_map(u)
        mask = np.uint64((3 << (2 * i)) | (3 << (2 * j)))
        cp_old = (((keys >> np.uint64(2 * i)) & np.uint64(3)).astype(np.int64)
                  | (((keys >> np.uint64(2 * j)) & np.uint64(3)).astype(np.int64) << 2))
        base = keys & ~mask

        nc = np.arange(16, dtype=np.uint64)
        nk = (base[None, :]
              | ((nc & np.uint64(3)) << np.uint64(2 * i))[:, None]
              | (((nc >> np.uint64(2)) & np.uint64(3)) << np.uint64(2 * j))[:, None])
        nc_coef = m[:, cp_old] * coeff[None, :]

        k = nk.ravel()
        c = nc_coef.ravel()
        # 先丢掉数值零，避免数组无谓膨胀
        keep0 = np.abs(c) > 1e-15
        k, c = k[keep0], c[keep0]
        order = np.argsort(k, kind="stable")
        k, c = k[order], c[order]
        uk, inv = np.unique(k, return_inverse=True)
        csum = np.zeros(uk.size, dtype=complex)
        np.add.at(csum, inv, c)

        # 截断
        if delta > 0:
            thr = delta
            if scheme == "relative":
                thr = delta * float(np.sqrt(np.sum(np.abs(csum) ** 2)))
            keep = np.abs(csum) > thr
            uk, csum = uk[keep], csum[keep]

        if uk.size > hard_cap:      # 安全阀：按 |c| 保留最大的 hard_cap 项
            idx = np.argpartition(-np.abs(csum), hard_cap)[:hard_cap]
            uk, csum = uk[idx], csum[idx]
            capped = True

        keys, coeff = uk, csum
        term_counts.append(int(uk.size))

    # <0|P|0>：只有每个 site 都是 I 或 Z 的项存活
    ok = np.ones(keys.size, dtype=bool)
    for q in range(n):
        code = ((keys >> np.uint64(2 * q)) & np.uint64(3)).astype(np.int64)
        ok &= ((code & 1) == 0)
    val = float(np.real(np.sum(coeff[ok])))

    return val, {
        "seconds": time.time() - t0,
        "terms_max": int(np.max(term_counts)) if term_counts else 1,
        "terms_mean": float(np.mean(term_counts)) if term_counts else 1.0,
        "terms_final": int(keys.size),
        "capped": capped,
    }


def marginals_pps(n, gates, scheme, delta, hard_cap=200000):
    z = np.empty(n)
    terms_max, capped = 0, False
    sec = 0.0
    for q in range(n):
        v, d = propagate(n, gates, q, scheme, delta, hard_cap)
        z[q] = v
        terms_max = max(terms_max, d["terms_max"])
        capped = capped or d["capped"]
        sec += d["seconds"]
    return z, {"seconds": sec, "terms_max": terms_max, "capped": capped}


# ============================================================================
def x_mask_gates(n, bits):
    """把峰值从 0^n 搬到比特串 bits 上：在每个 1 的位置补一个 X（借相邻站放 2-qubit 门）。"""
    x = PAULI[1]
    out = []
    for q, b in enumerate(bits):
        if not b:
            continue
        if q < n - 1:
            out.append((q, q + 1, np.kron(x, I2)))
        else:
            out.append((q - 1, q, np.kron(I2, x)))
    return out


def cmd_verify(args):
    """Δ=0 时 PPS 必须与精确态矢量完全一致。"""
    import quimb as qu
    n, depth = args.n, args.depth
    qu.seed_rand(args.seed)
    pairs = gate_pattern(n, depth)
    gates = [(i, j, np.asarray(qu.rand_uni(4, dtype=complex))) for i, j in pairs]
    psi = sv_gates_to_state(n, gates)
    z_ex = sv_marginals(psi)
    z_pp, d = marginals_pps(n, gates, "standard", 0.0)
    err = float(np.max(np.abs(z_pp - z_ex)))
    print(f"n={n} depth={depth} G={len(gates)}  Δ=0  max|<Z>_pps-<Z>_ex| = {err:.3e}  "
          f"terms_max={d['terms_max']}  用时={d['seconds']:.2f}s  "
          f"{'✓' if err < 1e-10 else '✗✗✗'}")
    print("   精确：", np.array2string(z_ex, precision=6, max_line_width=140))
    print("   PPS ：", np.array2string(z_pp, precision=6, max_line_width=140))


def cmd_sweep(args):
    circs = load_circuits()
    keep = {int(x) for x in args.ns.split(",")}
    depths = {int(x) for x in args.depths.split(",")} if args.depths else None
    circs = [c for c in circs if c["n"] in keep
             and (depths is None or c["depth"] in depths)
             and c["seed"] == args.seed_base]
    if not circs:
        raise SystemExit("没有符合条件的电路（PPS 默认只跑 s0 实例）")
    os.makedirs(OUT, exist_ok=True)
    deltas = [float(x) for x in args.deltas.split(",")]
    schemes = args.schemes.split(",")
    rows = []
    out_path = os.path.join(OUT, args.out)

    def flush_rows():
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    for rec in circs:
        n = rec["n"]
        # 峰值搬到随机串 s 上：否则答案恒为 0^n，而截断到 "全零估计" 也会被判满分
        rng = np.random.default_rng(args.mask_seed)
        bits = [int(b) for b in rng.integers(0, 2, n)]
        target = "".join(str(b) for b in bits)
        gates = rec["_gates"] + x_mask_gates(n, bits)
        psi = sv_gates_to_state(n, gates)
        z_ex = sv_marginals(psi)
        R_ex = score_R(target, bitstring_from_marginals(z_ex))
        print(f"\n=== {rec['_file']}  G={rec['gate_count']} δ={rec['delta']:.4f} "
              f"secret={target} 精确边缘 R={R_ex:.3f} ===", flush=True)
        for scheme in schemes:
            for delta in deltas:
                z, d = marginals_pps(n, gates, scheme, delta, hard_cap=args.cap)
                guess = bitstring_from_marginals(z)
                R = score_R(target, guess)
                err = float(np.max(np.abs(z - z_ex)))
                mean_err = float(np.mean(np.abs(z - z_ex)))
                dead = int(np.sum(np.abs(z) < 1e-12))     # 被截断掐死、无信息的站点数
                R0 = score_R(target, "0" * n)             # 全零猜测的基线 R
                print(f"  {scheme:8s} Δ={delta:<8.0e} R={R:.3f} "
                      f"maxerr={err:.3e} meanerr={mean_err:.3e} dead={dead:2d} "
                      f"terms_max={d['terms_max']:>7d} "
                      f"t={d['seconds']:6.2f}s{'  [撞上限]' if d['capped'] else ''}",
                      flush=True)
                rows.append(dict(case=rec["_file"], n=n, depth=rec["depth"],
                                 gate_count=rec["gate_count"], delta_circ=rec["delta"],
                                 scheme=scheme, trunc=delta, R=R, maxerr=err,
                                 mean_err=mean_err, dead_sites=dead, R_allzero=R0,
                                 terms_max=d["terms_max"], seconds=d["seconds"],
                                 capped=d["capped"], target=target,
                                 guess=guess, R_exact=R_ex))
                flush_rows()          # 增量落盘：中断也不丢已完成的配置
    print(f"\n写出 {out_path}（{len(rows)} 行）")


def cmd_report(args):
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, "pps_sweep*.csv"))):
        with open(path, encoding="utf-8-sig") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise SystemExit("没有 pps_sweep*.csv")
    seen, dedup = set(), []
    for r in rows:
        key = (r["case"], r["scheme"], r["trunc"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    rows = dedup

    for r in rows:
        for k in ("n", "depth", "gate_count", "terms_max"):
            r[k] = int(r[k])
        for k in ("delta_circ", "trunc", "R", "maxerr", "mean_err", "R_allzero",
                  "seconds", "R_exact"):
            r[k] = float(r[k])
        r["dead_sites"] = int(r["dead_sites"])
        r["capped"] = str(r["capped"]).lower() == "true"

    # 写 md
    md = ["# PPS（Pauli 路径）攻击与 relative truncation 对照", "",
          f"- 生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}",
          "- 攻击：把 <0|U† Z_i U|0> 反向逐门展开成 Pauli 和，按 Δ 截断后取边缘",
          "- 两种截断：standard 丢 |c|≤Δ；relative 丢 |c|≤‖O_k‖·Δ（论文 §4.4 的贡献）",
          "- 判据：maxerr = max_i |<Z_i>_pps - <Z_i>_exact|；R = 候选串的正确比特比例", "",
          "| n | G | 方案 | Δ | 保留最大项数 | maxerr | mean err | 掐死站点数 | R | 全零猜测 R0 | 用时(s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["gate_count"], r["scheme"], -r["trunc"])):
        md.append(f"| {r['n']} | {r['gate_count']} | {r['scheme']} | {r['trunc']:.0e} "
                  f"| {r['terms_max']} | {r['maxerr']:.2e} | {r['mean_err']:.2e} "
                  f"| {r['dead_sites']} | {r['R']:.3f} | {r['R_allzero']:.3f} "
                  f"| {r['seconds']:.1f} |")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # ---- 同项数预算下的插值对比：在固定 T 处比较两种截断的 maxerr ----
    def interp_err(G, scheme, T):
        ss = sorted([r for r in rows if r["gate_count"] == G and r["scheme"] == scheme
                     and not r["capped"]], key=lambda r: r["terms_max"])
        if len(ss) < 2:
            return None
        xs = np.array([r["terms_max"] for r in ss], dtype=float)
        ys = np.array([r["maxerr"] for r in ss], dtype=float)
        if T < xs[0] or T > xs[-1]:
            return None
        return float(np.exp(np.interp(np.log(T), np.log(xs), np.log(ys))))

    budget_lines = []
    budgets = [1e3, 3e3, 1e4, 3e4]
    for G in sorted({r["gate_count"] for r in rows}):
        row = []
        for T in budgets:
            a = interp_err(G, "standard", T)
            b = interp_err(G, "relative", T)
            row.append((T, a, b))
        budget_lines.append((G, row))

    if any(a is not None and b is not None for _, rr in budget_lines for _, a, b in rr):
        md += ["", "## 相同保留项数下的精度对比（对 log-log 直线插值）", "",
               "| G | 保留项数 T | standard maxerr | relative maxerr | relative 的精度倍数 |",
               "|---|---|---|---|---|"]
        for G, rr in budget_lines:
            for T, a, b in rr:
                if a is None or b is None:
                    continue
                fac = (a / b) if b > 0 else float("nan")
                md.append(f"| {G} | {T:.0e} | {a:.2e} | {b:.2e} | {fac:.2f}× |")
        print("\n=== 同项数预算对比（maxerr；relative 的倍数 = standard/relative）===")
        for G, rr in budget_lines:
            for T, a, b in rr:
                if a is None or b is None:
                    continue
                print(f"  G={G:<4d} T={T:<8.0e} standard={a:.2e} relative={b:.2e} "
                      f"-> {a / b:.2f}×")

    # Pareto 图：maxerr vs 保留项数
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200)
        ax = axes[0]
        for scheme, mk in (("standard", "o"), ("relative", "s")):
            for G in sorted({r["gate_count"] for r in rows}):
                ss = sorted([r for r in rows if r["scheme"] == scheme
                             and r["gate_count"] == G], key=lambda r: r["terms_max"])
                if not ss:
                    continue
                ax.plot([r["terms_max"] for r in ss], [r["maxerr"] for r in ss],
                        marker=mk, ms=4, lw=1, label=f"{scheme} G={G}")
                cap = [r for r in ss if r["capped"]]
                if cap:
                    ax.scatter([r["terms_max"] for r in cap], [r["maxerr"] for r in cap],
                               s=34, facecolors="none", edgecolors="k", linewidths=0.7,
                               zorder=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("retained Pauli terms (max over sites)")
        ax.set_ylabel(r"$\max_i|\langle Z_i\rangle_{\rm pps}-\langle Z_i\rangle_{\rm ex}|$")
        ax.set_title("PPS accuracy vs term budget")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6)

        ax2 = axes[1]
        dref = 3e-2
        for scheme, mk in (("standard", "o"), ("relative", "s")):
            ss = sorted([r for r in rows if r["scheme"] == scheme
                         and abs(r["trunc"] - dref) < 1e-12], key=lambda r: r["gate_count"])
            if not ss:
                continue
            ax2.plot([r["gate_count"] for r in ss], [r["R"] for r in ss],
                     marker=mk, lw=1.2, label=f"{scheme} ($\\Delta$={dref:g})")
            for r in ss:
                ax2.annotate(f"dead {r['dead_sites']}/{r['n']}",
                             xy=(r["gate_count"], r["R"]), xytext=(0, -11),
                             textcoords="offset points", fontsize=6, color="0.35",
                             ha="center")
        r0 = rows[0]["R_allzero"]
        ax2.axhline(r0, ls=":", color="k", lw=1,
                    label=f"all-zero guess R0={r0:.3f} (no information)")
        ax2.set_xlabel("gate count G"); ax2.set_ylabel("R")
        ax2.set_ylim(0, 1.05)
        ax2.set_title(f"PPS at $\\Delta$={dref:g}: collapse to the all-zero guess")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(fontsize=7)
        fig.tight_layout()
        png = os.path.join(OUT, "pps_pareto.png")
        fig.savefig(png)
        plt.close(fig)
        print(f"图: {png}")
    except Exception as e:
        print(f"（绘图失败: {e}）")


def main():
    ap = argparse.ArgumentParser(description="PPS 攻击 + relative truncation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify")
    v.add_argument("--n", type=int, default=8)
    v.add_argument("--depth", type=int, default=2)
    v.add_argument("--seed", type=int, default=3)

    s = sub.add_parser("sweep")
    s.add_argument("--ns", type=str, default="12")
    s.add_argument("--depths", type=str, default=None, help="按门数筛？不填=全部深度")
    s.add_argument("--seed-base", type=int, default=7, help="只跑该种子的实例")
    s.add_argument("--deltas", type=str, default="1e-1,1e-2,1e-3,1e-4")
    s.add_argument("--schemes", type=str, default="standard,relative")
    s.add_argument("--cap", type=int, default=200000)
    s.add_argument("--out", type=str, default="pps_sweep.csv", help="输出 csv 文件名")
    s.add_argument("--mask-seed", type=int, default=2026,
                   help="随机秘密比特串的种子（峰值搬到该串上，避免 0^n 退化）")

    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "verify":
        cmd_verify(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
