# -*- coding: utf-8 -*-
"""exp_tns_bp.py — 第三篇 §4 攻击 #3 的简化复现：PEPS（TNS）+ Belief Propagation

论文做法：全连接电路转译到简化拓扑（三棵二叉树）→ **simple update** 演化 TNS →
**BP** 做规范化并算单比特 ⟨Z_q⟩。

本文件复现其骨架，但拓扑取**有环的 2D 网格**。理由：本仓库的 peaked 电路是 1D 砖墙，
映射到树上没有环，BP ≡ 精确收缩（测不出东西）；只有"**打乱序的电路 + 2D 网格**"这一组合
（即论文 §4.5 的 swap 混淆之后）才让 BP 的近似误差有意义。

三种边缘一起量，误差才能拆开：

| 量 | 参照 | 衡量什么 |
|---|---|---|
| ⟨Z_q⟩_true | 精确态矢量 | — |
| ⟨Z_q⟩_peps | 精确收缩同一 PEPS | **ansatz 误差**（χ 不够） |
| ⟨Z_q⟩_bp | BP 收缩 | **ansatz + BP 近似误差** |

用法
----
    python exp_tns_bp.py verify                  # n=6, χ=χ_sat：必须与精确态矢量一致
    python exp_tns_bp.py sweep --case n12_d8     # n=12, 6×2 网格, 扫 χ
    python exp_tns_bp.py report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque

import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor import belief_propagation as bp

from exp_mps_break import (bitstring_from_marginals, gate_pattern, load_circuits,
                           norm_gate, relabel, score_R, sv_gates_to_state,
                           sv_marginals)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "autotest_results", "p3_tns_bp")

SWAP4 = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=complex)


# ============================================================================
# 网格
# ============================================================================
def grid_shape(n):
    assert n % 2 == 0, "本实现要求偶数比特"
    return n // 2, 2


def all_sites(Lx, Ly):
    return [(i, j) for i in range(Lx) for j in range(Ly)]


def grid_path(a, b, Lx, Ly):
    """网格上 a→b 的最短路径（BFS），返回含两端的站点列表。"""
    prev = {a: None}
    dq = deque([a])
    while dq:
        cur = dq.popleft()
        if cur == b:
            break
        i, j = cur
        for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= nb[0] < Lx and 0 <= nb[1] < Ly and nb not in prev:
                prev[nb] = cur
                dq.append(nb)
    path, cur = [], b
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


# ============================================================================
# simple update（含交换路由）
# ============================================================================
def evolve(n, gates, chi):
    """在 Lx×2 网格上用 simple update 跑完整电路；长程门靠交换链把一端挪到另一端旁边。

    返回 (peps, site2log, stats)。site2log[(i,j)] = 该站点当前承载的逻辑比特。
    """
    Lx, Ly = grid_shape(n)
    zero = np.array([1.0, 0.0])
    psi0 = qtn.PEPS.product_state([[zero.copy() for _ in range(Ly)] for _ in range(Lx)])
    # product_state 的键维是显式的 1；不撑开的话缠结门无法把键长出来（CNOT 会把叠加态塌掉）
    psi0.expand_bond_dimension_(max(2, chi))
    su = qtn.SimpleUpdate(psi0, ham=None, chi=chi, progbar=False)

    sites = all_sites(Lx, Ly)
    site2log = {s: k for k, s in enumerate(sites)}
    pos = {k: s for s, k in site2log.items()}
    n_swap = 0
    t0 = time.time()

    for a, b, u in gates:
        sa, sb = pos[a], pos[b]
        path = grid_path(sa, sb, Lx, Ly)
        # 把 b 沿路径一步步挪到 sa 旁边：k 从末端往前
        for k in range(len(path) - 1, 1, -1):
            x, y = path[k - 1], path[k]
            su.gate(SWAP4, (x, y))
            lx, ly = site2log[x], site2log[y]
            site2log[x], site2log[y] = ly, lx
            pos[lx], pos[ly] = y, x
            n_swap += 1
        nb = path[1]
        su.gate(np.asarray(u, dtype=complex), (sa, nb))

    return su.get_state(absorb_gauges=True), site2log, {
        "seconds": time.time() - t0, "n_swap": n_swap}


# ============================================================================
# 边缘
# ============================================================================
def dense_logical(peps, n, site2log):
    """PEPS 精确收缩成态矢量，并按逻辑比特序排轴。"""
    Lx, Ly = grid_shape(n)
    v = np.asarray(peps.to_dense()).reshape((2,) * n)
    axes = [None] * n
    for k, s in enumerate(all_sites(Lx, Ly)):
        axes[site2log[s]] = k
    return np.transpose(v, axes).reshape(-1)


def marginals_bp(peps, n, site2log):
    """每个站点的 ⟨Z_q⟩ 用 BP 收缩；分母是同一张网络的纯 norm 收缩。"""
    Lx, Ly = grid_shape(n)
    ket = peps.copy()
    bra = peps.copy().H
    for (i, j) in all_sites(Lx, Ly):
        bra.reindex_({f"k{i},{j}": f"kb{i},{j}"})
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def build(site_idx):
        tn = ket.copy() & bra.copy()
        for q, (i, j) in enumerate(all_sites(Lx, Ly)):
            m = Z if q == site_idx else I2
            tn &= qtn.Tensor(m.copy(), inds=(f"k{i},{j}", f"kb{i},{j}"))
        return tn

    norm = bp.contract_d1bp(build(-1))
    z = {}
    for k, s in enumerate(all_sites(Lx, Ly)):
        z[site2log[s]] = float(np.real(bp.contract_d1bp(build(k)) / norm))
    return np.array([z[q] for q in range(n)])


# ============================================================================
# verify：小网格上把整条链路钉死
# ============================================================================
def cmd_verify(args):
    n, depth = args.n, args.depth
    Lx, Ly = grid_shape(n)
    chi_sat = 2 ** max(Lx, Ly)         # 横切要 2^Lx、竖切要 2^Ly，取大者
    qu.seed_rand(args.seed)
    pairs = gate_pattern(n, depth)
    gates = [(i, j, np.asarray(qu.rand_uni(4, dtype=complex))) for i, j in pairs]
    perm = list(range(n))
    np.random.default_rng(args.perm_seed).shuffle(perm)
    gs = relabel(gates, perm)

    psi = sv_gates_to_state(n, gs)
    z_true = sv_marginals(psi)
    print(f"n={n} depth={depth} G={len(gs)} 网格 {Lx}×{Ly}  χ_sat={chi_sat}")

    for chi in (chi_sat, max(2, chi_sat // 2), 2):
        peps, s2l, st = evolve(n, gs, chi)
        v = dense_logical(peps, n, s2l)
        z_peps = sv_marginals(v.reshape((2,) * n))
        z_bp = marginals_bp(peps, n, s2l)
        e_peps = float(np.max(np.abs(z_peps - z_true)))
        e_bp = float(np.max(np.abs(z_bp - z_true)))
        print(f"  χ={chi:3d}  swaps={st['n_swap']:4d}  用时={st['seconds']:5.1f}s  "
              f"maxerr(PEPS)={e_peps:.2e}  maxerr(BP)={e_bp:.2e}  "
              f"{'✓' if max(e_peps, e_bp) < 1e-9 else '（χ 不足，正常）'}")


# ============================================================================
# sweep：n=12 上扫 χ
# ============================================================================
def cmd_sweep(args):
    circs = [c for c in load_circuits() if c["_file"].startswith(args.case)]
    if not circs:
        raise SystemExit(f"没找到电路 {args.case}")
    rec = circs[0]
    n, gates = rec["n"], rec["_gates"]
    Lx, Ly = grid_shape(n)

    rng = np.random.default_rng(args.mask_seed)
    bits = [int(b) for b in rng.integers(0, 2, n)]
    target = "".join(str(b) for b in bits)
    from exp_pps import x_mask_gates
    gates = gates + x_mask_gates(n, bits)

    perm = list(range(n))
    np.random.default_rng(args.perm_seed).shuffle(perm)
    gs = relabel(gates, perm)

    psi = sv_gates_to_state(n, gs)
    z_true = sv_marginals(psi)
    print(f"{rec['_file']}  G={len(gs)} δ={rec['delta']:.4f} secret={target} "
          f"网格 {Lx}×{Ly}")

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for chi in [int(x) for x in args.chis.split(",")]:
        peps, s2l, st = evolve(n, gs, chi)
        v = dense_logical(peps, n, s2l)
        z_peps = sv_marginals(v.reshape((2,) * n))
        z_bp = marginals_bp(peps, n, s2l)
        row = dict(case=rec["_file"], n=n, G=len(gs), chi=chi,
                   seconds=st["seconds"], n_swap=st["n_swap"],
                   err_peps=float(np.max(np.abs(z_peps - z_true))),
                   err_bp=float(np.max(np.abs(z_bp - z_true))),
                   err_bp_vs_peps=float(np.max(np.abs(z_bp - z_peps))),
                   R_peps=score_R(target, bitstring_from_marginals(z_peps)),
                   R_bp=score_R(target, bitstring_from_marginals(z_bp)),
                   dead_bp=int(np.sum(np.abs(z_bp) < 1e-12)))
        rows.append(row)
        print(f"  χ={chi:3d} swaps={st['n_swap']:5d} t={st['seconds']:6.1f}s  "
              f"err(PEPS)={row['err_peps']:.3e} err(BP)={row['err_bp']:.3e} "
              f"BP-vs-PEPS={row['err_bp_vs_peps']:.3e}  R(PEPS)={row['R_peps']:.3f} "
              f"R(BP)={row['R_bp']:.3f}", flush=True)

    path = os.path.join(OUT, "tns_bp_sweep.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n写出 {path}")


def cmd_report(args):
    rows = []
    for p in sorted([x for x in os.listdir(OUT) if x.startswith("tns_bp_sweep")]):
        with open(os.path.join(OUT, p), encoding="utf-8-sig") as f:
            rows.extend(csv.DictReader(f))
    for r in rows:
        for k in ("n", "G", "chi", "n_swap", "dead_bp"):
            r[k] = int(r[k])
        for k in ("seconds", "err_peps", "err_bp", "err_bp_vs_peps", "R_peps", "R_bp"):
            r[k] = float(r[k])
    md = ["# PEPS（TNS）+ BP 简化复现", "",
          "| 电路 | G | χ | 交换次数 | 用时(s) | err(PEPS vs 精确) | err(BP vs 精确) | BP 附加误差 | R(PEPS) | R(BP) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["chi"]):
        md.append(f"| {r['case']} | {r['G']} | {r['chi']} | {r['n_swap']} | {r['seconds']:.1f} "
                  f"| {r['err_peps']:.2e} | {r['err_bp']:.2e} | {r['err_bp_vs_peps']:.2e} "
                  f"| {r['R_peps']:.3f} | {r['R_bp']:.3f} |")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
        rs = sorted(rows, key=lambda r: r["chi"])
        ch = [r["chi"] for r in rs]
        ax.plot(ch, [r["err_peps"] for r in rs], "o-", lw=1.2, label="PEPS vs exact state")
        ax.plot(ch, [r["err_bp"] for r in rs], "s--", lw=1.2, label="BP vs exact state")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"bond dimension $\chi$")
        ax.set_ylabel(r"$\max_q|\langle Z_q\rangle-\langle Z_q\rangle_{\rm true}|$")
        ax.set_title("PEPS on a 6x2 grid + BP: error decomposition")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(OUT, "tns_bp_error.png")
        fig.savefig(png)
        plt.close(fig)
        print(f"图: {png}")
    except Exception as e:
        print(f"（绘图失败: {e}）")


def main():
    ap = argparse.ArgumentParser(description="TNS(PEPS) + BP 简化复现")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify")
    v.add_argument("--n", type=int, default=6)
    v.add_argument("--depth", type=int, default=3)
    v.add_argument("--seed", type=int, default=4)
    v.add_argument("--perm-seed", type=int, default=11)

    s = sub.add_parser("sweep")
    s.add_argument("--case", type=str, default="n12_d8_t015_s0")
    s.add_argument("--chis", type=str, default="1,2,4,8,16")
    s.add_argument("--perm-seed", type=int, default=11)
    s.add_argument("--mask-seed", type=int, default=2026)

    sub.add_parser("report")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.cmd == "verify":
        cmd_verify(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
