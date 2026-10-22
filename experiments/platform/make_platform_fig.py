# -*- coding: utf-8 -*-
"""make_platform_fig.py — 画平台 mps.cpu 上的 R(chi) 与 |<Z>| 随 chi 的变化

左：正确位比例 R(chi)，用二项分布给出"纯随机"的 1σ/2σ 带。
右：|<Z_i>| 的均值与最大值随 chi 的变化——这一栏才是"收敛"的证据。

数据来源
--------
    chi_sweep_<name>.csv              每档的 R（正序/反序）
    observables_<name>_chi<chi>.csv   每档 49 个 <Z_i> 原值

用法
----
    python experiments/platform/make_platform_fig.py
    python experiments/platform/make_platform_fig.py --circuit n12_d8_t015_s0_G88 --n 12
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # experiments/<group>/ → 仓库根
OUT_DIR = os.path.join(ROOT, "autotest_results", "p3_platform_mps")


def load_R(name):
    path = os.path.join(OUT_DIR, f"chi_sweep_{name}.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"找不到 {path}")
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            # 反序 R 才是与真值同向的那个（可观测量下标 i 对应 QASM 比特 n-1-i）
            val = r.get("R_vote_rev") or r.get("R_rev") or ""
            if val not in ("", None):
                out[int(r["chi"])] = float(val)
    return out


def load_absZ(name):
    out = {}
    for f in glob.glob(os.path.join(OUT_DIR, f"observables_{name}_chi*.csv")):
        m = re.search(r"chi(\d+)\.csv$", f)
        if not m:
            continue
        z = [float(r["expectation_Z"]) for r in
             csv.DictReader(open(f, encoding="utf-8-sig"))]
        if z:
            out[int(m.group(1))] = (float(np.mean(np.abs(z))),
                                    float(np.max(np.abs(z))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", type=str, default="heavy_hex_49x4020")
    ap.add_argument("--n", type=int, default=49)
    ap.add_argument("--blocked", type=str, default="32")
    args = ap.parse_args()

    R = load_R(args.circuit)
    A = load_absZ(args.circuit)
    chis = np.array(sorted(R), dtype=float)
    Rv = np.array([R[int(c)] for c in chis])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = np.sqrt(0.25 / args.n)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), dpi=200)

    ax = axes[0]
    ax.axhspan(0.5 - 2 * sig, 0.5 + 2 * sig, color="0.88", zorder=0,
               label=r"random $\pm2\sigma$")
    ax.axhspan(0.5 - sig, 0.5 + sig, color="0.74", zorder=0, label=r"$\pm1\sigma$")
    ax.axhline(0.5, color="0.4", lw=0.9, ls=":", zorder=1)
    ax.plot(chis, Rv, marker="o", ms=4.5, lw=1.2, color="#1f77b4",
            label=f"mps.cpu, {args.n} qubits")
    for c, r in zip(chis, Rv):
        ax.annotate(f"{r:.2f}", (c, r), textcoords="offset points", xytext=(0, 7),
                    fontsize=6, ha="center", color="#1f77b4")
    if args.blocked:
        for b in [int(x) for x in args.blocked.split(",") if x.strip()]:
            ax.axvline(b, color="#d62728", lw=1.0, ls="--")
            ax.annotate("blocked\n(needs $10)", (b, 0.80), fontsize=6,
                        color="#d62728", ha="right", va="top")
    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel(r"$R$ (correct-bit fraction)")
    ax.set_ylim(0.35, 0.85)
    ax.set_title(r"$R$ vs $\chi$: no trend, all within the random band")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6.5, loc="lower left")

    ax2 = axes[1]
    ca = np.array(sorted(A), dtype=float)
    mean = [A[int(c)][0] for c in ca]
    mx = [A[int(c)][1] for c in ca]
    ax2.plot(ca, mean, marker="s", ms=4.5, lw=1.2, color="#2ca02c",
             label=r"mean $|\langle Z_i\rangle|$")
    ax2.plot(ca, mx, marker="^", ms=4.5, lw=1.0, color="#ff7f0e", alpha=0.85,
             label=r"max $|\langle Z_i\rangle|$")
    ax2.set_xlabel(r"bond dimension $\chi$")
    ax2.set_ylabel(r"$|\langle Z_i\rangle|$")
    ax2.set_title(r"$|\langle Z_i\rangle|$ shrinks with $\chi$; direction stays random")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=6.5, loc="upper right")

    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"R_vs_chi_{args.circuit}.png")
    fig.savefig(png)
    plt.close(fig)
    print(f"图: {png}")
    print(f"R: {dict(zip(chis.astype(int).tolist(), np.round(Rv, 3).tolist()))}")
    print(f"mean|Z|: {dict(zip(ca.astype(int).tolist(), np.round(mean, 3).tolist()))}")


if __name__ == "__main__":
    main()
