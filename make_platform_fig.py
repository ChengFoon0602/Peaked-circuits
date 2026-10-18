# -*- coding: utf-8 -*-
"""make_platform_fig.py — 画平台 mps.cpu 上的 R vs chi（49 比特公开电路）

读 autotest_results/p3_platform_mps/chi_sweep_*.csv，画正确位比例 R(chi)，
并用二项分布给出"纯随机"的 1σ / 2σ 带。chi=32 被免费档拒绝，单独标注。

用法
----
    python make_platform_fig.py                  # 默认 49 比特那条
    python make_platform_fig.py --circuit n12_d8_t015_s0_G88
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "autotest_results", "p3_platform_mps")


def load(name):
    path = os.path.join(OUT_DIR, f"chi_sweep_{name}.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"找不到 {path}")
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            # 反序 R 才是与真值同向的那个（平台串与 QASM 位序相反）
            val = r.get("R_vote_rev") or r.get("R_rev") or ""
            if val in ("", None):
                continue
            rows.append((int(r["chi"]), float(val)))
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", type=str, default="heavy_hex_49x4020")
    ap.add_argument("--n", type=int, default=49)
    ap.add_argument("--blocked", type=str, default="32",
                    help="被资金闸拒绝的 chi，逗号分隔；留空则不标")
    args = ap.parse_args()

    data = load(args.circuit)
    chis = np.array([d[0] for d in data], dtype=float)
    R = np.array([d[1] for d in data])

    # 纯随机猜测的涨落：R = k/n, k ~ Binomial(n, 0.5)
    sig = np.sqrt(0.25 / args.n)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
    ax.axhspan(0.5 - 2 * sig, 0.5 + 2 * sig, color="0.85", zorder=0,
               label=r"random guess $\pm 2\sigma$")
    ax.axhspan(0.5 - sig, 0.5 + sig, color="0.72", zorder=0,
               label=r"$\pm 1\sigma$")
    ax.axhline(0.5, color="0.4", lw=0.9, ls=":", zorder=1)
    ax.plot(chis, R, marker="o", ms=5, lw=1.3, color="#1f77b4",
            label=f"platform mps.cpu ({args.n} qubits)")

    for c, r in zip(chis, R):
        ax.annotate(f"{r:.3f}", (c, r), textcoords="offset points", xytext=(0, 8),
                    fontsize=6.5, ha="center", color="#1f77b4")

    if args.blocked:
        for b in [int(x) for x in args.blocked.split(",") if x.strip()]:
            ax.axvline(b, color="#d62728", lw=1.0, ls="--")
            ax.annotate("blocked:\nneeds $10", (b, 0.72), textcoords="offset points",
                        xytext=(-4, 0), fontsize=6.5, color="#d62728", ha="right")

    ax.set_xlabel(r"bond dimension $\chi$")
    ax.set_ylabel(r"$R$  (correct-bit fraction)")
    ax.set_ylim(0.3, 0.95)
    ax.set_title(rf"{args.circuit.replace('_', ' ')}: no trend in $R(\chi)$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()

    png = os.path.join(OUT_DIR, f"R_vs_chi_{args.circuit}.png")
    fig.savefig(png)
    plt.close(fig)
    print(f"图: {png}")
    print("数据:", dict(zip(chis.astype(int).tolist(), np.round(R, 3).tolist())))


if __name__ == "__main__":
    main()
