# -*- coding: utf-8 -*-
"""exp_scaling_analysis.py — D2 标度律分析：读取多个 autotest 输出目录，
拟合 δ(n) 的指数衰减并与论文参考线（1.189^-n、2^-n）对比。

用法（在仓库根目录）：
    python exp_scaling_analysis.py autotest_results/d2_n8 autotest_results/d2_n10 autotest_results/d2_n12

每个目录需含 autotest 产出的 config.json 与 sweep_summary.csv（每个目录恰好一个 τp 数据点）。
输出（默认写入 autotest_results/d2_scaling/）：
    scaling_plot.png    δ(n) 对数图 + 拟合线 + 两条参考线（锚定在第一个数据点）
    scaling_summary.md  逐点数据、拟合结果与对照结论
"""
import argparse
import csv
import json
import os

import numpy as np


def load_point(d):
    with open(os.path.join(d, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)["configuration"]
    with open(os.path.join(d, "sweep_summary.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise SystemExit(f"{d}: 期望恰好 1 个 τp 数据点，实际 {len(rows)} 个（D2 应为单点目录）")
    depths = cfg.get("depths", [])
    if len(depths) != 1:
        raise SystemExit(f"{d}: 期望 config 中恰好 1 个 τp，实际 {depths}")
    r = rows[0]
    return {
        "n": int(cfg["n"]), "rqc_depth": cfg["rqc_depth"], "pqc_depth": depths[0],
        "mean": float(r["delta_mean"]), "std": float(r["delta_std"]), "runs": int(r["runs"]),
    }


def main():
    ap = argparse.ArgumentParser(description="D2 标度律分析（δ(n) 拟合 + 论文参考线）")
    ap.add_argument("dirs", nargs="+", help="autotest 输出目录列表")
    ap.add_argument("--out", default="autotest_results/d2_scaling", help="输出目录")
    args = ap.parse_args()

    pts = sorted((load_point(d) for d in args.dirs), key=lambda p: p["n"])
    ns = np.array([p["n"] for p in pts], dtype=float)
    means = np.array([p["mean"] for p in pts], dtype=float)
    stds = np.array([p["std"] for p in pts], dtype=float)

    # 拟合 ln δ = a·n + b（δ ∝ e^{a n} = decay^{-n}）
    a, b = np.polyfit(ns, np.log(means), 1)
    resid = np.log(means) - (a * ns + b)
    if len(ns) > 2:
        se = float(np.sqrt((resid ** 2).sum() / (len(ns) - 2) / ((ns - ns.mean()) ** 2).sum()))
    else:
        se = float("nan")
    decay = float(np.exp(-a))

    ref_paper, ref_random = 1.189, 2.0
    print("=== D2 标度律拟合 ===")
    for p in pts:
        print(f"  n={p['n']} (τr={p['rqc_depth']}, τp={p['pqc_depth']}, {p['runs']} 实例): "
              f"δ = {p['mean']:.5f} ± {p['std']:.5f}")
    print(f"拟合: δ ∝ {decay:.4f}^(-n)   (斜率 {a:+.4f}, 标准误 ≈ {se:.4f})")
    print(f"参考: 论文 1.189^(-n) ｜ 随机 2^(-n)")

    os.makedirs(args.out, exist_ok=True)

    # ---- 绘图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=200)
    ax.errorbar(ns, means, yerr=stds, marker="o", capsize=4, color="tab:blue",
                label="trained δ  (τr=n, τp=n/2)")
    nfit = np.linspace(ns.min(), ns.max(), 200)
    anchor = np.log(means[0])
    ax.plot(nfit, np.exp(anchor + np.log(1 / ref_paper) * (nfit - ns[0])), "--",
            lw=1.2, color="gray", label=f"paper ref: {ref_paper}^(-n)")
    ax.plot(nfit, np.exp(anchor + np.log(1 / ref_random) * (nfit - ns[0])), ":",
            lw=1.2, color="tab:red", label="random ref: 2^(-n)")
    ax.plot(nfit, np.exp(b + a * nfit), "-", lw=1, color="tab:blue", alpha=0.7,
            label=f"fit: {decay:.3f}^(-n)")
    ax.set_yscale("log")
    ax.set_xlabel("n (qubits, with τr = n, τp = n/2)")
    ax.set_ylabel("δ (mean ± std, log scale)")
    ax.set_title("Decay of trained peakedness with system size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out, "scaling_plot.png")
    fig.savefig(png)
    plt.close(fig)
    print("图已保存:", png)

    # ---- summary.md ----
    lines = [
        "# D2 标度律拟合摘要",
        "",
        "设置：τr = n、τp = n/2，固定随机种子；各行 n 的实例数见下表。",
        "",
        "| n | 每档实例 | 训练 δ（均值 ± 标准差） |",
        "|---|---|---|",
    ]
    for p in pts:
        lines.append(f"| {p['n']} | {p['runs']} | {p['mean']:.5f} ± {p['std']:.5f} |")
    lines += [
        "",
        f"**拟合**：δ ∝ {decay:.4f}^(-n)（即 ln δ 斜率 {a:+.4f}，标准误 ≈ {se:.4f}）。",
        f"**参考**：论文报告 ≈ 1.189^(-n)（衰减远慢于随机 2^(-n)）。",
        "",
    ]
    if decay < 2.0:
        lines.append(f"**结论**：拟合幂底数 {decay:.3f} **小于**随机标度 2.0 —— "
                     f"与论文「δ 的衰减显著慢于随机电路」的定性结论一致"
                     f"（与 1.189 的差距可按小样本/单起点口径解释）。")
    else:
        lines.append(f"**结论**：拟合幂底数 {decay:.3f} 未明显小于 2.0 —— 与论文预期不符，需检查参数与样本量。")
    with open(os.path.join(args.out, "scaling_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("摘要已保存:", os.path.join(args.out, "scaling_summary.md"))


if __name__ == "__main__":
    main()
