# -*- coding: utf-8 -*-
"""exp_obfuscation.py — E5：混淆定理的数值验证（第二篇复现）

验证 arXiv:2510.00132 §4.2 / Theorem 4.4（"固定 C + 随机起点 + 局部优化
= 退化实现的随机代表"）。固定一个随机砖墙电路 C（tau_r = n 层），用 K 个
独立随机起点训练 C'_k（tau_p = n 层），逐项检查：

  (i)  首列不变：所有 C'_k 的首列都逼近同一个 |c> = C|0>，
       故 delta_k 一致，且两两首列重叠 |<v_i|v_j>|^2 应 ≈ delta^2 + (1-delta)^2/(d-1)；
  (ii) 正交块被随机化：两两 E|Tr(C'_i^dag C'_j)|^2 应与"训练对"(C, C'_k) 同量级（≈2），
       即解之间在 Hilbert-Schmidt 意义下"看起来像相互独立的电路"。

若解间的 |Tr|^2 显著高于训练对，说明优化器只探索了纤维 F_C 的一小部分
（参数化/收敛限制），混淆强度弱于定理的理想情形——这本身是可报告的偏离。

三组对照（重尾分布，样本量见下）：
  A 训练对   (C, C'_k)      —— 共享首列，理论 E|Tr|^2 = 2
  B 解间     (C'_i, C'_j)   —— 若混淆彻底，理论同为 2
  C 随机对照 (C, rand)      —— 无共享首列，理论 E|Tr|^2 = 1（E2 已用 100 实例测到 0.998）

用法（仓库根目录，conda activate peaked）：
    python experiments/paper2/exp_obfuscation.py --quick     # 冒烟（n=6, K=4, 约 20 s）
    python experiments/paper2/exp_obfuscation.py             # 正式（n=8, K=16, 约 7-10 分钟）
输出（默认 autotest_results/e5_obfuscation/）：
    obf_train.csv / obf_pairs.csv / obf_control.csv
    obfuscation_plot.png / summary.md / config.json
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # experiments/<group>/ → 仓库根
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import quimb as qu  # noqa: E402
from circuits import qmps_f, negative_overlap, norm_f  # noqa: E402
from quimb.tensor.optimize import TNOptimizer  # noqa: E402
from exp_hs_overlap import state_from_sites, trace_two  # noqa: E402

SEED0 = 20261016


def train_one(n, rqc, seed, max_iter, loss_target):
    """固定目标 C（rqc），用一个随机起点训练 C'。返回 (TN, 训练报告的 delta)。"""
    qu.seed_rand(seed)
    pqc = qmps_f(n, in_depth=n, n_Qbit=n - 1, qmps_structure="brickwall")
    opt = TNOptimizer(pqc, loss_fn=negative_overlap, norm_fn=norm_f, loss_target=loss_target,
                      constant_tags=["MPS"], loss_constants={"target": rqc},
                      autodiff_backend="autograd", optimizer="L-BFGS-B")
    pqc_opt = opt.optimize(max_iter, tol=1e-10)
    return pqc_opt, float(-opt.loss_best)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="E5：Theorem 4.4 混淆定理的数值验证")
    ap.add_argument("--quick", action="store_true", help="冒烟（n=6, K=4, max_iter=200）")
    ap.add_argument("--n", type=int, default=None, help="比特数（默认 8）")
    ap.add_argument("--seeds", type=int, default=None, help="起点数 K（默认 20）")
    ap.add_argument("--max-iter", type=int, default=None, help="训练上限（默认 500）")
    ap.add_argument("--outdir", default="autotest_results/e5_obfuscation")
    args = ap.parse_args()

    if args.quick:
        n, K, max_iter = 6, 4, 200
    else:
        n, K, max_iter = args.n or 8, args.seeds or 20, args.max_iter or 500
    loss_target = -1 + 1e-2
    d = 2 ** n

    os.makedirs(args.outdir, exist_ok=True)
    print(f"=== E5 混淆定理验证（n={n}, K={K}, max_iter={max_iter}）===", flush=True)

    # ---- 固定的目标电路 C ----
    qu.seed_rand(SEED0)
    rqc = qmps_f(n, in_depth=n, n_Qbit=n - 1, qmps_structure="brickwall")
    u = state_from_sites(rqc, n, 0)          # |c> = C|0>

    # ---- K 个独立起点训练 ----
    sols = []
    rows_train = []
    t0 = time.time()
    for k in range(K):
        seed = SEED0 + 7919 * (k + 1)
        tk = time.time()
        pqc_opt, delta_rep = train_one(n, rqc, seed, max_iter, loss_target)
        v = state_from_sites(pqc_opt, n, 0)
        col2 = float(abs(np.vdot(u, v)) ** 2)
        tr2 = float(abs(trace_two(rqc, pqc_opt, n)) ** 2)
        ok = abs(col2 - delta_rep) < 1e-6
        sols.append({"tn": pqc_opt, "v": v})
        rows_train.append({"k": k, "seed": seed,
                           "delta_reported": round(delta_rep, 8),
                           "col2_vs_target": round(col2, 8),
                           "tr2_vs_target": round(tr2, 6),
                           "secs": round(time.time() - tk, 1),
                           "selfcheck_ok": ok})
        print(f"  [解 {k+1}/{K}] seed={seed}  δ={delta_rep:.4f}  "
              f"首列重叠={col2:.4f}  |Tr|²(vs C)={tr2:.3f}  ({time.time()-tk:.0f}s)", flush=True)
    train_s = time.time() - t0

    # ---- 解间两两 HS 重叠 ----
    print("\n  计算解间两两重叠 ...", flush=True)
    rows_pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            tr2 = float(abs(trace_two(sols[i]["tn"], sols[j]["tn"], n)) ** 2)
            c2 = float(abs(np.vdot(sols[i]["v"], sols[j]["v"])) ** 2)
            rows_pairs.append({"i": i, "j": j, "tr2": round(tr2, 6),
                               "col2": round(c2, 8)})

    # ---- 随机对照：未训练的随机电路 vs C ----
    print("  计算随机对照 ...", flush=True)
    rows_ctrl = []
    for k in range(K):
        qu.seed_rand(SEED0 + 100003 + k)
        rnd = qmps_f(n, in_depth=n, n_Qbit=n - 1, qmps_structure="brickwall")
        rows_ctrl.append({"k": k,
                          "tr2": round(float(abs(trace_two(rqc, rnd, n)) ** 2), 6)})

    write_csv(os.path.join(args.outdir, "obf_train.csv"), rows_train)
    write_csv(os.path.join(args.outdir, "obf_pairs.csv"), rows_pairs)
    write_csv(os.path.join(args.outdir, "obf_control.csv"), rows_ctrl)

    # ---- 统计 ----
    def stat(vals):
        a = np.asarray(vals, dtype=float)
        return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0

    A = [r["tr2_vs_target"] for r in rows_train]
    B = [r["tr2"] for r in rows_pairs]
    C = [r["tr2"] for r in rows_ctrl]
    dl = [r["delta_reported"] for r in rows_train]
    colB = [r["col2"] for r in rows_pairs]

    m_a, s_a = stat(A)
    m_b, s_b = stat(B)
    m_c, s_c = stat(C)
    m_dl, s_dl = stat(dl)
    dbar = float(np.mean(dl))
    col_pred = dbar ** 2 + (1.0 - dbar) ** 2 / (d - 1.0)
    m_colB, s_colB = stat(colB)

    selfcheck_fail = [r for r in rows_train if not r["selfcheck_ok"]]
    z_vs_2 = (m_b - 2.0) / s_b if s_b > 0 else 0.0
    z_ab = (m_b - m_a) / np.hypot(s_a, s_b) if (s_a + s_b) > 0 else 0.0

    # bootstrap 95% CI（重尾下比 SEM 更稳健）
    def boot_ci(vals, n_boot=20000, seed=7):
        a = np.asarray(vals, dtype=float)
        if len(a) < 2:
            return (float(a.mean()), float(a.mean()))
        rg = np.random.default_rng(seed)
        mm = rg.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
        lo, hi = np.percentile(mm, [2.5, 97.5])
        return (float(lo), float(hi))

    ci_a, ci_b, ci_c = boot_ci(A), boot_ci(B), boot_ci(C)

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=200)

    ax = axes[0]
    rng = np.random.default_rng(0)
    groups = [("A  trained pair\n$(C,\\,C'_k)$", A, "tab:blue"),
              ("B  inter-solution\n$(C'_i,\\,C'_j)$", B, "tab:purple"),
              ("C  random control\n$(C,\\,\\mathrm{rand})$", C, "tab:gray")]
    for idx, (name, vals, color) in enumerate(groups):
        x = idx + rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(x, vals, s=13, alpha=0.55, color=color, edgecolors="none")
        m = np.mean(vals)
        lo, hi = [ci_a, ci_b, ci_c][idx]
        ax.errorbar([idx], [m], yerr=[[m - lo], [hi - m]], fmt="none",
                    color=color, capsize=6, lw=1.8, zorder=6)
        ax.plot([idx - 0.28, idx + 0.28], [m, m], color=color, lw=2.2, zorder=5)
    ax.axhline(2.0, color="tab:red", ls="--", lw=1, label="theory 2 (shared column)")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="theory 1 (independent)")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    ax.set_ylabel(r"$|\mathrm{Tr}(C_a^\dagger C_b)|^2$")
    ax.set_title(f"(a) Overlap groups  (n={n}, d={d})", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    ax.scatter(np.arange(K), dl, s=22, color="tab:blue", label=r"$\delta_k$ (reported)")
    ax.scatter(np.arange(K), [r["col2_vs_target"] for r in rows_train], s=16,
               color="tab:orange", marker="x", label=r"$|\langle c|v_k\rangle|^2$ (measured)")
    ax.axhline(dbar, color="tab:blue", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("solution index  $k$")
    ax.set_ylabel(r"peak weight  $\delta$")
    ax.set_title(f"(b) First-column invariance  (mean $\\delta$={dbar:.4f}, sd={np.std(dl, ddof=1):.4f})",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    png = os.path.join(args.outdir, "obfuscation_plot.png")
    fig.savefig(png)
    plt.close(fig)
    print(f"\n图已保存: {png}")

    # ---- summary.md ----
    verdict = (
        f"**完全符合定理**：解间重叠与训练对无显著差异（{z_ab:+.1f}σ），且与理论值 2 一致（{z_vs_2:+.1f}σ）。"
    ) if (abs(z_ab) < 2.5 and abs(z_vs_2) < 2.0) else (
        f"**基本符合，但轻微偏高**：解间重叠与训练对同量级（{z_ab:+.1f}σ），却相对理论 2 偏离 {z_vs_2:+.1f}σ"
        f"（bootstrap 95% CI 下界 {ci_b[0]:.3f}）——提示正交块之间残留约 {100 * (m_b / 2.0 - 1.0):.0f}% 的额外相关；"
        f"重尾分布下该偏离的稳健性有限，需更大 K 确认。"
    ) if abs(z_ab) < 2.5 else (
        f"**偏离**：解间重叠显著高于训练对（{z_ab:+.1f}σ），说明优化器只探索了纤维的一小部分，"
        f"混淆强度弱于定理的理想情形——这是可报告的偏离。"
    )

    lines = [
        "# E5 摘要 · 混淆定理的数值验证（Theorem 4.4）",
        "",
        f"设置：$n={n}$（$d={d}$），$\\tau_r=\\tau_p=n$，固定一个目标电路 $C$，"
        f"用 **{K} 个独立随机起点**训练 $C'_k$（max_iter={max_iter}，L-BFGS-B）；"
        f"训练总耗时 {train_s:.0f} s。",
        f"自检：每例的报告 $\\delta$ 与实测首列重叠一致（失败 {len(selfcheck_fail)} 例）。",
        "",
        "## (i) 首列不变性",
        "",
        "| 量 | 实测 | 说明 |",
        "|---|---|---|",
        f"| $\\delta$ 均值 ± sd | {m_dl:.4f} ± {np.std(dl, ddof=1):.4f} | 各解峰值权重几乎相同 |",
        f"| 首列两两重叠 $\\|\\langle v_i\\|v_j\\rangle\\|^2$ | {m_colB:.4f} ± {s_colB:.4f} "
        f"| 预测 $\\delta^2+(1-\\delta)^2/(d-1)={col_pred:.4f}$ |",
        "",
        "## (ii) 正交块的随机化（Hilbert–Schmidt 重叠）",
        "",
        "| 组 | 关系 | 样本数 | $\\mathbb{E}\\|\\mathrm{Tr}\\|^2$ | bootstrap 95% CI | 理论 |",
        "|---|---|---|---|---|---|",
        f"| A 训练对 | $(C,\\,C'_k)$ | {len(A)} | **{m_a:.3f}** | [{ci_a[0]:.2f}, {ci_a[1]:.2f}] | 2 |",
        f"| B 解间 | $(C'_i,\\,C'_j)$ | {len(B)} | **{m_b:.3f}** | [{ci_b[0]:.2f}, {ci_b[1]:.2f}] | 2（若混淆彻底） |",
        f"| C 随机对照 | $(C,\\,\\mathrm{{rand}})$ | {len(C)} | **{m_c:.3f}** | [{ci_c[0]:.2f}, {ci_c[1]:.2f}] | 1 |",
        "",
        f"- B 相对理论值 2 的偏离：**{z_vs_2:+.1f}σ**（CI {'含' if ci_b[0] <= 2.0 <= ci_b[1] else '不含'} 2）；"
        f"B 与 A 的差异：**{z_ab:+.1f}σ**；",
        f"- 比值 B/C = **{m_b / m_c:.2f}**，A/C = **{m_a / m_c:.2f}**（若混淆彻底两者应同量级，理论均为 2）；",
        f"- 中位数：A {np.median(A):.2f} / B {np.median(B):.2f} / C {np.median(C):.2f}"
        f"（$-\\ln$ 尾的重尾分布，均值 > 中位数符合独立对照的 $\\mathrm{{Exp}}(1)$ 特征）。",
        "",
        f"**判定**：{verdict}",
        "",
        "## 读法",
        "",
        "- Theorem 4.4 的实质是「损失只依赖首列 ⇒ 所有最优解构成一个高维纤维 $F_C$，"
        "随机起点把结果撒在纤维上」。首列不变性验证了前半句，解间 HS 重叠验证了后半句；",
        "- 由于 $|\\mathrm{Tr}|^2$ 是重尾统计量（独立对照近似 $\\mathrm{Exp}(1)$），"
        f"B 组 {len(B)} 个样本给出的 SEM 约 {s_b:.2f}——判别 2 与 2.5 以上的偏离需要此量级以上的样本；",
        "- 该实验的可复现性完全由种子决定（目标电路 seed = %d，第 $k$ 个起点 seed = %d + 7919(k+1)）。"
        % (SEED0, SEED0),
    ]
    with open(os.path.join(args.outdir, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    cfg = {
        "experiment": "E5 obfuscation (Theorem 4.4)",
        "seed_target": SEED0,
        "seed_rule": "SEED0 + 7919*(k+1)",
        "n": n, "K": K, "max_iter": max_iter,
        "mean_A_trained": round(m_a, 6), "mean_B_inter": round(m_b, 6),
        "mean_C_random": round(m_c, 6),
        "z_B_vs_2": round(float(z_vs_2), 3), "z_B_vs_A": round(float(z_ab), 3),
        "ci_A": [round(x, 3) for x in ci_a], "ci_B": [round(x, 3) for x in ci_b],
        "ci_C": [round(x, 3) for x in ci_c],
        "train_s": round(train_s, 1),
    }
    with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"\nA 训练对 = {m_a:.3f} ± {s_a:.3f}  (理论 2)")
    print(f"B 解间   = {m_b:.3f} ± {s_b:.3f}  (理论 2，偏离 {z_vs_2:+.1f}σ)")
    print(f"C 随机   = {m_c:.3f} ± {s_c:.3f}  (理论 1)")
    print(f"δ = {m_dl:.4f} ± {np.std(dl, ddof=1):.4f} | 首列两两重叠 = {m_colB:.4f} (预测 {col_pred:.4f})")
    print(f"输出目录: {args.outdir}")


if __name__ == "__main__":
    main()
