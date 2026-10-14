# -*- coding: utf-8 -*-
"""exp_noise_robustness.py — E1：Peaked 电路输出的噪声鲁棒解码（第二篇复现·主实验）

复现 arXiv:2510.00132 §4.4–4.5 的噪声模型与解码算法，并对照论文公式核对标度：

  子实验 A（HBA 偏差界）：命题 4.6 —— p_max ≤ E[p̂(t)] ≤ p_max + b(|B_t|-1)
  子实验 B（BSC + 多数表决）：命题 4.11 —— 所需样本 N ∝ (1-2r)^-2 · p_max^-2 · log(n/η)
  子实验 C（t-稀疏 + Hamming-中心）：命题 4.10 —— 所需样本 N ∝ |B_2t| · p_max^-2 · log(n/η)
  子实验 D（全局去极化 + 去偏估计）：§4.5 —— SE ≤ 1/(2(1-ε)√N)

分布模型：p(s*) = p_max，其余 ≈ 均匀背景 b = (1-p_max)/(2^n - 1)
（与论文 Remark 4.7 的分析模型一致；峰值取全 0 串 WLOG）。
噪声实现：t-稀疏为"随机 t 位翻转"（论文的界是对抗式最坏情况；随机翻转是对平均情形的探测，见 summary 口径说明）。

用法（仓库根目录，conda activate peaked）：
    python exp_noise_robustness.py --quick    # 约 1 分钟
    python exp_noise_robustness.py            # 约 5-10 分钟
输出（默认 autotest_results/e1_noise/）：
    hba_bias.csv / bsc_majority.csv / tsparse_center.csv / depolarizing.csv
    e1_noise_panels.png（2×2 面板图）
    summary.md / config.json
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np

SEED = 20261014


# ---------------- 分布与噪声 ----------------

def sample_peaked(n, p_max, N, rng):
    """从两层分布采 N 个 n-bit 样本（s* = 全 0）。返回 (N, n) uint8。"""
    d = 1 << n
    b = (1.0 - p_max) / (d - 1)
    probs = np.full(d, b)
    probs[0] = p_max
    idx = rng.choice(d, size=N, p=probs)
    bits = ((idx[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.uint8)
    return bits


def add_tflips(bits, t, rng):
    """随机 t-稀疏比特翻转：每 shot 随机翻转恰好 t 位。"""
    if t <= 0:
        return bits.copy()
    N, n = bits.shape
    out = bits.copy()
    idx = np.argsort(rng.random((N, n)), axis=1)[:, :t]
    rows = np.repeat(np.arange(N), t)
    out[rows, idx.ravel()] ^= 1
    return out


def add_bsc(bits, r, rng):
    """i.i.d. 比特翻转 BSC(r)。"""
    if r <= 0:
        return bits.copy()
    return (bits ^ (rng.random(bits.shape) < r)).astype(np.uint8)


def add_depol(bits, eps, rng):
    """全局去极化：以概率 eps 将样本替换为完全随机样本。"""
    if eps <= 0:
        return bits.copy()
    N = bits.shape[0]
    mask = rng.random(N) < eps
    out = bits.copy()
    if mask.any():
        rand = rng.integers(0, 2, size=bits.shape, dtype=np.uint8)
        out[mask] = rand[mask]
    return out


# ---------------- 解码算法 ----------------

def hba_estimate(bits, t):
    """Hamming 球聚合（算法 2）：到 s*（全 0）距离 ≤ t 的样本占比。"""
    return float((bits.sum(axis=1) <= t).mean())


def hamming_center_decode(bits, t, max_n=1024):
    """Hamming-中心解码（算法 3）：聚类于 2t 球、球内逐位多数表决。"""
    N = bits.shape[0]
    use = bits[:min(N, max_n)]
    D = (use[:, None, :] != use[None, :, :]).sum(axis=2)   # (N,N) 两两距离
    within = D <= 2 * t
    ci = within.sum(axis=1)
    i_star = int(np.argmax(ci))
    core = use[within[i_star]]
    return (core.mean(axis=0) >= 0.5).astype(np.uint8)


def majority_decode(bits):
    """逐比特多数表决。"""
    return (bits.mean(axis=0) >= 0.5).astype(np.uint8)


def is_success(s_hat, n):
    return bool(np.array_equal(s_hat, np.zeros(n, dtype=np.uint8)))


def hamming_ball_size(n, t):
    return int(sum(math.comb(n, h) for h in range(0, min(t, n) + 1)))


def n90(Ns, rates):
    """从 (N, recovery) 网格估计 90% 恢复所需 N（log 线性插值）。"""
    Ns = list(Ns); rates = list(rates)
    if rates[0] >= 0.9:
        return float(Ns[0])
    for i in range(1, len(Ns)):
        if rates[i] >= 0.9 > rates[i - 1]:
            f = (0.9 - rates[i - 1]) / (rates[i] - rates[i - 1])
            return float(10 ** (np.log10(Ns[i - 1]) + f * (np.log10(Ns[i]) - np.log10(Ns[i - 1]))))
    return None


# ---------------- 四个子实验 ----------------

def exp_a_hba(n, p_max, t_list, M, N, rng):
    b = (1.0 - p_max) / ((1 << n) - 1)
    rows = []
    for t in t_list:
        vals = []
        for _ in range(M):
            bits = sample_peaked(n, p_max, N, rng)
            vals.append(hba_estimate(bits, t))
        vals = np.array(vals)
        rows.append({"n": n, "p_max": p_max, "t": t,
                     "ball": hamming_ball_size(n, t),
                     "mean": float(vals.mean()), "sem": float(vals.std(ddof=1) / np.sqrt(M)),
                     "bound_lo": p_max,
                     "bound_hi": p_max + b * (hamming_ball_size(n, t) - 1)})
        print(f"  [A] t={t:>2} |B_t|={rows[-1]['ball']:>6}  E[p̂]={rows[-1]['mean']:.5f}"
              f"  界=[{p_max:.4f}, {rows[-1]['bound_hi']:.5f}]", flush=True)
    return rows


def exp_b_majority(n, p_max, r_list, Ns, trials, rng):
    rows = []
    for r in r_list:
        for N in Ns:
            ok = 0
            for _ in range(trials):
                bits = sample_peaked(n, p_max, N, rng)
                bits = add_bsc(bits, r, rng)
                if is_success(majority_decode(bits), n):
                    ok += 1
            rows.append({"n": n, "p_max": p_max, "r": r, "N": N,
                         "trials": trials, "rate": ok / trials})
            print(f"  [B] r={r:<5} N={N:<5} 恢复率={ok/trials:.3f}", flush=True)
    return rows


def exp_c_tsparse(n, p_max, t_list, Ns, trials, rng):
    rows = []
    for t in t_list:
        for N in Ns:
            ok = 0
            for _ in range(trials):
                bits = sample_peaked(n, p_max, N, rng)
                bits = add_tflips(bits, t, rng)
                if is_success(hamming_center_decode(bits, t), n):
                    ok += 1
            rows.append({"n": n, "p_max": p_max, "t": t,
                         "ball2t": hamming_ball_size(n, 2 * t), "N": N,
                         "trials": trials, "rate": ok / trials})
            print(f"  [C] t={t} |B_2t|={rows[-1]['ball2t']:<7} N={N:<5} 恢复率={ok/trials:.3f}", flush=True)
    return rows


def exp_d_depolarizing(n, p_max, eps_list, Ns, trials, rng):
    rows = []
    d = 1 << n
    for eps in eps_list:
        pm = (1 - eps) * p_max + eps / d
        for N in Ns:
            est = []
            for _ in range(trials):
                bits = sample_peaked(n, p_max, N, rng)
                bits = add_depol(bits, eps, rng)
                p_hat_prime = float((bits.sum(axis=1) == 0).mean())
                est.append((p_hat_prime - eps / d) / (1 - eps))
            est = np.array(est)
            theory_se = math.sqrt(pm * (1 - pm) / N) / (1 - eps)
            rows.append({"n": n, "p_max": p_max, "eps": eps, "N": N, "trials": trials,
                         "mean_est": float(est.mean()),
                         "bias": float(est.mean() - p_max),
                         "std_est": float(est.std(ddof=1)),
                         "theory_se": theory_se})
            print(f"  [D] ε={eps:<5} N={N:<5} bias={rows[-1]['bias']:+.5f}"
                  f"  估计器SD={rows[-1]['std_est']:.5f} (理论 SE {theory_se:.5f})", flush=True)
    return rows


# ---------------- 主流程 ----------------

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="E1：peaked 输出的噪声鲁棒解码（对照论文 §4.4-4.5）")
    ap.add_argument("--quick", action="store_true", help="快速模式（约 1 分钟）")
    ap.add_argument("--outdir", default="autotest_results/e1_noise", help="输出目录")
    args = ap.parse_args()

    if args.quick:
        n, p_max = 8, 0.35
        cfg = dict(A_t=[0, 1, 2, 3], A_M=50, A_N=400,
                   B_r=[0.05, 0.20], B_N=[10, 20, 40, 80, 160, 320], B_trials=80,
                   C_t=[1, 2], C_N=[10, 20, 40, 80, 160, 320], C_trials=50,
                   D_eps=[0.1, 0.3], D_N=[100, 1000], D_trials=50)
    else:
        n, p_max = 12, 0.35
        cfg = dict(A_t=[0, 1, 2, 3, 4, 6, 8], A_M=200, A_N=2000,
                   B_r=[0.05, 0.10, 0.15, 0.20, 0.25],
                   B_N=[10, 20, 40, 80, 160, 320, 640, 1280], B_trials=200,
                   C_t=[1, 2, 3], C_N=[10, 20, 40, 80, 160, 320, 640, 1280], C_trials=100,
                   D_eps=[0.0, 0.1, 0.2, 0.3], D_N=[100, 400, 1600, 6400], D_trials=100)

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    print(f"=== E1 噪声鲁棒解码（n={n}, p_max={p_max}）===", flush=True)
    print("--- A: HBA 偏差界（命题 4.6）---", flush=True)
    A = exp_a_hba(n, p_max, cfg["A_t"], cfg["A_M"], cfg["A_N"], rng)
    write_csv(os.path.join(args.outdir, "hba_bias.csv"), A)

    print("--- B: BSC + 多数表决（命题 4.11）---", flush=True)
    B = exp_b_majority(n, p_max, cfg["B_r"], cfg["B_N"], cfg["B_trials"], rng)
    # p_max 依赖对照：r=0.10 下加测一个低 p_max
    B2 = exp_b_majority(n, 0.15, [0.10], cfg["B_N"], cfg["B_trials"], rng)
    B += B2
    write_csv(os.path.join(args.outdir, "bsc_majority.csv"), B)

    print("--- C: t-稀疏 + Hamming-中心（命题 4.10）---", flush=True)
    C = exp_c_tsparse(n, p_max, cfg["C_t"], cfg["C_N"], cfg["C_trials"], rng)
    write_csv(os.path.join(args.outdir, "tsparse_center.csv"), C)

    print("--- D: 全局去极化 + 去偏估计（§4.5）---", flush=True)
    D = exp_d_depolarizing(n, p_max, cfg["D_eps"], cfg["D_N"], cfg["D_trials"], rng)
    write_csv(os.path.join(args.outdir, "depolarizing.csv"), D)

    # ---- 标度拟合 ----
    fit = {}
    # B: N90 vs (1-2r)，期望斜率 ≈ -2
    rs, n90s = [], []
    for r in cfg["B_r"]:
        pts = [row for row in B if abs(row["r"] - r) < 1e-9 and abs(row["p_max"] - p_max) < 1e-9]
        v = n90([p["N"] for p in pts], [p["rate"] for p in pts])
        if v:
            rs.append(1 - 2 * r); n90s.append(v)
    if len(rs) >= 2:
        k = np.polyfit(np.log(rs), np.log(n90s), 1)[0]
        fit["B_slope_vs_(1-2r)"] = round(float(k), 3)
    # C: N90 vs |B_2t|，期望斜率 ≈ +1
    b2t, cs = [], []
    for t in cfg["C_t"]:
        pts = [row for row in C if row["t"] == t]
        v = n90([p["N"] for p in pts], [p["rate"] for p in pts])
        if v:
            b2t.append(hamming_ball_size(n, 2 * t)); cs.append(v)
    if len(b2t) >= 2:
        k = np.polyfit(np.log(b2t), np.log(cs), 1)[0]
        fit["C_slope_vs_|B_2t|"] = round(float(k), 3)

    # ---- 绘图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
    # A
    ax = axes[0][0]
    ts = [row["t"] for row in A if abs(row["p_max"] - p_max) < 1e-9]
    mean = [row["mean"] for row in A if abs(row["p_max"] - p_max) < 1e-9]
    sem = [row["sem"] for row in A if abs(row["p_max"] - p_max) < 1e-9]
    lo = [row["bound_lo"] for row in A if abs(row["p_max"] - p_max) < 1e-9]
    hi = [row["bound_hi"] for row in A if abs(row["p_max"] - p_max) < 1e-9]
    ax.errorbar(ts, mean, yerr=sem, marker="o", capsize=3, label="E[p̂(t)] measured")
    ax.fill_between(ts, lo, hi, alpha=0.2, color="tab:green", label="Prop. 4.6 bounds")
    ax.set_xlabel("t (Hamming ball radius)"); ax.set_ylabel("p̂(t)")
    ax.set_title("A: Hamming-ball aggregation bias"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    # B
    ax = axes[0][1]
    for r in cfg["B_r"]:
        pts = [row for row in B if abs(row["r"] - r) < 1e-9 and abs(row["p_max"] - p_max) < 1e-9]
        ax.plot([p["N"] for p in pts], [p["rate"] for p in pts], marker="o", label=f"r={r}")
    pts = [row for row in B if abs(row["r"] - 0.10) < 1e-9 and abs(row["p_max"] - 0.15) < 1e-9]
    if pts:
        ax.plot([p["N"] for p in pts], [p["rate"] for p in pts], marker="s", ls="--",
                label="r=0.10, p_max=0.15")
    ax.set_xscale("log"); ax.axhline(0.9, ls=":", color="gray")
    ax.set_xlabel("N (shots)"); ax.set_ylabel("recovery rate")
    ax.set_title("B: BSC + majority vote"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    # C
    ax = axes[1][0]
    for t in cfg["C_t"]:
        pts = [row for row in C if row["t"] == t]
        ax.plot([p["N"] for p in pts], [p["rate"] for p in pts], marker="o", label=f"t={t}")
    ax.set_xscale("log"); ax.axhline(0.9, ls=":", color="gray")
    ax.set_xlabel("N (shots)"); ax.set_ylabel("recovery rate")
    ax.set_title("C: t-sparse flips + Hamming-center"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    # D
    ax = axes[1][1]
    for eps in cfg["D_eps"]:
        pts = [row for row in D if abs(row["eps"] - eps) < 1e-9]
        ax.plot([p["N"] for p in pts], [p["std_est"] for p in pts], marker="o", label=f"ε={eps}")
        ax.plot([p["N"] for p in pts], [p["theory_se"] for p in pts], ls="--", lw=1, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (shots)"); ax.set_ylabel("estimator SE")
    ax.set_title("D: depolarizing — SE vs N (dashed = theory)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.outdir, "e1_noise_panels.png")
    fig.savefig(png)
    plt.close(fig)
    print("图已保存:", png)

    # ---- summary ----
    hba_ok = all(row["mean"] >= row["bound_lo"] - 4 * row["sem"] - 1e-9
                 and row["mean"] <= row["bound_hi"] + 4 * row["sem"] + 1e-9
                 for row in A if abs(row["p_max"] - p_max) < 1e-9)
    depol_ratios = [row["std_est"] / row["theory_se"] for row in D if row["theory_se"] > 0]
    depol_ok = all(0.5 <= q <= 2.0 for q in depol_ratios)
    lines = [
        "# E1 噪声鲁棒解码摘要",
        "",
        f"设置：n={n}，p_max={p_max}（分布模型：峰 + 均匀背景，与论文 Remark 4.7 一致）；种子={SEED}。",
        "噪声口径：t-稀疏为随机 t 位翻转（论文的界为对抗式最坏情况，随机翻转是对平均情形的探测）。",
        "",
        "## 子实验结论",
        f"- **A（HBA 偏差界，命题 4.6）**：{'✓ 实测点与论文界一致（均匀背景模型下与上界取等，容 4×SEM 统计涨落）' if hba_ok else '✗ 有超出容差的点'}。",
        f"- **B（多数表决，命题 4.11）**：N90 对 (1−2r) 的 log-log 斜率 = "
        f"{fit.get('B_slope_vs_(1-2r)', 'N/A')}（理论 ≈ −2）。",
        f"- **C（Hamming-中心，命题 4.10）**：N90 对 |B_2t| 的 log-log 斜率 = "
        f"{fit.get('C_slope_vs_|B_2t|', 'N/A')}（理论 ≈ +1）。",
        f"- **D（去极化去偏估计，§4.5）**：估计器标准差 / 理论 SE 比值 ∈ "
        f"[{min(depol_ratios):.2f}, {max(depol_ratios):.2f}]（{'✓ 一致' if depol_ok else '需检查'}）；"
        f"SD 随 N 按 1/√N 下降。",
        "",
        "## 明细",
        f"- A: {len(A)} 行 → hba_bias.csv；B: {len(B)} 行 → bsc_majority.csv；"
        f"C: {len(C)} 行 → tsparse_center.csv；D: {len(D)} 行 → depolarizing.csv。",
        "- 图：e1_noise_panels.png（2×2 面板）。",
    ]
    with open(os.path.join(args.outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"n": n, "p_max": p_max, "quick": args.quick, "seed": SEED,
                   "grids": {k: v for k, v in cfg.items() if isinstance(v, list)}},
                  f, indent=2, ensure_ascii=False)
    print("\n===== 摘要 =====")
    print("\n".join(lines))
    print(f"\n全部完成（用时 {time.time()-t0:.1f}s）。输出目录: {args.outdir}")


if __name__ == "__main__":
    main()
