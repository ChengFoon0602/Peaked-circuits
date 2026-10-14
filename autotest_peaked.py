# -*- coding: utf-8 -*-
"""autotest_peaked.py — Peaked-circuits 自动化实验脚本（多实例统计 + τp 扫描 + 结果落盘）

配套仓库：github.com/yuxuanzhang1995/Peaked-circuits（放本脚本到仓库根目录运行）
整理：申请工作区 ｜ 版本：2026-10-13（已在 peaked 环境实测通过）

用法（在仓库根目录、已 conda activate peaked 的前提下）：

    python autotest_peaked.py --quick     # 冒烟模式：小参数全流程，约 1-2 分钟
    python autotest_peaked.py             # 正式模式：默认约 20-40 分钟（CPU 笔记本）
    python autotest_peaked.py --seeds 10 --max-iter 800      # 自定义规模
    python autotest_peaked.py --depths 4,8 --seeds 3 --n 10  # 自定义扫描
    python autotest_peaked.py --restarts 3 --depths 6,9 --seeds 5  # 多起点：每实例 3 个随机初值取最优

输出（默认写到脚本旁的 autotest_results/run_<时间戳>/）：

    config.json        运行配置 + 软件版本（可复现性记录）
    raw_runs.csv       每次训练一行的原始数据（增量落盘，中途中断也不丢）
    baseline.csv       随机基线（各深度的平均尖峰权重）
    sweep_summary.csv  按 τp 汇总（均值/标准差/达标率/增强倍数）
    sweep_plot.png     δ–τp 曲线（训练 vs 随机基线，log 纵轴）
    summary.md         可直接贴进仓库 README 的结果摘要

要点：
- 可复现：每个实例用 qu.seed_rand(seed) 播种，同 seed 结果一致；
- 训练目标 loss_target=-0.99（与官方 notebook 一致，属"激进上限"）；单次未达标（局部收敛）是正常现象，
  不代表运行失败——脚本记录「达0.99比率」仅供参考（通常为 0）；
- --restarts R：多起点模式（R>1）。固定同一 rqc、用 R 个随机初值重开尖峰层训练并取最优；r=0 沿用旧播种序列，
  因此 R=1 时与原结果逐位一致。raw_runs.csv 的 delta_per_restart 列给出各起点明细（首个值即单起点结果）。
- 单次训练异常会记录错误并继续，不中断整个扫描。
"""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# 前置检查：必须在仓库根目录运行
if not os.path.exists(os.path.join(HERE, "circuits.py")):
    sys.exit("错误：请把本脚本放在 Peaked-circuits 仓库根目录（与 circuits.py 同级）再运行。")

# 先固定 matplotlib 后端，避免无界面环境报错
import matplotlib  # noqa: E402
matplotlib.use("Agg")

import quimb as qu  # noqa: E402
from circuits import qmps_f, negative_overlap, norm_f, average_peak_weight  # noqa: E402
from quimb.tensor.optimize import TNOptimizer  # noqa: E402


def train_one(n, rqc_depth, pqc_depth, seed, max_iter, loss_target, restarts=1):
    """训练一个实例，返回一条记录 dict。

    restarts > 1 时：固定同一 rqc，用不同随机初值重开尖峰层训练多次并取最优。
    r=0 沿用旧播种序列，因此 restarts=1 时与原结果逐位一致（同种子可复现）。
    """
    qu.seed_rand(seed)
    rqc = qmps_f(n, in_depth=rqc_depth, n_Qbit=n - 1, qmps_structure="brickwall")

    per_restart = []
    best = None
    total_seconds = 0.0
    for r in range(restarts):
        if r > 0:  # 第 0 个起点沿用旧播种序列（向后兼容）；其余起点用不同随机初值
            qu.seed_rand(seed + 100003 * r)
        pqc = qmps_f(n, in_depth=pqc_depth, n_Qbit=n - 1, qmps_structure="brickwall")
        opt = TNOptimizer(
            pqc,
            loss_fn=negative_overlap,
            norm_fn=norm_f,
            loss_target=loss_target,
            constant_tags=["MPS"],
            loss_constants={"target": rqc},
            autodiff_backend="autograd",
            optimizer="L-BFGS-B",
        )
        t0 = time.time()
        pqc_opt = opt.optimize(max_iter, tol=1e-10)
        total_seconds += time.time() - t0
        delta_best = float(-opt.loss_best)                        # 优化器跟踪的最佳峰值权重
        delta_measured = float(-negative_overlap(pqc_opt, rqc))   # 对返回电路直接测量
        per_restart.append(round(delta_best, 6))
        if best is None or delta_best > best["delta_best"]:
            best = {"delta_best": delta_best, "delta_measured": delta_measured}

    return {
        "n": n, "rqc_depth": rqc_depth, "pqc_depth": pqc_depth, "seed": seed,
        "max_iter": max_iter, "restarts": restarts,
        "delta_best": round(best["delta_best"], 6),
        "delta_measured": round(best["delta_measured"], 6),
        "delta_per_restart": ";".join(str(v) for v in per_restart),
        "seconds": round(total_seconds, 2),
        "reached_target": bool(best["delta_best"] >= (-loss_target) - 1e-6),
        "error": "",
    }


def run_baseline(n, depth, shots, seed):
    """随机基线：同深度随机电路的平均尖峰权重。"""
    qu.seed_rand(seed)
    mean, stderr, mx = average_peak_weight(n, depth=depth, shots=shots)
    return {"n": n, "depth": depth, "shots": shots,
            "mean": float(mean), "stderr": float(stderr), "max": float(mx)}


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Peaked-circuits 自动化实验（多实例统计 + τp 扫描）")
    ap.add_argument("--quick", action="store_true", help="小参数冒烟模式（约 1-2 分钟）")
    ap.add_argument("--n", type=int, default=None, help="qubit 数（默认 12）")
    ap.add_argument("--rqc-depth", type=int, default=None, help="随机层深度 τr（默认 = n）")
    ap.add_argument("--depths", type=str, default=None, help="τp 扫描列表，逗号分隔（默认 3,6,9,12）")
    ap.add_argument("--seeds", type=int, default=None, help="每个 τp 的实例数（默认 5）")
    ap.add_argument("--max-iter", type=int, default=None, help="每次训练的最大迭代数（默认 500）")
    ap.add_argument("--base-shots", type=int, default=None, help="基线每档采样数（默认 50）")
    ap.add_argument("--restarts", type=int, default=None, help="每个实例的随机起点数（默认 1；>1 时多重启取最优）")
    ap.add_argument("--outdir", type=str, default=None, help="输出目录（默认脚本旁 autotest_results/run_<时间戳>）")
    args = ap.parse_args()

    # 参数组合（--quick 覆盖默认；显式参数优先级最高）
    if args.quick:
        cfg = dict(n=8, rqc_depth=8, depths=[2, 4], seeds=2, max_iter=100, base_shots=10, restarts=1)
    else:
        cfg = dict(n=12, rqc_depth=12, depths=[3, 6, 9, 12], seeds=5, max_iter=500, base_shots=50, restarts=1)
    if args.n is not None:
        cfg["n"] = args.n
    if args.rqc_depth is not None:
        cfg["rqc_depth"] = args.rqc_depth
    if args.depths is not None:
        cfg["depths"] = [int(x) for x in args.depths.split(",")]
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    if args.max_iter is not None:
        cfg["max_iter"] = args.max_iter
    if args.base_shots is not None:
        cfg["base_shots"] = args.base_shots
    if args.restarts is not None:
        cfg["restarts"] = args.restarts

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join(HERE, "autotest_results", f"run_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    loss_target = -1 + 1e-2

    print("=== peaked-circuits autotest ===")
    print(f"配置: n={cfg['n']}, τr={cfg['rqc_depth']}, τp扫描={cfg['depths']}, "
          f"每档实例={cfg['seeds']}, max_iter={cfg['max_iter']}, 起点数={cfg['restarts']}, "
          f"基线shots={cfg['base_shots']}")
    print(f"输出目录: {outdir}\n", flush=True)

    # config.json（可复现性记录）
    import scipy
    meta = {
        "timestamp": stamp, "configuration": cfg, "loss_target": loss_target,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "quimb": qu.__version__,
                     "platform": platform.platform()},
    }
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    raw_path = os.path.join(outdir, "raw_runs.csv")
    raw_rows = []
    raw_fields = ["n", "rqc_depth", "pqc_depth", "seed", "max_iter", "restarts",
                  "delta_best", "delta_measured", "delta_per_restart", "seconds",
                  "reached_target", "error"]

    # ---- 逐档扫描（逐次增量落盘） ----
    total = len(cfg["depths"]) * cfg["seeds"]
    done = 0
    for tp in cfg["depths"]:
        for s in range(cfg["seeds"]):
            done += 1
            seed = 1000 * tp + s
            print(f"[{done}/{total}] τp={tp} seed={seed} ... ", end="", flush=True)
            try:
                rec = train_one(cfg["n"], cfg["rqc_depth"], tp, seed,
                                cfg["max_iter"], loss_target, restarts=cfg["restarts"])
            except Exception as e:  # 单次失败不中断
                rec = {"n": cfg["n"], "rqc_depth": cfg["rqc_depth"], "pqc_depth": tp,
                       "seed": seed, "max_iter": cfg["max_iter"],
                       "restarts": cfg["restarts"],
                       "delta_best": "", "delta_measured": "", "delta_per_restart": "",
                       "seconds": 0.0,
                       "reached_target": False, "error": f"{type(e).__name__}: {e}"}
            raw_rows.append(rec)
            write_csv(raw_path, raw_rows, raw_fields)
            if rec["error"]:
                print(f"失败（已记录继续）: {rec['error']}", flush=True)
            else:
                flag = " ✓达标" if rec["reached_target"] else ""
                extra = (f" [{rec['restarts']}起点: {rec['delta_per_restart']}]"
                         if rec.get("restarts", 1) > 1 else "")
                print(f"δ={rec['delta_best']:.4f} 用时={rec['seconds']:.1f}s{flag}{extra}", flush=True)

    # ---- 随机基线 ----
    print("\n计算随机基线 ...", flush=True)
    base_rows = []
    for tp in cfg["depths"]:
        rec = run_baseline(cfg["n"], tp, cfg["base_shots"], seed=90000 + tp)
        base_rows.append(rec)
        print(f"  depth={tp}: 均值={rec['mean']:.6f}", flush=True)
    write_csv(os.path.join(outdir, "baseline.csv"), base_rows,
              ["n", "depth", "shots", "mean", "stderr", "max"])

    # ---- 汇总 ----
    summary_rows = []
    for tp in cfg["depths"]:
        ok = [r for r in raw_rows if r["pqc_depth"] == tp and not r["error"]]
        vals = [float(r["delta_best"]) for r in ok]
        times = [float(r["seconds"]) for r in ok]
        reached = [r["reached_target"] for r in ok]
        base = next(b for b in base_rows if b["depth"] == tp)
        mean = float(np.mean(vals)) if vals else float("nan")
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append({
            "pqc_depth": tp, "runs": len(vals),
            "delta_mean": round(mean, 6), "delta_std": round(std, 6),
            "delta_max": round(float(np.max(vals)), 6) if vals else float("nan"),
            "reached_target_frac": round(sum(reached) / len(vals), 3) if vals else float("nan"),
            "mean_seconds": round(float(np.mean(times)), 2) if times else float("nan"),
            "baseline_mean": round(float(base["mean"]), 8),
            "enhancement": round(mean / base["mean"], 2) if (vals and base["mean"]) else float("nan"),
        })
    write_csv(os.path.join(outdir, "sweep_summary.csv"), summary_rows,
              list(summary_rows[0].keys()))

    # ---- 绘图 ----
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.6), dpi=200)
        tps = [r["pqc_depth"] for r in summary_rows]
        ax.errorbar(tps, [r["delta_mean"] for r in summary_rows],
                    yerr=[r["delta_std"] for r in summary_rows],
                    marker="o", capsize=4,
                    label=f"trained PQC (n={cfg['n']}, τr={cfg['rqc_depth']})")
        ax.errorbar(tps, [r["baseline_mean"] for r in summary_rows],
                    yerr=[b["stderr"] for b in base_rows],
                    marker="s", capsize=4, label="random baseline (same depth)")
        ax.axhline(-loss_target, ls="--", lw=1, color="gray",
                   label=f"loss target δ={-loss_target:.2f}")
        ax.set_yscale("log")
        ax.set_xlabel("τp (peaking layers)")
        ax.set_ylabel("peakedness δ (log scale)")
        ax.set_title("Peakedness vs peaking depth")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(outdir, "sweep_plot.png")
        fig.savefig(png)
        plt.close(fig)
        print(f"\n图已保存: {png}")
    except Exception as e:
        print(f"（绘图失败，不影响数据: {e}）")

    # ---- summary.md ----
    lines = [
        "# autotest 结果摘要",
        "",
        f"- 运行时间: {stamp} ｜ n={cfg['n']} ｜ τr={cfg['rqc_depth']} ｜ 每档实例={cfg['seeds']} ｜ max_iter={cfg['max_iter']} ｜ 起点数={cfg['restarts']}",
        f"- 训练目标 δ_target={-loss_target:.2f} ｜ 基线采样={cfg['base_shots']} shots",
        "",
        "| τp | 训练 δ（均值±标准差） | 最高 δ | 达0.99比率(参考) | 平均耗时(s) | 随机基线 | 增强倍数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['pqc_depth']} | {r['delta_mean']:.4f} ± {r['delta_std']:.4f} "
            f"| {r['delta_max']:.4f} | {r['reached_target_frac'] * 100:.0f}% "
            f"| {r['mean_seconds']:.1f} | {r['baseline_mean']:.6f} | {r['enhancement']:.1f}× |")
    lines += [
        "",
        "> 说明：δ = 训练后峰值权重 |⟨0|C'†C|0⟩|²（delta_best）；增强倍数 = 训练 δ 均值 / 随机基线均值。",
        "> 「达0.99比率」= 达到原 notebook 激进目标（δ≥0.99）的实例比例；通常为 0（论文同量级结果也远低于 0.99），属正常参考项，不代表运行失败。",
        "> 原始数据：raw_runs.csv ｜ 汇总：sweep_summary.csv ｜ 基线：baseline.csv ｜ 图：sweep_plot.png。",
    ]
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n===== 结果摘要 =====")
    print("\n".join(lines))
    print(f"\n全部完成。输出目录: {outdir}")


if __name__ == "__main__":
    main()
