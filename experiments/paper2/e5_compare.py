# -*- coding: utf-8 -*-
"""_e5_compare.py — 对比 E5 的 20 vs 40 起点，并检验原 +2.1σ 偏离是否为小样本涨落。

原实验用**对级 bootstrap**（重采样 190 个 |Tr|² 值）。但解对之间共享解，
独立性假设不成立，对级 CI 会偏窄。这里同时给：
  1) 对级 bootstrap（与原口径一致，便于对照）
  2) 解级 bootstrap（重采样 K 个解、重算全部对，口径更严）
并做分层：旧-旧 / 旧-新 / 新-新 —— 若"旧-新"与"旧-旧"一致，说明偏离稳定；
若"旧-新"更接近 2，说明原来那 20 个解只是碰巧偏高。
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # experiments/<group>/ → 仓库根
NEW = os.path.join(ROOT, "autotest_results", "e5_obfuscation_40seeds", "obf_pairs.csv")
K_OLD = 20


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [(int(r["i"]), int(r["j"]), float(r["tr2"])) for r in rows]


def boot_pair(vals, n_boot=20000, seed=7):
    a = np.asarray(vals, dtype=float)
    rg = np.random.default_rng(seed)
    mm = rg.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    return float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))


def boot_solution(pairs, K, n_boot=4000, seed=11):
    """解级 bootstrap：重采样 K 个解，用它们的全部二元组重算均值。"""
    rg = np.random.default_rng(seed)
    table = np.full((K, K), np.nan)
    for i, j, v in pairs:
        table[i, j] = table[j, i] = v
    out = np.empty(n_boot)
    for t in range(n_boot):
        idx = rg.integers(0, K, K)
        sub = table[np.ix_(idx, idx)]
        out[t] = np.nanmean(sub)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    if not os.path.isfile(NEW):
        sys.exit(f"找不到 {NEW}，先跑 40 起点实验")
    pairs = load(NEW)
    K = 1 + max(max(i, j) for i, j, _ in pairs)

    print(f"总对数 {len(pairs)}（K={K}，应为 {K*(K-1)//2}）\n")

    strata = {
        "全部 40 解": [(i, j, v) for i, j, v in pairs],
        "旧-旧 (i,j<20)": [(i, j, v) for i, j, v in pairs if i < K_OLD and j < K_OLD],
        "旧-新 (i<20<=j)": [(i, j, v) for i, j, v in pairs if i < K_OLD <= j],
        "新-新 (i,j>=20)": [(i, j, v) for i, j, v in pairs if i >= K_OLD and j >= K_OLD],
    }

    print(f"{'分层':<18}{'对数':>6}{'均值':>9}{'SEM':>8}{'对级 CI':>18}{'解级 CI':>18}{'z vs 2':>9}")
    for name, sub in strata.items():
        if len(sub) < 2:
            continue
        vals = np.array([v for _, _, v in sub])
        m = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals))
        lo1, hi1 = boot_pair(vals)
        lo2, hi2 = boot_solution(sub, K)
        z = (m - 2.0) / sem
        print(f"{name:<18}{len(sub):>6}{m:>9.4f}{sem:>8.4f}"
              f"{f'[{lo1:.3f}, {hi1:.3f}]':>18}{f'[{lo2:.3f}, {hi2:.3f}]':>18}{z:>+9.2f}")

    print("\n判读：")
    print("  · 对级 CI 与原实验同口径（原 20 起点：[2.027, 2.686]，均值 2.348）")
    print("  · 解级 CI 更宽属正常——解对之间共享解，独立性假设不成立")
    print("  · 若「旧-新」均值明显低于「旧-旧」，说明原偏离是小样本涨落")


if __name__ == "__main__":
    main()
