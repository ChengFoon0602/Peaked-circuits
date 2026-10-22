# -*- coding: utf-8 -*-
"""exp_mps_break.py — L2：MPS 键维断点（chi_break）扫描

对应论文 *Heuristic Quantum Advantage with Peaked Circuits* §4.2 / Fig. 3–4 的小规模版。

被攻击的对象
------------
用第一篇的构造训练 peaked 电路：R、P 都是 n 比特砖墙（每门为一般 2-qubit 幺正矩阵），
训练目标 delta = |<0|R^dag P|0>|^2，即攻击对象是

    C = R^dag · P ,   峰值比特串 s = 0^n ,   delta = |<0^n| C |0^n>|^2

攻击方法（边缘攻击 + MPS 近似）
------------------------------
1. 用电路的门接触图算 RCM（Reverse Cuthill–McKee）重排，把电路"掰"成一维链；
2. 按该顺序以 MPS（max_bond = chi）逐门演化：中心规范 + 逐门 SVD 截断（即 TEBD 式）；
3. 取各比特边缘 <Z_i>，按 sign(-<Z_i>) 拼出候选串 s_hat；
4. R = (n - Hamming(s, s_hat)) / n（随机猜 R = 0.5）。
chi_break = 使 R = 1 的最小 chi。理论上限 chi_sat = 2^(n/2)（此时 MPS 精确）。

子命令
------
    # 1) 训练并存盘（需要 peaked 环境：quimb 1.7.3 + circuits.py）
    python experiments/paper3/exp_mps_break.py gen --n 12 --depths 4,8,12 --max-iter 800

    # 2) 扫描（只依赖 numpy/scipy，任何环境都能跑）
    python experiments/paper3/exp_mps_break.py attack --chis 1,2,4,8,16,32,64

    # 3) 报告（csv -> md + png）
    python experiments/paper3/exp_mps_break.py report

全部产物落在 autotest_results/l2_mps_break/。
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # experiments/<group>/ → 仓库根
OUT = os.path.join(ROOT, "autotest_results", "l2_mps_break")
CIRC_DIR = os.path.join(OUT, "circuits")


# ============================================================================
# 电路的确定性结构（与 circuits.range_unitary 的循环顺序逐字对应）
# ============================================================================
def gate_pattern(n: int, depth: int):
    """返回砖墙电路里各门的两比特位置，顺序与 circuits.py 的施加顺序一致。"""
    n_qbit = n - 1
    pairs = []
    for r in range(depth):
        if r % 2 == 0:
            for i in range(0, 0 + n_qbit, 2):
                pairs.append((i, i + 1))
        else:
            for i in range(0, 0 + n_qbit - 1, 2):
                pairs.append((i + 1, i + 2))
    return pairs


# ============================================================================
# 精确态矢量（用于校验 MPS 与取"精确边缘"基线）
# ============================================================================
def norm_gate(i, j, u):
    """统一成 i < j；若交换了比特次序，则同时交换 U 的两条物理腿。"""
    if i < j:
        return i, j, u
    u4 = np.asarray(u, dtype=complex).reshape(2, 2, 2, 2)
    return j, i, np.transpose(u4, (1, 0, 3, 2)).reshape(4, 4)


def sv_gates_to_state(n: int, gates):
    """按 |0...0> 起算，逐个施加 gates（元素为 (i, j, U)），返回态矢量 (2,)*n。"""
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0
    for i0, j0, u0 in gates:
        i, j, u = norm_gate(i0, j0, u0)
        u4 = np.asarray(u, dtype=complex).reshape(2, 2, 2, 2)
        psi = psi.reshape(2 ** i, 2, 2 ** (j - i - 1), 2, 2 ** (n - j - 1))
        psi = np.einsum("abcd,XcYdZ->XaYbZ", u4, psi)
        psi = psi.reshape((2,) * n)
    return psi


def sv_marginals(psi):
    """各比特 <Z_i>：返回 shape (n,)。比特 0 是最高位。"""
    probs = np.abs(psi.reshape(-1)) ** 2
    probs = probs / probs.sum()
    idx = np.indices((2,) * psi.ndim).reshape(psi.ndim, -1)
    out = np.empty(psi.ndim)
    for i in range(psi.ndim):
        out[i] = 1.0 - 2.0 * probs[idx[i] == 1].sum()
    return out


# ============================================================================
# 重建候选串与打分
# ============================================================================
def bitstring_from_marginals(z):
    """论文口径：p(b_i=0) = (1+<Z_i>)/2，p<0.5 -> b_i=1。"""
    return "".join("1" if v < 0 else "0" for v in z)


def score_R(ans: str, guess: str) -> float:
    n = len(ans)
    return (n - sum(a != b for a, b in zip(ans, guess))) / n


# 基矢次序 = 2*b_first + b_second
SWAP4 = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=complex)


def robust_svd(m):
    """np.linalg.svd 在退化/病态矩阵上偶发 "SVD did not converge"；
    换 LAPACK 驱动（gesvd）兜底。实测同输入换线程数就能过，属数值偶发失败。"""
    try:
        return np.linalg.svd(m, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as ssvd
        return ssvd(m, full_matrices=False, lapack_driver="gesvd")


# ============================================================================
# MPS：中心规范 + 逐门 SVD 截断
# ============================================================================
class MPS:
    """site 张量 A_k 形状 (chi_l, 2, chi_r)；center 指示哪个 site 承载规范中心。

    长程门靠交换链实现：site 上的 label 会随 SWAP 变动，pos/site2log 负责记账，
    最后 z_marginals() 返回的是**按逻辑比特编号**的边缘。
    """

    def __init__(self, n: int):
        self.n = n
        self.A = [np.zeros((1, 2, 1), dtype=complex) for _ in range(n)]
        for t in self.A:
            t[0, 0, 0] = 1.0
        self.center = 0
        self.discarded = 0.0     # 累计截断权重
        self.norm_acc = 1.0      # 逐步归一化后单独记账的范数（= 截断造成的存活率）
        self.n_gate = 0
        self.n_swap = 0
        self.max_bond_seen = 1
        self.site2log = list(range(n))
        self.pos = list(range(n))

    # -- 合并两个相邻 site 再 SVD 拆分：direction 决定规范的落点 --
    #    'right'：A[k] 左等距，中心落到 k+1（用于把中心向右推）
    #    'left' ：A[k+1] 右等距，中心落到 k  （用于把中心向左推）
    def _merge_split(self, k, u, chi, direction):
        a, b = self.A[k], self.A[k + 1]
        t = np.tensordot(a, b, axes=([2], [0]))              # (cl,2,2,cr) = (X,p,q,Z)
        cl, cr = t.shape[0], t.shape[3]
        if u is not None:                                    # 在两条物理腿上施加 U
            u4 = np.asarray(u, dtype=complex).reshape(2, 2, 2, 2)   # (a,b,c,d)
            t = np.tensordot(u4, t, axes=([2, 3], [1, 2]))   # (a,b,X,Z)
            t = np.transpose(t, (2, 0, 1, 3))                # (X,a,b,Z)
        m = t.reshape(cl * 2, 2 * cr)
        u_s, s, vh = robust_svd(m)
        keep = min(chi, len(s))
        if keep < len(s):
            tot = float(np.sum(s ** 2))
            if tot > 0:
                self.discarded += float(np.sum(s[keep:] ** 2) / tot)
        u_s, s, vh = u_s[:, :keep], s[:keep], vh[:keep, :]
        nrm = float(np.linalg.norm(s))
        if nrm > 0 and np.isfinite(nrm):
            s = s / nrm
            self.norm_acc *= nrm
        if direction == "right":
            self.A[k] = u_s.reshape(cl, 2, keep)
            self.A[k + 1] = (s[:, None] * vh).reshape(keep, 2, cr)
            self.center = k + 1
        else:
            self.A[k] = (u_s * s).reshape(cl, 2, keep)
            self.A[k + 1] = vh.reshape(keep, 2, cr)
            self.center = k
        self.max_bond_seen = max(self.max_bond_seen, keep)

    def move_center_to(self, k, chi):
        while self.center < k:
            self._merge_split(self.center, None, chi, "right")
        while self.center > k:
            self._merge_split(self.center - 1, None, chi, "left")

    def _swap_adjacent(self, k, chi):
        """把 site k 与 k+1 上的逻辑比特交换（SWAP 门 + 截断），同步更新布局。"""
        self._merge_split(k, SWAP4, chi, "right")
        self.site2log[k], self.site2log[k + 1] = self.site2log[k + 1], self.site2log[k]
        self.pos[self.site2log[k]] = k
        self.pos[self.site2log[k + 1]] = k + 1
        self.n_swap += 1

    # -- 单门施加：靠交换链把 j 挪到 i 旁边，再在相邻两站上施加并截断 --
    #    距离 L 的门需要 (L-1) 次 SWAP，代价 O(L·chi^3)——这正是 RCM 重排要省的东西
    def apply_gate(self, i0, j0, u0, chi):
        i, j, u = norm_gate(i0, j0, u0)
        si, sj = self.pos[i], self.pos[j]
        if si > sj:                       # 让位置靠后的一侧向靠前的靠拢
            i, j, u = norm_gate(j, i, u)
            si, sj = sj, si
        while sj > si + 1:
            self._swap_adjacent(sj - 1, chi)
            sj -= 1
        self.move_center_to(si, chi)
        self._merge_split(si, u, chi, "right")
        self.n_gate += 1

    # -- 边缘 --
    def z_marginals(self):
        n = self.n
        env_l = [None] * (n + 1)
        env_r = [None] * (n + 1)
        env_l[0] = np.ones((1, 1), dtype=complex)
        for k in range(n):
            a = self.A[k]
            env_l[k + 1] = np.einsum("ac,asb,csd->bd", env_l[k], a, a.conj())
        env_r[n] = np.ones((1, 1), dtype=complex)
        for k in range(n - 1, -1, -1):
            a = self.A[k]
            env_r[k] = np.einsum("asb,csd,bd->ac", a, a.conj(), env_r[k + 1])
        z = np.empty(n)
        norm = float(np.real(env_l[n][0, 0]))
        zvec = np.array([1.0, -1.0])
        for k in range(n):
            a = self.A[k]
            val = np.einsum("ac,asb,csd,s,bd->", env_l[k], a, a.conj(), zvec, env_r[k + 1])
            z[self.site2log[k]] = float(np.real(val)) / norm
        return z, norm


def evolve_mps(n, gates, chi):
    """按 gates 顺序以 MPS 演化，返回 (z_marginals, 诊断 dict)。"""
    t0 = time.time()
    psi = MPS(n)
    chi_eff = min(chi, 2 ** (n // 2))
    for i, j, u in gates:
        psi.apply_gate(i, j, u, chi_eff)
    z, norm = psi.z_marginals()
    return z, {
        "seconds": time.time() - t0,
        "discarded": psi.discarded,
        "norm": psi.norm_acc,
        "max_bond": psi.max_bond_seen,
        "n_swap": psi.n_swap,
        "chi_eff": chi_eff,
    }


# ============================================================================
# RCM 重排
# ============================================================================
def contact_graph(n, gates):
    from scipy.sparse import coo_matrix
    rows, cols, vals = [], [], []
    for i, j, _ in gates:
        rows += [i, j]
        cols += [j, i]
        vals += [1.0, 1.0]
    g = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return g


def bandwidth(order, gates):
    pos = {q: k for k, q in enumerate(order)}
    return max(abs(pos[i] - pos[j]) for i, j, _ in gates)


def rcm_order(n, gates):
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    g = contact_graph(n, gates)
    return list(reverse_cuthill_mckee(g, symmetric_mode=True))


def relabel(gates, perm):
    """把比特 q 换成 perm[q]（perm 是"新标签 -> 原标签"还是反之，这里定义为 new[old]）。"""
    return [(perm[i], perm[j], u) for i, j, u in gates]


def reorder_gates_to_chain(gates, order):
    """按给定线性顺序 order 重标号，使链序 = 0..n-1。order[k] = 原来第 k 位的物理比特。"""
    inv = {q: k for k, q in enumerate(order)}
    return [(inv[i], inv[j], u) for i, j, u in gates]


# ============================================================================
# gen：训练 + 校验 + 存盘
# ============================================================================
def cmd_gen(args):
    import quimb as qu
    from quimb.tensor.optimize import TNOptimizer
    from circuits import qmps_f, negative_overlap, norm_f

    def gate_tensors(tn, n_gates):
        """按 tag 编号取门张量**对象**（不是数据）。

        不直接用 save_para：优化后期 isometrize 可能让 tag 同时挂到站点张量上，
        save_para 取 t[0] 就会拿错。这里按"4 维且 16 个元素"筛出真正的门张量。
        """
        objs = []
        for k in range(n_gates):
            cand = [t for t in tn.select(f"G{k}") if t.ndim == 4 and t.size == 16]
            if len(cand) != 1:
                raise RuntimeError(f"tag G{k} 命中 {len(cand)} 个候选门张量")
            objs.append(cand[0])
        return objs

    os.makedirs(CIRC_DIR, exist_ok=True)
    for n in args.ns:
        for d in args.depths:
            for si in range(args.nseeds):
                seed_i = args.seed + 1000 * si
                pairs = gate_pattern(n, d)
                G = 2 * len(pairs)
                print(f"[gen] n={n} d={d} s={si} G={G} target δ={args.target_delta} ... ",
                      end="", flush=True)
                t0 = time.time()

                # R：自己生成 Haar 门表并以 uni_list 传入。不从 TN 里提取——
                # 未优化的网络里门张量被吸收进站点张量（G0 会带 I0/I1 标签），
                # 只有被 TNOptimizer unpack 过的网络才有独立门张量。
                qu.seed_rand(seed_i)
                mats_R = [np.asarray(qu.rand_uni(4, dtype=complex)) for _ in pairs]
                rqc = qmps_f(n, in_depth=d, n_Qbit=n - 1, qmps_structure="brickwall",
                             uni_list=mats_R)

                qu.seed_rand(seed_i + 1)
                pqc = qmps_f(n, in_depth=d, n_Qbit=n - 1, qmps_structure="brickwall")

                opt = TNOptimizer(
                    pqc, loss_fn=negative_overlap, norm_fn=norm_f,
                    loss_target=-args.target_delta, constant_tags=["MPS"],
                    loss_constants={"target": rqc},
                    autodiff_backend="autograd", optimizer="L-BFGS-B",
                )
                trained = opt.optimize(args.max_iter, tol=1e-10)
                delta_ref = float(-negative_overlap(trained, rqc))

                # 注意：opt.optimize() 返回的是**新的 TN 对象**，必须从里面重新抓门张量
                gP = [np.asarray(t.data).reshape(4, 4).T
                      for t in gate_tensors(trained, len(pairs))]

                # 校验 A：把提取的门塞回 qmps_f，与训练后的 TN 求重叠，必须为 1
                reb = qmps_f(n, in_depth=d, n_Qbit=n - 1, qmps_structure="brickwall",
                             uni_list=gP)
                ov = float(abs((trained.H & reb) ^ all))
                if abs(ov - 1.0) > 1e-8:
                    raise RuntimeError(f"n={n} d={d}: P 电路重建失败 overlap={ov:.6f}")

                # C = R^dag · P ：先 P，再 R 的逆序共轭转置
                gates = [(i, j, u) for (i, j), u in zip(pairs, gP)]
                gates += [(i, j, np.asarray(u).conj().T) for (i, j), u in
                          reversed(list(zip(pairs, mats_R)))]

                # 校验 B：独立态矢量算 <0|C|0>，须与 delta_ref 一致
                psi = sv_gates_to_state(n, gates)
                delta_sv = float(abs(psi[(0,) * n]) ** 2)
                err = abs(delta_sv - delta_ref)
                if err > 1e-6:
                    raise RuntimeError(f"n={n} d={d}: δ 校验失败 {delta_sv:.6f} vs {delta_ref:.6f}")

                rec = {
                    "n": n, "depth": d, "gate_count": G, "seed": seed_i,
                    "target_delta": args.target_delta,
                    "delta": delta_sv, "delta_ref": delta_ref, "check_err": err,
                    "target": "0" * n,
                    "gates": [[int(i), int(j),
                               [[float(v.real), float(v.imag)] for v in u.reshape(-1)]]
                              for i, j, u in gates],
                }
                ttag = f"t{int(round(args.target_delta * 100)):03d}"
                path = os.path.join(CIRC_DIR, f"n{n}_d{d}_{ttag}_s{si}_G{G}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(rec, f)
                print(f"δ={delta_sv:.4f} 双校验✓ 用时={time.time()-t0:.1f}s",
                      flush=True)


def load_circuits():
    files = sorted(glob.glob(os.path.join(CIRC_DIR, "n*_d*_G*.json")))
    out = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        gates = []
        for i, j, uv in rec["gates"]:
            u = np.array([complex(a, b) for a, b in uv]).reshape(4, 4)
            gates.append((i, j, u))
        rec["_gates"] = gates
        rec["_file"] = os.path.basename(p)
        out.append(rec)
    return out


def cmd_attack(args):
    chis = [int(x) for x in args.chis.split(",")]
    circs = load_circuits()
    if args.ns:
        keep = {int(x) for x in args.ns.split(",")}
        circs = [c for c in circs if c["n"] in keep]
    if not circs:
        raise SystemExit("没有找到电路，先跑 gen")
    rows = []
    for rec in circs:
        n, gates, ans = rec["n"], rec["_gates"], rec["target"]
        # 精确基线
        psi = sv_gates_to_state(n, gates)
        z_exact = sv_marginals(psi)
        R_exact = score_R(ans, bitstring_from_marginals(z_exact))
        print(f"\n=== {rec['_file']}  G={rec['gate_count']} δ={rec['delta']:.4f} "
              f"精确边缘 R={R_exact:.4f} (min<Z>={z_exact.min():+.4f}) ===", flush=True)
        rows.append(dict(case=rec["_file"], n=n, depth=rec["depth"], gate_count=rec["gate_count"],
                         delta=rec["delta"], chi=0, chi_eff=0, R=R_exact, seconds=0.0,
                         discarded=0.0, norm=1.0, max_bond=0, bandwidth=bandwidth(range(n), gates),
                         layout="identity", use_rcm="exact", ans=ans,
                         guess=bitstring_from_marginals(z_exact), note=""))

        # 两种布局 × (是否用 RCM)
        want = {x.strip() for x in args.layouts.split(",")}
        layouts = ({"identity": list(range(n))} if "identity" in want
                   else {})
        if args.scramble is not None and "scrambled" in want:
            rng = np.random.default_rng(args.scramble)
            forced = list(range(n))
            rng.shuffle(forced)
            layouts["scrambled"] = forced

        for lname, perm in layouts.items():
            gl = relabel(gates, perm) if lname != "identity" else gates
            orders = {"identity": list(range(n))}
            if lname != "identity":
                orders["rcm"] = rcm_order(n, gl)
            for oname, order in orders.items():
                gg = reorder_gates_to_chain(gl, order)
                bw = bandwidth(range(n), gg)
                # 只扫到该 n 的理论上限 chi_sat = 2^(n/2)，更大没有意义
                for chi in [c for c in chis if c <= 2 ** (n // 2)]:
                    z, diag = evolve_mps(n, gg, chi)
                    guess = bitstring_from_marginals(z)
                    R = score_R(ans, guess)
                    print(f"  {lname:9s}/{oname:8s} χ={chi:<4d} R={R:.4f} "
                          f"t={diag['seconds']:7.3f}s disc={diag['discarded']:.2e} bw={bw}",
                          flush=True)
                    rows.append(dict(case=rec["_file"], n=n, depth=rec["depth"],
                                     gate_count=rec["gate_count"], delta=rec["delta"],
                                     chi=chi, chi_eff=diag["chi_eff"], R=R,
                                     seconds=diag["seconds"], discarded=diag["discarded"],
                                     norm=diag["norm"], max_bond=diag["max_bond"],
                                     bandwidth=bw, layout=lname, use_rcm=oname,
                                     ans=ans, guess=guess,
                                     note="exact" if chi == 0 else ""))
    path = os.path.join(OUT, args.out)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n写出 {path}（{len(rows)} 行）")


# ============================================================================
# report：chi_break 汇总 + 图 + md
# ============================================================================
def cmd_report(args):
    import collections

    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, "chi_scan*.csv"))):
        with open(path, encoding="utf-8-sig") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise SystemExit("没有 chi_scan*.csv，先跑 attack")
    for r in rows:
        for k in ("chi", "chi_eff", "gate_count", "n", "depth", "max_bond", "bandwidth"):
            r[k] = int(r[k])
        for k in ("R", "seconds", "discarded", "delta"):
            r[k] = float(r[k])

    # ---- 按 (n, depth, layout, rcm) 聚合 R(chi)：实例间取均值 ----
    grp = collections.defaultdict(list)
    for r in rows:
        if r["chi"] == 0:
            continue
        grp[(r["n"], r["depth"], r["layout"], r["use_rcm"])].append(r)

    curves = []
    for (n, d, layout, rcm), sub in sorted(grp.items()):
        by_chi = collections.defaultdict(list)
        for r in sub:
            by_chi[r["chi"]].append(r)
        for chi in sorted(by_chi):
            ss = by_chi[chi]
            curves.append({
                "n": n, "depth": d, "gate_count": ss[0]["gate_count"],
                "delta_mean": float(np.mean([x["delta"] for x in ss])),
                "layout": layout, "use_rcm": rcm, "chi": chi,
                "R_mean": float(np.mean([x["R"] for x in ss])),
                "R_std": float(np.std([x["R"] for x in ss], ddof=1)) if len(ss) > 1 else 0.0,
                "instances": len(ss),
                "seconds_mean": float(np.mean([x["seconds"] for x in ss])),
                "discarded_mean": float(np.mean([x["discarded"] for x in ss])),
            })
    with open(os.path.join(OUT, "curves.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
        w.writeheader(); w.writerows(curves)

    # ---- 每个实例单独找 chi_break（R=1 的最小 chi），再对实例取中位 ----
    per_inst = collections.defaultdict(dict)   # (layout,rcm,case) -> {chi: R}
    for r in rows:
        if r["chi"] == 0:
            continue
        per_inst[(r["layout"], r["use_rcm"], r["case"])][r["chi"]] = r["R"]

    summ = []
    for (layout, rcm) in sorted({(r["layout"], r["use_rcm"]) for r in rows}):
        cases = sorted({c for (l, m, c) in per_inst if l == layout and m == rcm})
        by_nd = collections.defaultdict(list)
        for c in cases:
            meta = next(r for r in rows if r["case"] == c and r["chi"] > 0)
            by_nd[(meta["n"], meta["depth"])].append(c)
        for (n, d), cs in sorted(by_nd.items()):
            breaks, never = [], 0
            for c in cs:
                ks = sorted(per_inst[(layout, rcm, c)])
                hit = [k for k in ks if per_inst[(layout, rcm, c)][k] >= 1.0 - 1e-12]
                if hit:
                    breaks.append(hit[0])
                else:
                    never += 1
            meta = next(r for r in rows if r["case"] == cs[0] and r["chi"] > 0)
            summ.append({
                "n": n, "depth": d, "gate_count": meta["gate_count"],
                "delta_mean": float(np.mean([next(r["delta"] for r in rows
                                                  if r["case"] == c and r["chi"] > 0) for c in cs])),
                "layout": layout, "use_rcm": rcm,
                "instances": len(cs), "never_reached": never,
                "chi_break_median": float(np.median(breaks)) if breaks else float("nan"),
                "chi_break_min": float(np.min(breaks)) if breaks else float("nan"),
                "chi_sat": 2 ** (n // 2),
            })
    with open(os.path.join(OUT, "chi_break.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)

    print("\n=== chi_break 汇总（每实例 R=1 的最小 chi，再取中位）===")
    print(f"{'n':>3s}{'G':>6s}{'δ':>7s}{'布局':>16s}{'实例':>5s}{'未达1':>6s}"
          f"{'chi_break':>10s}{'chi_sat':>8s}")
    for r in summ:
        print(f"{r['n']:3d}{r['gate_count']:6d}{r['delta_mean']:7.3f}"
              f"{r['layout'] + '/' + r['use_rcm']:>16s}{r['instances']:5d}"
              f"{r['never_reached']:6d}{r['chi_break_median']:10.0f}{r['chi_sat']:8d}")

    # ---- 时间标度：每门耗时 + 最大两档 chi 间的有效指数 ----
    # 说明：n 小时键维只有中央若干条键能打满 chi，耗时被每门固定开销支配，
    # 直接拟合 a*chi^2 + b*chi^3 会得到负的 b（无意义）。这里给经验有效指数。
    fit_lines = []
    for n in sorted({c["n"] for c in curves}):
        for d in sorted({c["depth"] for c in curves if c["n"] == n}):
            pts = sorted([c for c in curves if c["n"] == n and c["depth"] == d
                          and c["layout"] == "identity"], key=lambda c: c["chi"])
            if len(pts) < 3:
                continue
            G = pts[0]["gate_count"]
            per_gate = [q["seconds_mean"] / G for q in pts]
            c1, c2 = pts[-3], pts[-1]
            p_eff = (np.log(c2["seconds_mean"] / c1["seconds_mean"])
                     / np.log(c2["chi"] / c1["chi"]))
            fit_lines.append((n, d, G, float(np.max(per_gate)), float(p_eff),
                              pts[-1]["chi"], pts[-1]["max_bond"] if "max_bond" in pts[-1] else 0))

    # ---- 写 md ----
    md = ["# L2 · MPS chi_break 扫描结果", "",
          f"- 生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}",
          f"- 电路族：C = R†P（第一篇构造），峰值比特串 0^n，训练到目标 δ 即停",
          f"- 攻击：RCM 重排 + MPS(max_bond=chi) 逐门演化（中心规范 + SVD 截断）→ 边缘 <Z_i> → 候选串",
          f"- 指标：R = 正确比特比例（随机猜 0.5）；chi_break = 单实例 R=1 的最小 chi，再对实例取中位",
          "", "## chi_break 汇总", "",
          "| n | G | δ | 布局/重排 | 实例 | chi_break | chi_sat=2^{n/2} |",
          "|---|---|---|---|---|---|---|"]
    for r in summ:
        md.append(f"| {r['n']} | {r['gate_count']} | {r['delta_mean']:.3f} "
                  f"| {r['layout']}/{r['use_rcm']} | {r['instances']} "
                  f"| {r['chi_break_median']:.0f} | {r['chi_sat']} |")
    if fit_lines:
        md += ["", "## 时间标度（identity 布局）", "",
               "| n | G | 最大每门耗时 (s) | 有效指数 p | 最大 chi |",
               "|---|---|---|---|---|",
               "",
               "> 有效指数 p 取最大三档 chi 中首尾两点的 log-log 斜率。n 小时只有中央少数几条键能打满",
               "> chi，总耗时被每门固定开销支配，故 p 明显小于 3；chi^3 主导区要等 n≥16 才显现。"]
        for n, d, G, pg, p_eff, c_max, mb in fit_lines:
            md.append(f"| {n} | {G} | {pg:.3e} | {p_eff:.2f} | {c_max} |")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
        ax = axes[0]
        pts = [c for c in curves if c["layout"] == "identity"]
        # 单实例 R 的查表（画散点用）：R 只能取 k/n，误差棒会拉得又长又乱
        raw = collections.defaultdict(list)
        for r in rows:
            if r["chi"] > 0 and r["layout"] == "identity":
                raw[(r["n"], r["depth"], r["chi"])].append(r["R"])
        for n in sorted({c["n"] for c in pts}):
            for d in sorted({c["depth"] for c in pts if c["n"] == n}):
                ss = sorted([c for c in pts if c["n"] == n and c["depth"] == d],
                            key=lambda c: c["chi"])
                chis = [c["chi"] for c in ss]
                line, = ax.plot(chis, [c["R_mean"] for c in ss], marker="o", ms=3.4,
                                lw=1.2, label=f"n={n} G={ss[0]['gate_count']}")
                # 每个实例一个细点，横向微抖开；均值仍是折线
                for c in ss:
                    vals = raw[(n, d, c["chi"])]
                    if len(vals) <= 1:
                        continue
                    offs = np.linspace(0.90, 1.10, len(vals))
                    ax.scatter([c["chi"] * o for o in offs], vals, s=10,
                               color=line.get_color(), alpha=0.55, linewidths=0, zorder=1)
        ax.axhline(0.5, ls=":", color="gray", lw=1, label="random guess")
        ax.set_xscale("log", base=2); ax.set_ylim(0.30, 1.04)
        ax.set_xlabel(r"bond dimension $\chi$")
        ax.set_ylabel("R (correct-bit fraction)")
        ax.set_title("MPS marginal attack: R vs $\\chi$  (line = mean, dots = instances)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6, ncol=2)

        ax2 = axes[1]
        for (layout, rcm), mk in [(("identity", "identity"), "o"),
                                  (("scrambled", "identity"), "s"),
                                  (("scrambled", "rcm"), "^")]:
            ss = [r for r in summ if r["layout"] == layout and r["use_rcm"] == rcm
                  and np.isfinite(r["chi_break_median"])]
            if not ss:
                continue
            for n in sorted({r["n"] for r in ss}):
                tt = [r for r in ss if r["n"] == n]
                ax2.plot([r["gate_count"] for r in tt], [r["chi_break_median"] for r in tt],
                         marker=mk, ls="-" if rcm == "rcm" or layout == "identity" else "--",
                         label=f"{layout}/{rcm} n={n}")
        for n in sorted({r["n"] for r in summ}):
            ax2.axhline(2 ** (n / 2), ls=":", lw=1, color="0.55")
            ax2.annotate(r"$\chi_{\rm sat}(n=%d)=%d$" % (n, 2 ** (n // 2)),
                         xy=(0.34, 2 ** (n / 2)),
                         xycoords=("axes fraction", "data"), fontsize=6, color="0.4",
                         va="bottom")
        ax2.set_xlabel("gate count G")
        ax2.set_ylabel(r"$\chi_{\rm break}$ (median)")
        ax2.set_yscale("log", base=2)
        ax2.set_title(r"$\chi_{\rm break}$ vs G")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(fontsize=6)
        fig.tight_layout()
        png = os.path.join(OUT, "chi_break.png")
        fig.savefig(png)
        plt.close(fig)
        print(f"\n图: {png}")
    except Exception as e:
        print(f"（绘图失败: {e}）")
    print(f"\n表: {os.path.join(OUT, 'chi_break.csv')} / curves.csv / summary.md")


def main():
    ap = argparse.ArgumentParser(description="L2: MPS chi_break 扫描")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--ns", type=str, default="12")
    g.add_argument("--depths", type=str, default="4,8,12")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--max-iter", type=int, default=800)
    g.add_argument("--nseeds", type=int, default=1, help="每个 (n,depth) 训练几个实例")
    g.add_argument("--target-delta", type=float, default=0.99,
                   help="训练到该尖峰度就停；论文公开电路 δ_s≈0.1，用 0.15 更接近")

    a = sub.add_parser("attack")
    a.add_argument("--chis", type=str, default="1,2,4,8,16,32,64")
    a.add_argument("--scramble", type=int, default=None, help="给该种子做一次随机布局打乱")
    a.add_argument("--ns", type=str, default=None, help="只跑这些 n（逗号分隔）")
    a.add_argument("--layouts", type=str, default="identity,scrambled",
                   help="identity=原链序；scrambled=随机布局（自动带 RCM 对照）")
    a.add_argument("--out", type=str, default="chi_scan.csv", help="输出 csv 文件名")

    sub.add_parser("report")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.cmd == "gen":
        args.ns = [int(x) for x in args.ns.split(",")]
        args.depths = [int(x) for x in args.depths.split(",")]
        cmd_gen(args)
    elif args.cmd == "attack":
        cmd_attack(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
