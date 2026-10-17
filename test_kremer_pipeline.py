# -*- coding: utf-8 -*-
"""Paper-3 (HQAP): end-to-end check of the Kremer-Dupuis TNO pipeline.

We do not have the BlueQubit QASM files here, so we build a small R+P circuit
ourselves (same architecture as the trilogy papers), run Kremer's
``contract_core -> tno_to_tne -> extract_bitstring`` chain on it in CPU mode,
and cross-check the recovered bitstring against an exact statevector
computation.

Pass criteria
-------------
1. The pipeline runs without error.
2. The recovered bitstring is the true argmax of the exact output
   distribution, OR its exact probability is within the top-k of the
   distribution (k = 5) -- low-bond truncation is allowed to miss slightly.

Usage (from this repo's root, inside the hqap environment):
    set KREMER_HOME=D:\\桌面文件\\大三\\peaked-circuit-simulation
    python test_kremer_pipeline.py --n 10 --depth 8
"""
import argparse
import os
import sys
import time

import numpy as np
from qiskit import QuantumCircuit

KREMER_HOME = os.environ.get(
    "KREMER_HOME", r"D:\桌面文件\大三\peaked-circuit-simulation"
)
if KREMER_HOME not in sys.path:
    sys.path.insert(0, KREMER_HOME)


def build_product(n, angle, seed):
    """No-entanglement circuit with a controlled peak at |0...0>.

    Each qubit gets Ry(angle); P(0...0) = cos(angle/2)^(2n). With a small
    angle this is a genuine (if trivial) peaked circuit and the TNO pipeline
    can represent it exactly -- so the run tests the *numerical* path, not
    the truncation heuristic.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, n)
    for q in range(n):
        qc.ry(float(angle + rng.normal(0, 0.05)), q)
    return qc


def build_rqc(n, depth, seed):
    """Random quantum circuit: per layer, random 1q rotations + a CZ/CX layer.

    Note: qiskit's little-endian convention means qubit 0 is the least
    significant bit of a sampled bitstring.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, n)
    for _ in range(depth):
        for q in range(n):
            kind = rng.integers(0, 3)
            theta = rng.uniform(0, 2 * np.pi)
            if kind == 0:
                qc.rx(theta, q)
            elif kind == 1:
                qc.ry(theta, q)
            else:
                qc.rz(theta, q)
        pair_start = rng.integers(0, 2)
        for q in range(pair_start, n - 1, 2):
            if rng.random() < 0.5:
                qc.cz(q, q + 1)
            else:
                qc.cx(q, q + 1)
    return qc


def exact_distribution(qc):
    """Exact probabilities via qiskit statevector (small n only)."""
    from qiskit.quantum_info import Statevector

    sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
    return np.abs(np.asarray(sv.data)) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["product", "rqc"], default="product",
                    help="product: 无纠缠（验证数值链路）；rqc: 随机电路（验证截断行为）")
    ap.add_argument("--angle", type=float, default=0.35,
                    help="product 模式下每比特 Ry 角度（越小峰值越高）")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--max-bond", type=int, default=16)
    ap.add_argument("--cutoff", type=float, default=0.01)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    print("=" * 64)
    print(f"Kremer TNO 端到端验证  (mode={args.mode}, n={args.n}, "
          f"max_bond={args.max_bond})")
    print("=" * 64)

    from utils import contract_core, tno_to_tne, extract_bitstring, iter_layers

    if args.mode == "product":
        qc = build_product(args.n, args.angle, args.seed)
    else:
        qc = build_rqc(args.n, args.depth, args.seed)
    print(f"电路: {qc.num_qubits} qubits, {qc.size()} 门, 深度 {qc.depth()}")

    # ---- exact reference ----
    t_exact = time.perf_counter()
    probs = exact_distribution(qc)
    exact_s = time.perf_counter() - t_exact
    order = np.argsort(probs)[::-1]
    top = [(format(int(i), f"0{args.n}b"), float(probs[i])) for i in order[: args.topk]]
    print(f"\n精确 statevector ({exact_s:.2f}s)：")
    for bs, p in top:
        print(f"    {bs}   p = {p:.6f}")
    exact_argmax = top[0][0]

    # ---- Kremer pipeline ----
    t0 = time.perf_counter()
    layers = list(iter_layers(qc))
    print(f"\n分层: {len(layers)} 层（qiskit DAG 自动切分）")

    tno, stats = contract_core(
        layers,
        chunk_size=args.chunk,
        cutoff=args.cutoff,
        max_bond=args.max_bond,
        to_backend=None,          # CPU
    )
    t1 = time.perf_counter()
    print(f"contract_core: {t1 - t0:.2f}s, 最大 bonds {max(s['max_bond'] for s in stats)}")

    tne = tno_to_tne(tno, max_bond=8, cutoff=args.cutoff, to_backend=None)
    bs_rec, marginals = extract_bitstring(tne)
    t2 = time.perf_counter()
    print(f"tno_to_tne + extract_bitstring: {t2 - t1:.2f}s")

    # Kremer returns bitstring in their own qubit ordering; normalise to the
    # same convention as the exact reference by trying both orientations.
    cand = {bs_rec, bs_rec[::-1]}
    p_rec = max(float(probs[int(b, 2)]) for b in cand if len(b) == args.n)
    print(f"\n恢复的比特串: {bs_rec}   精确概率 p = {p_rec:.6f}")

    # ---- verdict ----
    top_set = {b for b, _ in top}
    hit_argmax = exact_argmax in cand
    hit_topk = bool(cand & top_set)

    print("\n" + "-" * 64)
    if hit_argmax:
        print("PASS: 恢复串就是精确分布的 argmax")
    elif hit_topk:
        print(f"PASS（宽松）: 恢复串落在精确分布 top-{args.topk} 内（低键截断的正常容差）")
    else:
        print("FAIL: 恢复串既不是 argmax，也不在 top-k 内")
        print(f"      精确 argmax = {exact_argmax}")
        return 1
    print(f"      总耗时 {time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
