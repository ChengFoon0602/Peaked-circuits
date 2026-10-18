# -*- coding: utf-8 -*-
"""exp_qasm_roundtrip.py — 把训练出的 peaked 电路导出为 OpenQASM，并做闭环校验

流程
----
1. 读 `autotest_results/l2_mps_break/circuits/*.json` 里的门表（一般 2-qubit 幺正，C = R†P）；
2. 转译到基组 {u, cx}（`UnitaryGate` 必须展开才能在 QASM 里表达）；
3. 分别导出 OpenQASM 3（qasm3.dumps）与 OpenQASM 2.0（QuantumCircuit.qasm）；
4. **从文件重新读回**（qasm3.load / QuantumCircuit.from_qasm_file），再模拟；
5. 校验三件事：峰值比特串不变、δ 一致、整个态矢量在比特反序下逐幅一致（到全局相位）。

为什么校验用"全零比特串的振幅"
------------------------------
qiskit 的 qubit 0 是态矢量最低位，本仓库 site 0 是最高位，两者互为反序。
|0…0⟩ 在两种约定下是同一个基矢，所以 δ = |⟨0…0|ψ⟩|² 可以直接比，与约定无关；
再额外做一次比特反序的整矢量比对，把"只有 δ 对、别处错"的情况也堵掉。

用法
----
    python exp_qasm_roundtrip.py                       # 默认 n12_d8（176 门）
    python exp_qasm_roundtrip.py --cases n12_d8_t015_s0,n12_d20_t015_s0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
from qiskit import QuantumCircuit, transpile, qasm2, qasm3
from qiskit.quantum_info import Statevector

from exp_mps_break import load_circuits, sv_gates_to_state
from exp_pps import x_mask_gates

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "autotest_results", "p3_qasm")


def build_circuit(n, gates, mask_bits):
    """把 (i, j, U4) 门表变成 qiskit 电路；qubit i 是这一对里更靠前的那个。

    mask_bits（长度 n 的 0/1 列表）：最后补单比特 X，把峰值搬到该比特串上。
    """
    qc = QuantumCircuit(n)
    for i, j, u in gates:
        # qiskit 的 unitary(U, [a, b]) 把列表**第一个**比特当最低位；
        # 本仓库的 U 以 site i 为最高位，所以这里要写成 [j, i]。
        qc.unitary(np.asarray(u, dtype=complex), [j, i])
    for q, b in enumerate(mask_bits):
        if b:
            qc.x(q)
    return qc


def amp_allzero(circ):
    """⟨0…0|ψ⟩（与比特序约定无关）。"""
    sv = Statevector.from_instruction(circ)
    return complex(sv.data[0])


def prob_list(circ):
    sv = Statevector.from_instruction(circ)
    p = np.abs(sv.data) ** 2
    return p / p.sum()


def bitrev_perm(n):
    """把 qiskit 的位序（qubit 0 为最低位）换成仓库的位序（site 0 为最高位）。"""
    idx = np.arange(2 ** n)
    rev = np.array([int(format(x, f"0{n}b")[::-1], 2) for x in idx])
    return idx[rev]


def run_case(rec, mask_seed, do_plot=True):
    n, gates = rec["n"], rec["_gates"]
    name = rec["_file"].replace(".json", "")
    os.makedirs(OUT, exist_ok=True)

    # 随机秘密串：峰值搬到 s 上，避免"答案恒为全零"这种非真实设定
    rng = np.random.default_rng(mask_seed)
    bits = [int(b) for b in rng.integers(0, 2, n)]
    secret = "".join(str(b) for b in bits)
    # 参考引擎用 2-qubit 形式的 X（借相邻站），与 qiskit 的单比特 X 等价
    ref_gates = gates + x_mask_gates(n, bits)

    print(f"\n=== {name}  n={n} 门数={len(gates)} δ_json={rec['delta']:.4f} "
          f"secret={secret} ===")

    # 原始（本仓库自己的引擎）
    psi_ref = sv_gates_to_state(n, ref_gates).reshape(-1)
    ref_idx = int(secret, 2)           # 仓库位序：site0 是字符串最高位
    d_ref = abs(psi_ref[ref_idx]) ** 2

    t0 = time.time()
    qc = build_circuit(n, gates, bits)
    # QASM3 用 u/cx（u 是 QASM3 原生门）；QASM2 用 u3/cx（qelib1.inc 里有 u3，平台普遍吃这个）
    tqc = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
    tqc2 = transpile(qc, basis_gates=["u3", "cx"], optimization_level=0)
    print(f"  转译：{len(gates)} 个 UnitaryGate -> QASM3 {tqc.count_ops()} / "
          f"QASM2 {tqc2.count_ops()}（{time.time()-t0:.1f}s）")

    q3 = os.path.join(OUT, f"{name}.qasm3")
    q2 = os.path.join(OUT, f"{name}.qasm")
    # qiskit 2.6 的 qasm3.dump 要文件对象而不是路径，这里统一走字符串
    q3_text = qasm3.dumps(tqc)
    with open(q3, "w", encoding="utf-8") as f:
        f.write(q3_text)
    q2_text = qasm2.dumps(tqc2)
    with open(q2, "w", encoding="utf-8") as f:
        f.write(q2_text)

    # 从文件读回
    t1 = time.time()
    back3 = qasm3.loads(q3_text)
    back2 = qasm2.loads(q2_text)
    print(f"  读回：QASM3 {back3.num_qubits}q / QASM2 {back2.num_qubits}q"
          f"（{time.time()-t1:.1f}s）")

    # 校验 A：秘密串上的振幅一致（相位对齐后再比）
    sv3 = Statevector.from_instruction(back3).data[bitrev_perm(n)]
    sv2 = Statevector.from_instruction(back2).data[bitrev_perm(n)]
    phase = sv3[ref_idx] / psi_ref[ref_idx]
    a3, a2 = sv3[ref_idx] / phase, sv2[ref_idx] / phase
    d3, d2 = abs(sv3[ref_idx]) ** 2, abs(sv2[ref_idx]) ** 2
    err_amp3 = abs(a3 - psi_ref[ref_idx])
    err_amp2 = abs(a2 - psi_ref[ref_idx])
    print(f"  δ: 引擎={d_ref:.6f}  QASM3={d3:.6f}  QASM2={d2:.6f}"
          f"   |Δamp|(扣相位)={err_amp3:.2e}/{err_amp2:.2e}")

    # 校验 B：整条态矢量在比特反序下逐幅一致（扣全局相位）
    err_full = float(np.max(np.abs(sv3 / phase - psi_ref)))
    print(f"  整矢量逐幅最大偏差（扣全局相位）= {err_full:.2e}")

    # 校验 C：峰值比特串与背景
    p = prob_list(back3)
    top = int(np.argmax(p))
    top_bits = format(top, f"0{n}b")[::-1]      # 换回仓库位序的串
    uniform = 1.0 / 2 ** n
    ratio = p[top] / uniform
    print(f"  argmax 串={top_bits}  概率={p[top]:.4f}  "
          f"{'= secret ✓' if top_bits == secret else '≠ secret ✗'}  "
          f"均匀背景={uniform:.2e}  增强={ratio:.0f}×")

    ok = (abs(d3 - d_ref) < 1e-9 and abs(d2 - d_ref) < 1e-9 and err_full < 1e-9)
    print(f"  判定：{'✓ 闭环通过' if ok else '✗ 不一致'}")

    if do_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            order = np.argsort(p)[::-1][:24]
            fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
            ax.bar(np.arange(len(order)), p[order], color="#4C78A8")
            ax.axhline(uniform, ls="--", lw=1, color="crimson",
                       label=f"uniform $2^{{-{n}}}$ = {uniform:.1e}")
            ax.set_yscale("log")
            ax.set_xlabel("basis state (rank by probability)")
            ax.set_ylabel("probability")
            ax.set_title(f"QASM round-trip: peak $\\delta$={p[top]:.3f}, "
                         f"{ratio:.0f}$\\times$ over uniform")
            ax.legend(fontsize=7)
            ax.grid(True, axis="y", which="both", alpha=0.3)
            fig.tight_layout()
            png = os.path.join(OUT, f"{name}_spectrum.png")
            fig.savefig(png)
            plt.close(fig)
            print(f"  图：{png}")
        except Exception as e:
            print(f"  （绘图失败: {e}）")

    return {
        "case": name, "n": n, "gates_in": len(gates),
        "gates_transpiled": sum(tqc.count_ops().values()),
        "ops": dict(tqc.count_ops()),
        "delta_engine": d_ref, "delta_qasm3": d3, "delta_qasm2": d2,
        "err_amp_qasm3": err_amp3, "err_amp_qasm2": err_amp2,
        "err_full_vector": err_full,
        "secret": secret, "peak_bits": top_bits, "peak_prob": float(p[top]),
        "uniform": uniform, "enhancement": float(ratio),
        "qasm3_bytes": os.path.getsize(q3), "qasm2_bytes": os.path.getsize(q2),
        "ok": bool(ok),
    }


def main():
    ap = argparse.ArgumentParser(description="QASM 导出与闭环校验")
    ap.add_argument("--mask-seed", type=int, default=2026,
                    help="随机秘密串种子（峰值搬到该串上）")
    ap.add_argument("--cases", type=str, default="n12_d8_t015_s0",
                    help="电路文件名前缀，逗号分隔；默认取 176 门那个")
    args = ap.parse_args()
    want = [x.strip() for x in args.cases.split(",")]
    circs = [c for c in load_circuits() if any(c["_file"].startswith(w) for w in want)]
    if not circs:
        raise SystemExit(f"没找到电路：{want}")
    rows = [run_case(c, args.mask_seed) for c in circs]
    path = os.path.join(OUT, "roundtrip.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n写出 {path}")
    print("\n| 电路 | 输入门数 | 转译后门数 | δ(引擎) | δ(QASM3) | δ(QASM2) | 整矢量偏差 | 峰值增强 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['case']} | {r['gates_in']} | {r['gates_transpiled']} "
              f"| {r['delta_engine']:.4f} | {r['delta_qasm3']:.4f} | {r['delta_qasm2']:.4f} "
              f"| {r['err_full_vector']:.1e} | {r['enhancement']:.0f}× |")


if __name__ == "__main__":
    main()
