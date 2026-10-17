# -*- coding: utf-8 -*-
"""Paper-3 (HQAP) environment smoke test.

Checks that the `hqap` conda environment can (a) convert Qiskit circuits to
quimb tensor networks and (b) import the internal compression routine that the
Kremer-Dupuis reference code relies on.

Usage (any directory):
    python smoke_test_p3.py

Exit code 0 = environment usable. Non-zero = something below failed.
"""
import sys
import warnings


def sec(title):
    print()
    print("=" * 62)
    print(title)
    print("=" * 62)


def main():
    fails = []

    # ---------------- 1. versions ----------------
    sec("[1/4] 版本清单")
    print(f"  python       : {sys.version.split()[0]}")
    try:
        import numpy
        print(f"  numpy        : {numpy.__version__}")
    except Exception as e:  # pragma: no cover
        fails.append(("numpy", e))
    try:
        import scipy
        print(f"  scipy        : {scipy.__version__}")
    except Exception as e:  # pragma: no cover
        fails.append(("scipy", e))
    try:
        import quimb
        print(f"  quimb        : {quimb.__version__}")
    except Exception as e:
        fails.append(("quimb import", e))
    try:
        import qiskit
        print(f"  qiskit       : {qiskit.__version__}")
    except Exception as e:
        fails.append(("qiskit import", e))
    try:
        import qiskit_quimb
        ver = getattr(qiskit_quimb, "__version__", "n/a")
        print(f"  qiskit_quimb : {ver}")
    except Exception as e:
        fails.append(("qiskit_quimb import", e))
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  torch        : {torch.__version__}  (CUDA: {cuda})")
        if cuda:
            print("                 -> GPU 可用；Kremer 的 GPU notebook 也能跑")
        else:
            print("                 -> 仅 CPU；Kremer 用的 CPU notebook 正合适")
    except Exception as e:
        fails.append(("torch import", e))

    # ---------------- 2. qiskit -> quimb ----------------
    sec("[2/4] Qiskit 电路 -> quimb 张量网络（Bell 态采样）")
    try:
        import quimb.tensor as qtn
        from qiskit import QuantumCircuit
        from qiskit_quimb import quimb_circuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        tn_circ = quimb_circuit(qc)
        samples = list(tn_circ.sample(20, seed=7))
        only = set(samples)
        ok = only <= {"00", "11"} and len(samples) == 20
        print(f"  采样 20 次 -> {sorted(only)}  ({'通过' if ok else '异常'})")
        if not ok:
            fails.append(("bell sampling", ValueError(f"got {only}")))

        # MPS 后端（论文 §MPS 一节的工作方式）
        tn_mps = quimb_circuit(qc, quimb_circuit_class=qtn.CircuitMPS, max_bond=8)
        mps_samples = list(tn_mps.sample(5, seed=7))
        print(f"  MPS 后端 max_bond=8 -> {mps_samples}")
    except Exception as e:
        fails.append(("qiskit->quimb", e))
        print(f"  失败: {type(e).__name__}: {e}")

    # ---------------- 3. Kremer 依赖的内部 API ----------------
    sec("[3/4] Kremer 参考代码依赖的内部压缩例程")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from quimb.tensor.tensor_arbgeom_compress import (
                tensor_network_ag_compress,
            )
        import inspect
        params = list(inspect.signature(tensor_network_ag_compress).parameters)
        need = {"max_bond", "cutoff", "method", "site_tags", "canonize", "equalize_norms"}
        missing = need - set(params)
        print(f"  导入成功，签名参数 {len(params)} 个")
        print(f"  Kremer 用到的关键字: {sorted(need)}")
        print(f"  缺失: {sorted(missing) if missing else '无'}")
        if missing:
            fails.append(("compress signature", KeyError(sorted(missing))))
    except Exception as e:
        fails.append(("quimb internal compress", e))
        print(f"  失败: {type(e).__name__}: {e}")

    # ---------------- 4. QASM 读取能力 ----------------
    sec("[4/4] QASM 读取（Peak Portal 下载的电路用这个入口）")
    try:
        from qiskit import QuantumCircuit
        qasm = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[3];\n"
            "creg c[3];\n"
            "h q[0];\n"
            "cx q[0],q[1];\n"
            "cx q[1],q[2];\n"
            "rz(0.3) q[2];\n"
        )
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".qasm")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(qasm)
            qc = QuantumCircuit.from_qasm_file(path)
            print(f"  读入 {qc.num_qubits} qubits, 深度 {qc.depth()}, 门数 {qc.size()}")
        finally:
            os.remove(path)
    except Exception as e:
        fails.append(("qasm read", e))
        print(f"  失败: {type(e).__name__}: {e}")

    # ---------------- summary ----------------
    sec("结果")
    if fails:
        for name, err in fails:
            print(f"  FAIL  {name}: {type(err).__name__}: {err}")
        print(f"\n  {len(fails)} 项失败 —— 环境不可用。")
        return 1
    print("  全部通过 —— 环境可用，可以进入 L1/L2/L3。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
