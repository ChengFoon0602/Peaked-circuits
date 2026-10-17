# -*- coding: utf-8 -*-
"""Paper-3 (HQAP): run the Kremer-Dupuis CPU pipeline on a public QASM circuit.

This wraps ``utils.contract_core -> tno_to_tne -> extract_bitstring`` from
https://github.com/d-kremer/peaked-circuit-simulation so it can be run from the
command line with reproducible settings.

Windows note: cotengra's hyper-optimiser spawns worker processes, which requires
a real script file plus the ``if __name__ == "__main__"`` guard below. Running
this logic through a heredoc / stdin will crash with BrokenProcessPool.

Usage (hqap environment, this repo's root):
    set KREMER_HOME=D:\\桌面文件\\大三\\peaked-circuit-simulation
    python run_kremer_cpu.py --qasm D:\\桌面文件\\大三\\peaked_qasm\\peaked_circuit_heavy_hex_49x5072.qasm
"""
import argparse
import csv
import json
import os
import sys
import time

KREMER_HOME = os.environ.get(
    "KREMER_HOME", r"D:\桌面文件\大三\peaked-circuit-simulation"
)


def load_circuit(path):
    """Load a QASM file, dispatching on the version declared in the header."""
    from qiskit import QuantumCircuit

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(200)
    if "OPENQASM 3" in head:
        from qiskit import qasm3
        return qasm3.load(path), "3.0"
    return QuantumCircuit.from_qasm_file(path), "2.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qasm", required=True, help="QASM 文件路径")
    ap.add_argument("--chunk-size", type=int, default=2)
    ap.add_argument("--max-bond", type=int, default=16)
    ap.add_argument("--cutoff", type=float, default=0.01)
    ap.add_argument("--tne-max-bond", type=int, default=8)
    ap.add_argument("--outdir", default="autotest_results/p3_kremer_cpu")
    ap.add_argument("--tag", default=None, help="输出文件名标签（默认取 qasm 主名）")
    args = ap.parse_args()

    if KREMER_HOME not in sys.path:
        sys.path.insert(0, KREMER_HOME)
    from utils import contract_core, tno_to_tne, extract_bitstring, iter_layers

    stem = args.tag or os.path.splitext(os.path.basename(args.qasm))[0]
    os.makedirs(args.outdir, exist_ok=True)

    t_start = time.perf_counter()
    print("=" * 68)
    print(f"Kremer CPU pipeline  |  {stem}")
    print(f"chunk={args.chunk_size}  max_bond={args.max_bond}  "
          f"cutoff={args.cutoff}  tne_max_bond={args.tne_max_bond}")
    print("=" * 68)

    qc, version = load_circuit(args.qasm)
    t_load = time.perf_counter() - t_start
    print(f"[load] {qc.num_qubits} qubits, {qc.size()} gates, "
          f"depth {qc.depth()}  (QASM {version})  {t_load:.1f}s", flush=True)

    t0 = time.perf_counter()
    layers = list(iter_layers(qc))
    print(f"[layers] {len(layers)} 层  {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    tno, stats = contract_core(
        layers,
        chunk_size=args.chunk_size,
        cutoff=args.cutoff,
        max_bond=args.max_bond,
        to_backend=None,          # CPU: Kremer's default is cuda:0
    )
    t_contract = time.perf_counter() - t1
    print(f"[contract_core] {t_contract:.1f}s", flush=True)

    t2 = time.perf_counter()
    tne = tno_to_tne(tno, max_bond=args.tne_max_bond, cutoff=args.cutoff,
                     to_backend=None)
    t_tne = time.perf_counter() - t2
    print(f"[tno_to_tne] {t_tne:.1f}s", flush=True)

    t3 = time.perf_counter()
    bs, marginals = extract_bitstring(tne)
    t_extract = time.perf_counter() - t3
    print(f"[extract_bitstring] {t_extract:.1f}s", flush=True)

    total = time.perf_counter() - t_start

    # ---- persist ----
    row = {
        "circuit": stem,
        "n_qubits": qc.num_qubits,
        "n_gates": qc.size(),
        "depth": qc.depth(),
        "qasm_version": version,
        "chunk_size": args.chunk_size,
        "max_bond": args.max_bond,
        "cutoff": args.cutoff,
        "tne_max_bond": args.tne_max_bond,
        "bitstring": bs,
        "bitstring_reversed": bs[::-1],
        "load_s": round(t_load, 2),
        "contract_s": round(t_contract, 2),
        "tne_s": round(t_tne, 2),
        "extract_s": round(t_extract, 2),
        "total_s": round(total, 2),
    }
    with open(os.path.join(args.outdir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    csv_path = os.path.join(args.outdir, "runs.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)

    print()
    print(f"[RESULT] bitstring          = {bs}")
    print(f"[RESULT] bitstring reversed = {bs[::-1]}")
    print(f"[total] {total:.1f}s   -> {args.outdir}/{stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
