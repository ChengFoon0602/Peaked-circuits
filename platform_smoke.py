# -*- coding: utf-8 -*-
"""platform_smoke.py — BlueQubit 平台的冒烟测试与对照实验

设计原则：**默认不花额度**。只有显式加 `--run` 才会真正提交作业；默认只做
本地构造 + `estimate()`（平台给的运行时长/额度估算）。

三个阶段
--------
    stage 1   Bell 态：用 cpu 设备跑，取 statevector 与手算值对照（验证 token/网络/SDK）
    stage 2   我们导出的 12 比特 QASM：上传 → 跑 → 确认 argmax 等于我们的秘密串
    stage 3   公开挑战电路（49/56 比特）：默认只 estimate；`--run` 才真跑

用法
----
    set BLUEQUBIT_API_TOKEN=你的token
    python platform_smoke.py --stage 1
    python platform_smoke.py --stage 2
    python platform_smoke.py --stage 3 --qasm D:/桌面文件/大三/peaked_qasm/peaked_circuit_P9_Hqap_56x1917.qasm --device mps.cpu

注意
----
* `device` 只允许 cpu / mps.cpu / pauli-path。`gpu`、`mps.gpu`、`quantum` 需要余额或
  真机排队，本脚本直接拒绝（要跑请手工调用，别误触）。
* 平台侧 `get_counts()` 的比特序是 qiskit 约定（qubit 0 在最右），与仓库约定相反，
  所以比对时两种顺序都试，并打印哪一种对上。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import qiskit
from qiskit import qasm2, qasm3, QuantumCircuit

SAFE_DEVICES = {"cpu", "mps.cpu", "pauli-path"}


def load_qasm(path):
    """自动识别 OpenQASM 2.0 / 3.0。"""
    with open(path, encoding="utf-8") as f:
        head = f.readline()
    if "3.0" in head or "3;" in head:
        return qasm3.load(path), "3.0"
    return qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS), "2.0"


def stage1(bq):
    print("\n=== stage 1 · Bell 态 statevector 对照 ===")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    res = bq.run(qc, device="cpu", job_name="smoke_bell")
    sv = np.asarray(res.get_statevector())
    want = np.array([1, 0, 0, 1]) / np.sqrt(2)
    print("  平台返回:", np.round(sv, 4))
    print("  期望    :", np.round(want, 4))
    print(f"  最大偏差 = {np.max(np.abs(sv - want)):.2e}  "
          f"{'✓' if np.max(np.abs(sv - want)) < 1e-6 else '✗'}")


def stage2(bq, qasm_path, shots):
    print(f"\n=== stage 2 · 上传我们的 QASM：{os.path.basename(qasm_path)} ===")
    qc, ver = load_qasm(qasm_path)
    print(f"  读入：{qc.num_qubits} 比特 / {sum(qc.count_ops().values())} 门（QASM {ver}）")
    res = bq.run(qc, device="cpu", shots=shots, job_name="smoke_our_peak")
    counts = res.get_counts()
    top = max(counts, key=counts.get)
    p_top = counts[top] / sum(counts.values())
    print(f"  平台 argmax = {top}（{p_top:.3f}，共 {sum(counts.values())} shots）")

    ref = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "autotest_results", "p3_qasm", "roundtrip.json")
    if os.path.exists(ref):
        with open(ref, encoding="utf-8") as f:
            rows = json.load(f)
        secret = rows[0]["secret"]
        match_fwd = (top == secret)
        match_rev = (top[::-1] == secret)
        print(f"  我们的秘密串   = {secret}")
        print(f"  直接比对       = {'✓' if match_fwd else '✗'}；"
              f"反转后比对 = {'✓' if match_rev else '✗'}"
              f"（{'平台与仓库位序一致' if match_fwd else '平台与仓库位序相反，正常'}）")
        ok = match_fwd or match_rev
        print(f"  判定：{'✓ 平台读入与执行一致' if ok else '✗ 不一致，需排查'}")
        return ok
    print("  （没找到 roundtrip.json，跳过秘密串比对）")
    return None


def stage3(bq, qasm_path, device, do_run, options):
    print(f"\n=== stage 3 · 公开电路：{os.path.basename(qasm_path)} · device={device} ===")
    qc, ver = load_qasm(qasm_path)
    print(f"  读入：{qc.num_qubits} 比特 / {sum(qc.count_ops().values())} 门（QASM {ver}）")
    try:
        est = bq.estimate(qc, device=device)
        print(f"  estimate：{est}")
    except Exception as e:
        print(f"  estimate 失败（{type(e).__name__}: {e}），跳过")
    if not do_run:
        print("  （未加 --run，只做估算，不提交作业）")
        return
    res = bq.run(qc, device=device, job_name="public_attack", options=options or None)
    print("  作业已提交，等待结果 ...")
    counts = res.get_counts()
    top = max(counts, key=counts.get)
    print(f"  argmax = {top}  概率 = {counts[top] / sum(counts.values()):.4f}")
    return top


def main():
    ap = argparse.ArgumentParser(description="BlueQubit 平台冒烟")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--token", type=str, default=None,
                    help="不填则读环境变量 BLUEQUBIT_API_TOKEN")
    ap.add_argument("--qasm", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--run", action="store_true", help="stage 3 才需要：真正提交作业")
    args = ap.parse_args()

    if args.device not in SAFE_DEVICES:
        sys.exit(f"拒绝在 {args.device} 上跑：本脚本只允许 {sorted(SAFE_DEVICES)}，"
                 f"真机/GPU 请手工调用以免误触额度。")

    token = args.token or os.environ.get("BLUEQUBIT_API_TOKEN")
    if not token:
        sys.exit("缺少 token：设置环境变量 BLUEQUBIT_API_TOKEN，或用 --token 传入。")
    import bluequbit
    bq = bluequbit.init(token)
    print(f"BlueQubit SDK {getattr(bluequbit, '__version__', '?')} 已初始化")

    if args.stage == 1:
        stage1(bq)
    elif args.stage == 2:
        path = args.qasm or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "autotest_results", "p3_qasm", "n12_d8_t015_s0_G88.qasm")
        stage2(bq, path, args.shots)
    else:
        if not args.qasm:
            sys.exit("stage 3 需要 --qasm 指向一个公开电路文件。")
        opts = {}
        if args.device == "mps.cpu":
            opts["mps_bond_dimension"] = 16
        if args.device == "pauli-path":
            opts["pauli_path_truncation_threshold"] = 1e-2
        stage3(bq, args.qasm, args.device, args.run, opts)


if __name__ == "__main__":
    main()
