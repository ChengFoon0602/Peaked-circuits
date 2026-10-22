# -*- coding: utf-8 -*-
"""platform_smoke.py — BlueQubit 平台的冒烟测试与对照实验

设计原则：**默认不花额度**。cpu / mps.cpu / pauli-path 三个模拟器都是免费的（有日配额），
gpu / mps.gpu / quantum 计费，本脚本直接拒绝。stage 3 不加 `--run` 不提交作业。

四个阶段
--------
    stage 0   抓官方 peaked circuit（`bq.get_peaked_circuit`）：拿到 QASM + 真值 key，不耗作业
    stage 1   Bell 态：用 cpu 设备跑，取 statevector 与手算值对照（验证 token/网络/SDK）
    stage 2   我们导出的 12 比特 QASM：上传 → 跑 → 确认 argmax 等于我们的秘密串
    stage 3   公开挑战电路（49/56 比特）：默认只打印参数不提交；`--run` 才真跑

用法
----
    set BLUEQUBIT_API_TOKEN=你的token
    python experiments/platform/platform_smoke.py --stage 0 --difficulty 40
    python experiments/platform/platform_smoke.py --stage 1
    python experiments/platform/platform_smoke.py --stage 2
    python experiments/platform/platform_smoke.py --stage 3 --qasm data/qasm/peaked_circuit_P9_Hqap_56x1917.qasm --device mps.cpu

注意
----
* `device` 只允许 cpu / mps.cpu / pauli-path。`gpu`、`mps.gpu`、`quantum` 需要余额或
  真机排队，本脚本直接拒绝（要跑请手工调用，别误触）。
* 设备名的坑：官方文档表格写 `pauli-path.cpu`，但 SDK 的 DEVICE_TYPES 里只有
  `pauli-path` / `pauli-path.gpu`——传 `pauli-path.cpu` 会被 validate_device 拒绝。
* `bq.estimate()` **只支持 cpu / gpu / quantum**（不支持 mps.cpu、pauli-path）。
  后两个本来就免费，不需要估算。
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

DEVICE_ALIASES = {
    # 同一个设备在不同地方有三种写法：
    #   平台定价页 / UI  -> "pps.cpu"（PPS = Pauli Path Simulation）
    #   官方文档表格      -> "pauli-path.cpu"（这个 SDK 反而会拒绝）
    #   SDK DEVICE_TYPES -> "pauli-path"（唯一能通过 validate_device 的）
    "pps": "pauli-path",
    "pps.cpu": "pauli-path",
    "pauli-path.cpu": "pauli-path",
    "pauli-path.gpu": "pauli-path.gpu",
}

SAFE_DEVICES = {"cpu", "mps.cpu", "pauli-path"}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # → 仓库根
RESULTS = os.path.join(ROOT, "autotest_results")


def load_qasm(path):
    """自动识别 OpenQASM 2.0 / 3.0。"""
    with open(path, encoding="utf-8") as f:
        head = f.readline()
    if "3.0" in head or "3;" in head:
        return qasm3.load(path), "3.0"
    return qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS), "2.0"


def stage0(bq, difficulties, circuit_id, out_dir):
    """抓官方 peaked circuit：平台直接给 QASM 与真值 key，不用去网页下载。"""
    os.makedirs(out_dir, exist_ok=True)
    for d in difficulties:
        print(f"\n=== stage 0 · get_peaked_circuit(difficulty={d}, id={circuit_id}) ===")
        try:
            pc = bq.get_peaked_circuit(d, circuit_id)
        except Exception as e:
            msg = str(e)
            print(f"  失败：{type(e).__name__}: {msg}")
            if "403" in msg or "Access denied" in msg:
                print("  → 这是 403，不是 401：token 已被接受，是**端点级别的拒绝**。")
                print("    差别很重要——401 才说明 token 错。先用 `--stage 1` 或")
                print("    `python experiments/platform/bq_diag.py` 确认 /jobs 能通，再回来查这个端点。")
                print("    可能原因：账号未加入对应的挑战、套餐不含该功能、或该端点尚未开放。")
            elif "401" in msg or "Unauthorized" in msg:
                print("  → 401：token 无效。重新复制（注意首尾空格），重设后重开终端。")
            else:
                print("  → 参考 `python experiments/platform/bq_diag.py` 的逐条探测结果。")
            print("  不依赖它的替代方案：本地已有 3 个公开电路，其中两个 49 比特电路")
            print("  已有与作者输出逐位一致的恢复串（autotest_results/p3_kremer_cpu/）。")
            continue
        n = pc.circuit.num_qubits
        nops = sum(pc.circuit.count_ops().values())
        print(f"  {pc!r}")
        print(f"  规模：{n} 比特 / {nops} 门 | key 长度 = {len(pc.key)} | QASM {len(pc.qasm_str)} 字节")
        print(f"  key = {pc.key}")
        print(f"  metadata = {pc.metadata}")
        tag = f"difficulty{d}_id{circuit_id if circuit_id is not None else 'rand'}"
        path = os.path.join(out_dir, f"official_{tag}.qasm")
        with open(path, "w", encoding="utf-8") as f:
            f.write(pc.qasm_str)
        with open(os.path.join(out_dir, f"official_{tag}.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"difficulty": d, "circuit_id": circuit_id,
                       "num_qubits": n, "num_gates": nops,
                       "key": pc.key, "metadata": pc.metadata}, f,
                      ensure_ascii=False, indent=2)
        print(f"  已存：{path}")


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

    ref = os.path.join(RESULTS, "p3_qasm", "roundtrip.json")
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
    ap.add_argument("--stage", type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument("--token", type=str, default=None,
                    help="不填则读环境变量 BLUEQUBIT_API_TOKEN")
    ap.add_argument("--qasm", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--difficulty", type=str, default="40",
                    help="stage 0：难度（大概率是比特数），可逗号分隔试多个")
    ap.add_argument("--circuit-id", type=int, default=None,
                    help="stage 0：不填则平台随机给一个该难度的电路")
    ap.add_argument("--run", action="store_true", help="stage 3 才需要：真正提交作业")
    args = ap.parse_args()

    args.device = DEVICE_ALIASES.get(args.device.lower(), args.device.lower())
    if args.device not in SAFE_DEVICES:
        sys.exit(f"拒绝在 {args.device} 上跑：本脚本只允许 {sorted(SAFE_DEVICES)}，"
                 f"真机/GPU 请手工调用以免误触额度。")

    token = args.token or os.environ.get("BLUEQUBIT_API_TOKEN")
    if not token:
        sys.exit("缺少 token：设置环境变量 BLUEQUBIT_API_TOKEN，或用 --token 传入。")
    import bluequbit
    bq = bluequbit.init(token)
    print(f"BlueQubit SDK {getattr(bluequbit, '__version__', '?')} 已初始化")

    if args.stage == 0:
        diffs = [int(x) for x in args.difficulty.split(",") if x.strip()]
        stage0(bq, diffs, args.circuit_id, os.path.join(RESULTS, "p3_platform"))
    elif args.stage == 1:
        stage1(bq)
    elif args.stage == 2:
        path = args.qasm or os.path.join(
            RESULTS, "p3_qasm", "n12_d8_t015_s0_G88.qasm")
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
