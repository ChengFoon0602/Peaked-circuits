# Peaked-circuits · 复现学习记录

> **原始仓库**：https://github.com/yuxuanzhang1995/Peaked-circuits  （作者：Yuxuan Zhang，与 Scott Aaronson 合作）
> **论文**：*On verifiable quantum advantage with peaked circuit sampling* — [arXiv:2404.14493](https://arxiv.org/abs/2404.14493)
> **第二篇（同系列）**：*Complexity and hardness of random peaked circuits* — [arXiv:2510.00132](https://arxiv.org/abs/2510.00132)（2026-10-15 完成数值复现，见「第二篇复现」节）
> 本仓库为个人复现学习：保留原始代码（`circuits.py`、`functions.py`、两个 notebook 等），新增运行记录、环境自检与自动化实验脚本。**原始算法代码与论文版权归原作者所有。**

---

## 复现结果（2026-10-13）

**设置**：n = 12 qubits，随机层深度 τr = 12，尖峰层 τp 扫描 {3, 6, 9, 12}，每档 5 个随机实例（固定种子，可复现）。

| τp | 训练 δ（均值 ± 标准差） | 随机基线 | 增强倍数 |
|---|---|---|---|
| 3 | 0.155 ± 0.044 | 0.0117 | 13× |
| 6 | 0.343 ± 0.081 | 0.0056 | 61× |
| 9 | 0.459 ± 0.053 | 0.0038 | 121× |
| 12 | **0.484 ± 0.079** | 0.0032 | **151×** |

![sweep plot](autotest_results/run_20261013_221358/sweep_plot.png)

- δ = |⟨0|C′†C|0⟩|²，即"尖峰比特串"上的概率集中度（训练后峰值权重）；
- 单实例基准（官方 notebook 原参数，τp = 6）：δ ≈ 0.455，CPU 上约 1 分钟；
- **结论**：训练尖峰层显著提升输出集中度（约为同深度随机电路基线的 13–151 倍），趋势与论文一致（详见下一节对照说明）。

## 与论文的对照说明

**论文的关键数字**（arXiv:2404.14493）：
- τr = 50、τp = 8、n = 10/12（100 实例平均）：平均峰值权重 **δ > 0.15**；τr = 40、τp = 10 → δ ≈ 0.2；
- τr = n、τp = n/2 时，δ 随 n 按 **≈ 1.189⁻ⁿ** 衰减（远慢于随机标度 2⁻ⁿ）；
- 论文的另一条对照结论：优化后的尖峰层**不是**"后 τp 层随机门的平凡逆"（其效果远超该基线，见论文 Fig. 3c）。

**"平凡反演"对照（本仓库整理）**：若把尖峰层取为后 τp 层随机门的逆，则整体电路等效于深度 τr−τp 的随机电路；其平均峰值权重可直接从 `baseline.csv` 按深度 τr−τp 读出（与论文 Fig. 3c 的 dashed "base" 对照同款逻辑）：

| τp | 训练 δ | "平凡反演"等效基线 | 倍数 |
|---|---|---|---|
| 3 | 0.155 ± 0.044 | 0.0038（深度 9 基线） | 40.8× |
| 6 | 0.343 ± 0.081 | 0.0056（深度 6 基线） | 61.0× |
| 9 | 0.459 ± 0.053 | 0.0117（深度 3 基线） | 39.2× |
| 12 | 0.484 ± 0.079 | —（τp = τr 时平凡反演即恒等，不适用） | — |

**结论：优化后的 δ 高出该基线约 40–60 倍——训练出的尖峰层绝非"平凡反演"，与论文一致。**

**参数区差异说明**：本节（v0.1）取 τr = 12（比论文更浅、更易），δ 数值不能与论文直接比大小；**τr = 50 的忠实对照已于 2026-10-14 完成，见下节「深入复现」D1**（该节 δ(τp=8)=0.1545 ± 0.0035 与论文 ">0.15" 直接吻合）。

## 深入复现（2026-10-14）

在 v0.1（τr=12）基础上补齐三个实验：**忠实参数区（τr=50）**、**标度律**、**多起点优化**。全部固定随机种子、可复现。

### D1 · τr = 50 忠实区间（对齐论文 Fig. 3a）

设置：n = 12、τr = 50、每档 10 个实例（论文为 100 个）。

| τp | 训练 δ（均值 ± 标准差） | 随机基线 | 增强 |
|---|---|---|---|
| 4 | 0.0662 ± 0.0019 | 0.0083 | 8.0× |
| 8 | **0.1545 ± 0.0035** | 0.0044 | 34.7× |
| 12 | 0.2380 ± 0.0029 | 0.0032 | 74.1× |
| 16 | 0.3114 ± 0.0036 | 0.0025 | 123.5× |

**与论文对照：论文报告 τr=50、τp=8 时平均峰值权重 δ > 0.15；本复现实测 0.1545 ± 0.0035 —— 吻合。**

![d1 sweep](autotest_results/d1_taur50/sweep_plot.png)

### D2 · 标度律（对齐论文 Fig. 3c / Conjecture 3.1）

设置：τr = n、τp = n/2，n = 8/10/12，每档 8 个实例。

| n | 训练 δ（均值 ± 标准差） |
|---|---|
| 8 | 0.6687 ± 0.0368 |
| 10 | 0.4681 ± 0.0469 |
| 12 | 0.3545 ± 0.0660 |

**拟合：δ ∝ 1.172⁻ⁿ；论文报告 ≈ 1.189⁻ⁿ —— 高度一致（相对偏差约 1.4%），且远慢于随机标度 2⁻ⁿ。**

![d2 scaling](autotest_results/d2_scaling/scaling_plot.png)

### D3 · 多起点优化（论文未涉及，补充研究）

每个实例固定同一随机电路（rqc），用 3 个随机初值训练尖峰层并取最优：

| τp | 单起点 δ（均值） | 3 起点取优 δ（均值） | 增益 |
|---|---|---|---|
| 6 | 0.3428 | 0.3538 | +3.2% |
| 9 | 0.4586 | 0.4850 | +5.7% |

平均增益 ≈ **+4.7%**（个别实例最高 +14.3%；10 个实例中 8 个受益）。说明单起点已接近该构造的局部极限，多起点收益有限——与论文关于 barren plateau / 优化质量的讨论一致。

### 结论汇总（对照论文核心主张）

| 论文主张 | 本仓库验证 |
|---|---|
| 少量尖峰层即可产生非平凡 δ（τp ≪ τr） | ✅ D1：τr=50、τp=8 时 δ=0.1545 |
| δ 随 n 指数衰减，远慢于随机标度（≈1.189⁻ⁿ） | ✅ D2：拟合 1.172⁻ⁿ |
| 优化尖峰层 ≠ "平凡反演" | ✅ v0.1：超平凡反演基线 40–61×（τr=12） |

## 第二篇复现（2026-10-15）· Complexity and hardness of random peaked circuits

**对象**：Zhang, *Complexity and hardness of random peaked circuits* — [arXiv:2510.00132](https://arxiv.org/abs/2510.00132)（独著）。第二篇以理论为主（#P 硬度 / PromiseBQP / QCMA 引理等，不做数值复现）；以下为 §4 可数值核对的三个实验，均固定种子、可复现。

### E3 · Postselection 稀有性（Lemma 4.1）

Monte Carlo 核对 $\Pr[|\langle v|\psi\rangle|^2 \ge \delta] = (1-\delta)^{d-1}$：d = 4–256 全部数据点与理论在 95% 置信区间内一致。产出：`autotest_results/e3_acceptance/`。

### E1 · 噪声鲁棒解码（§4.4–4.5）

n = 12 峰值电路 + 三种噪声通道，四个子实验：

| 子实验 | 对照 | 结果 |
|---|---|---|
| A · Hamming-ball aggregation（算法 2） | 命题 4.6 | 实测与命题界一致（均匀背景下达到上界） |
| B · 逐比特多数表决 | 命题 4.11 | N90 对 (1−2r) 的 log-log 斜率 = **−2.028**（理论 −2） |
| C · Hamming-中心解码（算法 3） | 命题 4.10 | 斜率 = **+0.819**（理论 +1，量级一致） |
| D · 去极化通道 + 去偏估计 | §4.5 | 估计器 SD / 理论 SE ∈ [0.88, 1.09] |

产出：`autotest_results/e1_noise/`（4 组 CSV + 2×2 面板图）。

![e1 panels](autotest_results/e1_noise/e1_noise_panels.png)

### E2 · HS 重叠统计（§4.2 / Obs. 2.5 / Fig. 2 右）

设置与论文 §4.4 相同：τr = τp = n，n = 8/10/12，**每档 100 个实例**（与论文一致）。Tr(C†C′) 由双张量网络**精确收缩**；每个实例内嵌自检（300/300 通过，逐位误差 0）。

| n | 平均 δ | E\|Tr(C†C′)\|²（训练对） | 对理论 2 之比 | E\|Tr\|²（独立对照） | 对理论 1 之比 |
|---|---|---|---|---|---|
| 8 | 0.933 | 2.116 ± 0.214 | 1.058 | 0.998 ± 0.097 | 0.998 |
| 10 | 0.767 | 2.731 ± 0.243 | 1.365 | 1.100 ± 0.111 | 1.100 |
| 12 | 0.576 | 1.839 ± 0.199 | 0.920 | 1.040 ± 0.123 | 1.040 |

**联合统计**：训练对归一化统计量 E\|Tr\|²/2 = **1.114 ± 0.064**（理论 1.0）；独立对照 = **1.046 ± 0.064**（理论 1.0）。训练对约为独立对照的 2 倍，跨 n 无系统性尺寸依赖，与块分解模型 P = C′†C ≅ diag(1, V)（V ~ Haar(d−1)，E\|1+Tr V\|² = 2）相符。

![hs overlap](autotest_results/e2_hs_overlap/hs_overlap_plot.png)

**归一化口径澄清**（数值发现）：论文 Obs. 2.5 正式定义 ⟨C,C′⟩_HS = Tr(C†C′)/d，故 E|⟨·⟩_HS|² = 2/d²；而 §1 与 §4.4 的文字及 Fig. 2 右图纵轴的实际读数为 **2/d = 2^(1−n)**（Fig. 2 中 n=8 数据点 ≈ 8×10⁻³ = 2/2⁸；若为 2/d² 应低约 2.4 个数量级、不在图内）——二者相差一个 1/d 因子，系论文记号层面的不一致。本复现直接测不变量 E|Tr|²（训练对 ≈2、独立对 ≈1）绕开该问题，两种口径均可由它推出。

### 结论汇总（对照论文第二篇）

| 论文主张 | 本仓库验证 |
|---|---|
| postselection 生成 RPC 的接受率 = (1−δ)^{d−1}（指数稀有） | ✅ E3：d = 4–256 全部一致 |
| 训练生成的电路对满足 E\|⟨C,C′⟩_HS\|² ≈ 2/d（Fig. 2） | ✅ E2：训练对 ≈ 2× 独立对照、跨 n 无尺寸依赖 |
| 优化电路在非峰维度保持设计随机性（Thm. 4.4 的数值证据） | ✅ E2：联合归一化统计量 1.11 ± 0.06 |
| 噪声下可恢复：多数表决 ∝ (1−2r)⁻²、Hamming-中心 ∝ \|B_t\|、去偏估计样本量 | ✅ E1：斜率 −2.03 / +0.82，SD 比值 0.88–1.09 |

## 本仓库新增内容（相对原仓库）

| 文件 | 说明 |
|---|---|
| `smoke_test.py` | 环境自检：构建小电路 / 峰值计算 / 微型训练 / 绘图（四项全过即环境就绪） |
| `autotest_peaked.py` | 自动化实验：多实例训练统计 + τp 扫描 + 随机基线，结果自动落盘、种子可复现 |
| `autotest_results/` | 实验产出（逐次原始数据 CSV / 汇总 / 基线 / 曲线图 / 摘要 md） |
| `exp_acceptance.py` / `exp_noise_robustness.py` / `exp_hs_overlap.py` | 第二篇复现三实验（E3 接受率 / E1 噪声鲁棒解码 / E2 HS 重叠统计）：断点续跑 + 多进程分片并行 |
| `peaked_circuits .ipynb` | 已按原参数运行并保存训练输出 |

## 环境与运行

```bash
conda create -n peaked python=3.12 -y
conda activate peaked
pip install numpy==1.26.4 scipy==1.11.2 quimb==1.7.3 matplotlib==3.8.3 autograd networkx ipykernel jupyterlab
```

复现踩坑记录（很重要）：
- **numpy 必须为 1.x**（本代码基于 quimb 1.7.3 的依赖体系，勿升级到 numpy 2.x）；
- notebook 第 5 个代码单元需将 `autodiff_backend='tensorflow'` 改为 `'autograd'`（原依赖未包含 TensorFlow）；
- 电路绘图需要额外安装 `networkx`（原 requirements.txt 未列出）。

运行：

```bash
python smoke_test.py                 # 环境自检
python autotest_peaked.py --quick    # 冒烟模式（约 1–2 分钟，小参数全流程）
python autotest_peaked.py            # 完整扫描（CPU 约 20–40 分钟）
```

## 后续计划

- [x] **τr = 50 忠实区间**（2026-10-14 完成，见「深入复现」D1）；
- [x] **标度律复现**（完成，见 D2：拟合 1.172⁻ⁿ vs 论文 1.189⁻ⁿ）；
- [x] **多起点优化**（完成，见 D3：平均增益 ≈ +4.7%）；
- [x] **第二篇数值复现**（2026-10-15 完成，见「第二篇复现」节）：E3 接受率 / E1 噪声鲁棒解码 / E2 HS 重叠统计（每档 100 实例，与论文一致）；
- [ ] （可选延伸）n=14 加密标度律数据点；D2 每档实例数扩展到 ~100；其他架构（k-正则图等）对照；第二篇 E2 补齐 n=9/11 两档。

## 上游简介（原文摘要）

> Over a decade after its proposal, the idea of using quantum computers to sample hard distributions has remained a key path to demonstrating quantum advantage. Yet a severe drawback remains: verification seems to require exponential classical computation. As an attempt to overcome this difficulty, we propose a new candidate for quantum advantage experiments with otherwise-random ''peaked circuits'', i.e., quantum circuits whose outputs have high concentrations on a computational basis state. Naturally, the heavy output string can be used for classical verification.
>
> In this work, we analytically and numerically study an explicit construction of peaked circuits by attaching random quantum circuits (RQCs) with parametrized quantum circuits (PQCs). If the trend in our result persists, at system size n = 50 and circuit depth n, we would be able to generate an average peak weight of 0.05% by attaching a highly scrambled circuit, whose gate count is half of the RQC, to the RQC layers. This means the peaked circuits we found could serve as a potential candidate for future verifiable quantum advantage experiments.
>
> Publication: arXiv:2404.14493

---

*复现工作：[ChengFoon0602](https://github.com/ChengFoon0602)。原始代码与论文归原作者所有。*
