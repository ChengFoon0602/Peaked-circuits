# Peaked-circuits · 复现学习记录

> **原始仓库**：https://github.com/yuxuanzhang1995/Peaked-circuits  （作者：Yuxuan Zhang，与 Scott Aaronson 合作）
> **论文**：*On verifiable quantum advantage with peaked circuit sampling* — [arXiv:2404.14493](https://arxiv.org/abs/2404.14493)
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

## 本仓库新增内容（相对原仓库）

| 文件 | 说明 |
|---|---|
| `smoke_test.py` | 环境自检：构建小电路 / 峰值计算 / 微型训练 / 绘图（四项全过即环境就绪） |
| `autotest_peaked.py` | 自动化实验：多实例训练统计 + τp 扫描 + 随机基线，结果自动落盘、种子可复现 |
| `autotest_results/` | 实验产出（逐次原始数据 CSV / 汇总 / 基线 / 曲线图 / 摘要 md） |
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
- [ ] （可选延伸）n=14 加密标度律数据点；每档实例数扩展到 ~100 与论文完全对齐；其他架构（k-正则图等）对照。

## 上游简介（原文摘要）

> Over a decade after its proposal, the idea of using quantum computers to sample hard distributions has remained a key path to demonstrating quantum advantage. Yet a severe drawback remains: verification seems to require exponential classical computation. As an attempt to overcome this difficulty, we propose a new candidate for quantum advantage experiments with otherwise-random ''peaked circuits'', i.e., quantum circuits whose outputs have high concentrations on a computational basis state. Naturally, the heavy output string can be used for classical verification.
>
> In this work, we analytically and numerically study an explicit construction of peaked circuits by attaching random quantum circuits (RQCs) with parametrized quantum circuits (PQCs). If the trend in our result persists, at system size n = 50 and circuit depth n, we would be able to generate an average peak weight of 0.05% by attaching a highly scrambled circuit, whose gate count is half of the RQC, to the RQC layers. This means the peaked circuits we found could serve as a potential candidate for future verifiable quantum advantage experiments.
>
> Publication: arXiv:2404.14493

---

*复现工作：[ChengFoon0602](https://github.com/ChengFoon0602)。原始代码与论文归原作者所有。*
