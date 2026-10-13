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
- **结论**：训练尖峰层显著提升输出集中度（约为同深度随机电路基线的 13–151 倍），趋势与论文一致（论文使用更深的 τr = 40–50，参数区不同，不可直接比较数值大小）。

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

## 上游简介（原文摘要）

> Over a decade after its proposal, the idea of using quantum computers to sample hard distributions has remained a key path to demonstrating quantum advantage. Yet a severe drawback remains: verification seems to require exponential classical computation. As an attempt to overcome this difficulty, we propose a new candidate for quantum advantage experiments with otherwise-random ''peaked circuits'', i.e., quantum circuits whose outputs have high concentrations on a computational basis state. Naturally, the heavy output string can be used for classical verification.
>
> In this work, we analytically and numerically study an explicit construction of peaked circuits by attaching random quantum circuits (RQCs) with parametrized quantum circuits (PQCs). If the trend in our result persists, at system size n = 50 and circuit depth n, we would be able to generate an average peak weight of 0.05% by attaching a highly scrambled circuit, whose gate count is half of the RQC, to the RQC layers. This means the peaked circuits we found could serve as a potential candidate for future verifiable quantum advantage experiments.
>
> Publication: arXiv:2404.14493

---

*复现工作：[ChengFoon0602](https://github.com/ChengFoon0602)。原始代码与论文归原作者所有。*
