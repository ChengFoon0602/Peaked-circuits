# -*- coding: utf-8 -*-
"""烟雾测试：验证 Peaked-circuits 仓库的核心代码在本环境可运行。

用法（先装好 peaked 环境，见《复现指引_环境搭建与操作步骤.md》）：
    1. 把本文件复制到 clone 下来的仓库根目录（与 circuits.py 同级）
    2. conda activate peaked
    3. python smoke_test.py

全部打印 [1]-[4] 通过即环境就绪；[3] 做一次微型训练（约 10 秒）。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
print('numpy   :', np.__version__)

import quimb
print('quimb   :', quimb.__version__)

import scipy
print('scipy   :', scipy.__version__)

try:
    import numba
    print('numba   :', numba.__version__)
except Exception:
    print('numba   : 不可用（本仓库核心代码不依赖它，可忽略）')

t0 = time.time()
from circuits import *   # noqa: E402,F403

# ---- 测试 1：构建随机电路并测量单次尖峰权重 ----
psi = qmps_f(6, in_depth=3, n_Qbit=5)
amp = (psi ^ all).data.reshape(2 ** 6)
peak = float(np.max(np.abs(amp) ** 2))
print(f'[1] 6-qubit/3 层随机电路构建成功，单次尖峰权重 = {peak:.5f}')

# ---- 测试 2：平均尖峰权重（小规模基准） ----
m, se, mx = average_peak_weight(6, depth=3, shots=3)
print(f'[2] average_peak_weight(6, depth=3, shots=3) = {m:.5f} ± {se:.5f}（max {mx:.5f}）')

# ---- 测试 3：TNOptimizer 微型训练（autograd 后端）----
rqc = qmps_f(6, in_depth=6, n_Qbit=5)
pqc = qmps_f(6, in_depth=3, n_Qbit=5)
from quimb.tensor.optimize import TNOptimizer  # noqa: E402

optmzr = TNOptimizer(
    pqc,
    loss_fn=negative_overlap,
    norm_fn=norm_f,
    loss_target=-1 + 1e-2,
    constant_tags=['MPS'],
    loss_constants={'target': rqc},
    autodiff_backend='autograd',   # 原 notebook 写的是 'tensorflow'，需改成 'autograd'
    optimizer='L-BFGS-B',
)
print('[3] TNOptimizer 构建成功（autograd 后端）')
optmzr.optimize(5)
print(f'[3] 5 步优化完成，loss = {optmzr.loss:.4f}（-1 为完美重叠，越小越好）')

# ---- 测试 4：电路绘图（notebook 第 3 个单元用到；需要 networkx） ----
try:
    import matplotlib
    matplotlib.use('Agg')
    psi.draw()
    print('[4] 电路绘图 OK')
except Exception as e:
    print(f'[4] 绘图失败: {type(e).__name__} —— 若提示缺 networkx 请运行: pip install networkx')

print(f'SMOKE TEST PASSED（总耗时 {time.time()-t0:.1f}s）')
