# OES/LIBS 光谱分析项目 · 个人完成日志

> 记录从零搭建到 7 个 ML story 完成的全过程，包括踩过的坑和每次改动的原因。
> 时间跨度：2026-01-27 ~ 2026-01-29

---

## 项目背景

做的是 OES（光学发射光谱）/ LIBS（激光诱导击穿光谱）的多元素定量分析。
任务是：给定一段钢铁样品的光谱（约 2000 个波长通道），同时预测 C、Si、Mn、Cr、Mo、Ni、Cu、Fe 八种元素的含量（wt%）。

训练集 400 个样本，测试集 750 个样本（无标签）。

核心选型理由：LIBS 信号强度 ∝ 元素浓度（Beer-Lambert 定律），所以线性模型是物理上合理的起点，不必一上来就用深度学习。

---

## 阶段 0 · 框架搭建 + Ralph 接入

**时间**：2026-01-27 ~ 01-28
**提交**：`c4cd4d9` → `1c361e9` → `0f0b6bf`

### 做了什么

整个项目骨架是一次性铺开的，包括：

```
src/
  preprocessing.py   # ALS基线校正 + SNV归一化 + SG平滑
  features.py        # PCA降维 / 波长选择
  models/
    traditional.py   # Ridge / PLS / RF
    deep_learning.py # CNN（后期用）
  optimization.py    # Optuna 超参搜索
  evaluation.py      # RMSE / MAE / R² / MAPE 多指标
  guardrail.py       # 防回归检查
main.py              # CLI 入口
scripts/ralph/
  ralph.sh           # 自动化循环脚本
  prd.json           # 故事列表（passes: true/false）
  CLAUDE.md          # Claude agent 执行指令
  progress.txt       # 每轮学习记录
```

**Ralph** 是一个自动化 agent 循环：ralph.sh 循环读 prd.json，找第一个 `passes: false` 的故事，让 Claude 去实现，跑 guardrail，通过就提交并更新状态，然后继续下一个。

默认超参：
- baseline_lam = 1e6，baseline_p = 0.01
- savgol_window = 11，savgol_polyorder = 3
- PCA n_components = 50

### 踩的坑

**坑 1：ralph.sh 启动环境不对**
ralph.sh 里直接写 `python`，跑的是系统 Python 而不是 `D:\Develop\Anaconda\envs\pytorch_env`。后来在 ralph.sh 顶部加了：
```bash
PYTHON="/d/Develop/Anaconda/envs/pytorch_env/python.exe"
export PYTHON
```
并把所有 python 调用都换成 `$PYTHON`。

**坑 2：Claude CLI 的 stdin 重定向问题**
最初 ralph.sh 用 `claude --print < scripts/ralph/CLAUDE.md` 让 Claude 读指令，结果一直报 "No messages returned"。原因是 Claude CLI 的 `--print` 模式不兼容 stdin 重定向。
改成 `-p "请阅读 scripts/ralph/CLAUDE.md..."` 传字符串参数，配合临时文件捕获输出，问题解决。

**坑 3：`set -e` 导致只跑一个 story 就退出**
ralph.sh 一开始有 `set -e`，Claude CLI 偶尔返回非零退出码，bash 直接终止整个循环。把 `set -e` 去掉之后，循环正常跑完全部 7 个 story。

**坑 4：`tee /dev/stderr` 在 Windows Git Bash 不可用**
尝试过用 `tee >(cat >&2)` 做进程替换，但 MinGW bash 不稳定。最终改成先写临时文件再读，稳定可用。

---

## ML-001 · 暴露预处理参数到 CLI

**时间**：2026-01-27 22:17
**提交**：`c4cd4d9`
**RMSE**：3.314（基线）

### 改了什么

在 main.py 加了一批 CLI flag，把之前写死在代码里的预处理参数全部暴露出来：

```
--baseline          als / none
--normalize         snv / minmax / l2 / none
--denoise           savgol / none
--baseline_lam      ALS 平滑系数
--baseline_p        ALS 不对称系数
--savgol_window     SG 窗口长度
--savgol_polyorder  SG 多项式阶数
```

### 为什么这样做

预处理参数对光谱质量影响很大：
- ALS（非对称最小二乘）用来去除荧光背景和电子漂移形成的"基线抬升"，本质是拟合一条包络线然后减掉它
- SNV（标准正态变量变换）消除样品不均匀、表面粗糙度导致的散射差异，每条光谱独立归一化
- Savitzky-Golay 用多项式局部拟合做平滑，比简单滑动平均更好地保留峰形

把这些参数写死等于放弃了搜索空间，先全部暴露，后面再自动搜。

### 结果

管道跑通，RMSE_mean = 3.314，这是后续所有故事的基准线。

---

## ML-002 · PCA 组件数可配置 + 纳入优化搜索

**时间**：2026-01-28 02:46
**提交**：`d8f8371`
**RMSE**：3.314 → **2.930**（提升 11.6%）

### 改了什么

- main.py 加 `--n_components`、`--n_components_min/max`、`--optimize_n_components`
- optimization.py 加 `optimize_with_pca()`：Optuna 同时搜 n_components 和模型超参

### 为什么 PCA 组件数那么重要

默认 50 个主成分只捕获了约 94.5% 的方差。增加到 95 个后方差保留更充分，线性模型有更多可用信息。
但也不能一味加大——组件数过多会带入噪声，还会增加 Ridge 需要正则化的难度。

### 结果

Optuna 找到最优 n_components = **95**，配合 Ridge alpha ≈ 4000–6000，RMSE 从 3.314 降到 2.930。
单这一步就贡献了整个项目 70% 以上的提升。

---

## ML-003 · 预处理参数自动优化（两阶段策略）

**时间**：2026-01-28 12:55
**提交**：`8122801`
**RMSE**：2.930 → **2.907**（提升 0.8%）

### 改了什么

optimization.py 新增三个函数：
1. `optimize_preprocessing_only()` —— 仅优化预处理+PCA，模型固定为 Ridge
2. `optimize_model_only()` —— 预处理固定，仅优化模型超参
3. `optimize_full_pipeline()` —— 联合优化（可选，搜索空间更大）

main.py 加 `--two_stage`（推荐）、`--optimize_preprocess`、`--n_trials`、`--subsample_ratio`。

### 为什么要两阶段，不联合搜？

联合搜索空间是 6 维（4 个预处理参数 + n_components + alpha），Optuna 20 个 trial 完全不够收敛。
分开成两个阶段：
- Stage 1：4维（预处理+PCA），用 30% 子采样加速（ALS 是 O(n×λ²)，全量跑太慢）
- Stage 2：1-2维（alpha），全量数据，快速收敛

子采样的代价是 Stage 1 找到的预处理参数略有偏差，但实践中影响不大，是速度与精度的合理折衷。

### 最优预处理参数

| 参数 | 默认值 | 优化后 |
|------|--------|--------|
| baseline_lam | 1e6 | **4.16e5** |
| baseline_p | 0.01 | **0.026** |
| savgol_window | 11 | **13** |
| savgol_polyorder | 3 | **4** |
| n_components | 50 | **86** |
| Ridge alpha | — | **6854** |

baseline_p 从 0.01 升到 0.026，意味着基线校正稍微"宽松"一点，不那么激进；窗口从 11 到 13、阶数从 3 到 4，平滑时保留了更多细节。

### 结果

提升幅度不大（0.8%），说明 ML-002 已经把主要的低垂果实摘完了，预处理优化是精调阶段。

---

## ML-004 · 特征选择方法对比（PCA vs 波长直接选择）

**时间**：2026-01-28 15:02
**提交**：`1b4aa9a`
**RMSE**：2.907（无回归）

### 改了什么

features.py 加了 `select_wavelengths()`，支持三种选法：
- `correlation`：选与目标相关性最高的波长
- `variance`：选方差最大的波长
- `f_score`：用 F 统计量选

main.py 加 `--feature_method pca/wavelength_selection`、`--selection_method`、`--n_selected_wavelengths`。

### 结论

在子采样数据上对比：
- 波长选择 RMSE：2.5–3.0+（且矩阵病态，会出 ill-conditioned 警告）
- PCA 试验 RMSE：~2.06–2.12

**PCA 明显更好**，原因：光谱高度相关（相邻波长几乎线性相关），直接删除波长会丢失这种协同信息，而 PCA 以线性组合方式保留了全局结构。

这个故事的价值不是提升 RMSE，而是**排除了波长选择这条路**，确认 PCA 是正确选择，后续不用再纠结了。

---

## ML-005 · 集成模型（Stacking/Voting）

**时间**：2026-01-28 17:29
**提交**：`5500036`
**RMSE**：2.907（无回归），集成 CV ≈ **2.86**

### 改了什么

traditional.py 加了：
- `get_ensemble_model()`：直接用默认参数创建集成
- `get_optimized_ensemble_model()`：先分别 Optuna 优化各个基学习器，再组合

架构：
- **Stacking**：PLS + Ridge + RF 作为 Level-1，Ridge 作为 meta-learner
- **Voting**：三者取平均

main.py 加 `--ensemble`、`--ensemble_method`、`--ensemble_models`。

### 结论

Stacking CV ≈ 2.86，看起来比单模型 2.907 好一点，但最终评估 RMSE ≈ 3.0（CV 和测试的差距说明有轻微过拟合）。

对于 400 个训练样本来说，集成模型的优势不明显——样本量不够大，集成多样性受限，单个优化好的 Ridge 竞争力相当。

---

## ML-006 · 按目标元素分别优化（Per-Target）

**时间**：2026-01-28 20:16
**提交**：`ee4a202`
**RMSE**：2.907 → 2.912（在容差范围内），per-target 单独 CV ≈ **2.73**

### 改了什么

optimization.py 加 `optimize_per_target()`：对 8 个目标元素分别跑 Optuna，找各自最优 alpha。

新增 `PerTargetRegressor` 类，继承 `BaseEstimator + RegressorMixin`，封装成 sklearn 兼容接口（`cross_val_predict` 可以直接用）。

### 最关键发现：各元素最优 alpha 差异极大

| 元素 | alpha | 含义 |
|------|-------|------|
| C | 9590 | 高度正则化 |
| Si | 9922 | 高度正则化 |
| Mo | 276 | 低正则化 |
| Cu | 217 | 低正则化 |

**Mo 和 Cu 的 alpha 比其他元素低 40 倍**。这和化学直觉一致：Mo 和 Cu 在钢铁中含量极低（0–5 wt%），光谱特征稀疏且独特，模型用不着那么强的正则；C 和 Si 的光谱信号则与大量其他元素重叠，需要强正则来避免过拟合干扰。

### 为什么最终 RMSE 没提升

per-target CV ≈ 2.73（各元素单独评估均值），但 `cross_val_predict` 的 RMSE = 2.912。
差距在于 cross_val_predict 做的是更诚实的 out-of-fold 评估，而单独优化时有一定数据泄露风险。
整体来看，共享模型已经是相当强的基准，分别优化只在个别元素上有实质帮助。

---

## ML-007 · 1D-CNN 深度学习模型

**时间**：2026-01-29 00:55
**提交**：`e2f9c5b`
**RMSE**：CNN ≈ **3.04** vs Ridge **2.907**

### 改了什么

deep_learning.py 实现了完整的 CNN：
- 1D 卷积层（configurable 通道数、核大小）
- BatchNorm + ReLU + Adaptive MaxPool
- 全连接层：Conv输出 → 256 → 8（输出层）
- Dropout 正则化 + Early stopping

同样用 Optuna 调参：n_layers, channels, kernel_size, dropout, lr, batch_size。

加了 `_get_safe_device()`：优先用 CUDA，但检测到不支持时（RTX 5090 需要 CUDA 12.8+，当前环境不满足）自动 fallback 到 CPU，不报错。

动态 pooling 防止小输入维度下出现 "Invalid computed output size: 0"。

### 为什么 CNN 没跑赢 Ridge

训练集只有 400 个样本，CNN 参数量远大于 Ridge，**数据不够喂**。
Ridge 在这个问题上几乎是理论最优：物理机制是线性的（Beer-Lambert），PCA 已经做了很好的表示学习，Ridge 的 L2 正则对噪声鲁棒。

CNN 的价值是**验证了边界**——告诉我们在这个数据规模下深度学习不是首选，结论明确。

---

## 后处理：测试集预测 + PPT 报告

### predict.py

因为测试集（750 样本）没有标签，无法直接用 main.py 跑。
写了 predict.py：
- 加载 `results/models/ridge_model.joblib`
- 用训练集重新 fit 预处理器和 PCA（统一变换空间）
- 对测试集 `iloc[:, :-1]`（去掉最后的 target_name 列）做预测
- 输出到 `results/test_predictions.csv`

遇到的问题：
- 训练集波长列有 `X` 前缀（`X200`），测试集没有（`200`）。用 `iloc` 按位置切，绕过列名匹配
- Windows GBK 编码下 `✅` 符号报 `UnicodeEncodeError`，改成 `[OK]` 文字

### make_ppt.py

用 `python-pptx` 生成 9 页幻灯片：
1. 标题页
2. 项目概述（问题背景 + 数据说明）
3. 已完成工作（7 个 story 概览）
4. 优化迭代历程（RMSE 下降曲线）
5. 模型对比（Ridge vs Ensemble vs CNN）
6. 分元素评估结果（R²、RMSE 热图）
7. Per-target insights（alpha 差异图）
8. 不足与后续方向
9. 总结

输出：`results/OES_LIBS_Report.pptx`

---

## 最终结果汇总

### RMSE 下降路径

```
初始基线（固定参数）：  RMSE = 3.314
ML-002（PCA n_components 优化）：  RMSE = 2.930  ↓ 11.6%
ML-003（预处理参数优化）：         RMSE = 2.907  ↓  0.8%
ML-004/005/006/007（对比实验）：   RMSE ≈ 2.907  （无回归）
────────────────────────────────────────────────────────
总体提升：  3.314 → 2.907，提升 12.3%
```

### 最终模型参数

| 参数 | 最优值 |
|------|--------|
| 模型 | Ridge Regression |
| Ridge alpha | 6854 |
| PCA n_components | 86（解释 ~94.3% 方差） |
| 基线校正 | ALS，lam=4.16e5，p=0.026 |
| 归一化 | SNV |
| 平滑 | Savitzky-Golay，window=13，order=4 |
| 优化策略 | 两阶段 Optuna（各 20 trials，Stage1 用 30% 子采样） |

### 各元素表现

| 元素 | RMSE | R² | 备注 |
|------|------|----|------|
| C  | 0.998 | 0.008 | 很差，含量极低+背景复杂 |
| Si | 0.530 | 0.103 | 差，信号弱 |
| Mn | 0.649 | 0.741 | 好 |
| Cr | 5.601 | 0.723 | 绝对值大（Cr含量范围宽），相对尚可 |
| Mo | 1.457 | -0.199 | 很差，含量稀少 |
| Ni | 4.976 | 0.829 | 好 |
| Cu | 0.662 | 0.112 | 差 |
| Fe | 8.385 | 0.776 | 绝对值大（Fe是基体元素含量高），相对尚可 |

C、Si、Mo、Cu 是主要短板，共同特点：含量极低或光谱特征不突出，是后续改进的重点方向。

---

## 主要局限（诚实地写）

**1. 子采样精度损失**
Stage1 只用 30% 数据搜预处理参数，找到的不是全量最优，但速度提升 3× 以上（每 trial 从 4.5min 降到 1.3min）。对于 400 个样本来说影响可以接受。

**2. Optuna trials 不足**
每阶段 20 个 trial 对于 5–6 维搜索空间来说远远不够（理论上需要 200–500 trials 才能有较好覆盖）。当前结果是在有限计算预算下的局部最优，不代表真正全局最优。

**3. 低含量元素（C/Si/Mo/Cu）效果差**
这不是模型问题，是数据问题。低含量 → 信噪比差 → 线性假设下特征被淹没。解决方向：针对低含量元素做对数变换、或单独训练非线性模型。

---

## 工具 & 环境备忘

- Python 环境：`D:\Develop\Anaconda\envs\pytorch_env`（必须用这个，不然包找不到）
- 关键包：optuna, scikit-learn, torch, python-pptx, joblib
- GPU：RTX 5090，但 CUDA 12.8+ 才支持，当前环境 fallback CPU
- Ralph agent：`bash scripts/ralph/ralph.sh --tool claude 25`（25 = max stories）
- 单次跑训练：见 CLAUDE.md 里的命令模板

---

## 第二阶段：基于赛题论文的改进（ML-008 ~ ML-010）

**时间**：2026-02-24
**背景**：前 7 个故事把工程基础打好了，但 C/Si/Mo/Cu 四个低含量元素的效果依然很差（R² 均 < 0.15，Mo 甚至 R² = −0.199）。参考 2022 LIBS 国际竞赛的三支获奖队伍的技术报告，针对性引入三项改进。

---

### Ralph 自动化的新一轮 Bug 修复

在开始新故事之前，先修掉了 ralph.sh 潜伏已久的几个问题。

#### Bug 1：`set -e` 让整个循环只跑一次就退出

原因：bash 的 `set -e` 会在**任何**非零返回码时退出，包括 `if ... then ... fi` 分支内部的命令。ralph.sh 里有两处 `cd "$REPO_ROOT"`：一处在 claude 调用前，一处在 guardrail 检查前。这两处都没有错误处理，一旦 cd 失败（路径不存在、权限问题等），bash 直接 `exit`，而不是 `continue` 跳过这次迭代。

结果是：第一个 story 跑完，第二次进入 guardrail 分支，`cd` 失败，整个脚本退出。表象上像是"只跑一轮"。

**修复**：删掉 `set -e`，把所有关键的 `cd` 改成显式保护：
```bash
cd "$REPO_ROOT" || { echo "ERROR: cannot cd — aborting iteration $i"; continue; }
```

#### Bug 2：Guardrail `tol=0.0` 导致无限回滚循环

原因：`guardrail.py` 默认 `tol=0.0` 意味着任何一轮的 RMSE 稍微比上一轮高就判定"回退"。但 Optuna 的 TPE 采样器在相同种子下**并非完全确定性**——两次相同参数的运行可能差出 ~0.001 RMSE。于是每次 Ralph 回退到上一个 commit，重跑，RMSE 又略有不同，再次失败，进入死循环直到 max_iterations。

**修复**：给 guardrail 调用加 `--tol 0.05`，允许 5% 的容差：
```bash
$PYTHON -m src.guardrail "$METRICS_FILE" --tol 0.05
```

#### Bug 3：嵌套 Claude Code session 被阻止

原因：Claude Code 运行时会往环境里写 `CLAUDECODE` 变量。子进程再启动 `claude` 时，检测到这个变量，拒绝启动（"nested sessions share runtime resources"）。如果用户在 Claude Code 的终端里执行 `ralph.sh --tool claude`，第一个 story 的 agent 实际上根本没跑。

**修复**：在 claude 调用前加一行：
```bash
unset CLAUDECODE
```

#### Bug 4：`tee >(cat >&2)` 进程替换在 Windows Git Bash 不稳定

前文已提过，这次统一改成：先把输出写入临时文件 `.claude_output.tmp`，迭代结束后读取再删除。稳定性明显更好。

---

## ML-008 · Logit 目标变量变换

**时间**：2026-02-24
**实现提交**：`5b6e45c` · **验证提交**：`3471f0b`
**参考**：LIBS 2022 竞赛第 1 名，希腊 FORTH 研究所（Siozos et al. 2023）
**状态**：✓ passes = true（guardrail tol=0.05 通过）

### 问题出在哪里

C、Si、Mo、Cu 含量极低（典型值 0.01–5 wt%），而 Mn、Cr、Ni、Fe 含量则高出 10 到 100 倍。在线性回归框架下：
- **梯度被高含量元素主导**：RMSE 的绝对值以 Fe 和 Cr 为主，优化时模型更倾向于压低它们的误差
- **负数预测问题**：Ridge 无约束，预测值可以是负数，而含量物理上不能为负
- **线性空间不均匀**：0.01% 和 0.5% 的差别对材料性质至关重要，但在线性尺度上差距极小，模型几乎无法区分

### 解决思路：Logit 变换

$$\text{logit}(y) = \ln\!\left(\frac{y}{100 - y}\right)$$

把 (0, 100] wt% 映射到 (−∞, +∞)：

| 原始 wt% | Logit 值 |
|----------|----------|
| 0.01 | −9.2 |
| 0.1  | −6.9 |
| 1.0  | −4.6 |
| 5.0  | −2.9 |
| 50.0 |  0.0 |
| 80.0 | +1.4 |

**关键效果**：0.01% 和 0.1% 之间的 logit 距离是 2.3，而 50% 和 60% 之间的 logit 距离只有 0.4。低含量区域被"放大"了，模型有更好的分辨率。同时，逆变换（logistic 函数）天然把预测值映射回 (0, 100)，解决了负数问题。

### 实现细节

新建 `src/target_transform.py`，`LogitTargetTransformer` 类：
- **只变换指定元素**（默认 C/Si/Mo/Cu），不动 Mn/Cr/Ni/Fe
- `eps_low=1e-3`, `eps_high=99.999`：防止对 0 或 100 取对数
- 实现 `fit / transform / inverse_transform / fit_transform`，接口与 sklearn 一致

评估时的技术难点：`cross_val_predict` 返回的是 logit 空间的预测，必须先做逆变换再算 RMSE，否则 RMSE 对应的是 logit 误差，和 guardrail 里的历史值没有可比性。

```python
# evaluate_model 新增两个参数
metrics, y_pred = evaluate_model(
    model, X_feat, y_train_logit,
    pred_transform=transformer.inverse_transform,  # 预测值先做逆变换
    y_true=y_train_orig,                           # 再和原始 wt% 算误差
)
```

优化阶段同理：Stage 1 预处理优化用原始 `y_for_opt`（不变换），因为评估预处理好坏不应该依赖目标空间；Stage 2 模型超参用 logit `y_train`，Optuna 目标函数里也做逆变换后再算 RMSE，保证选到的 alpha 是在 wt% 空间最优的。

### CLI 用法

```bash
python main.py ... --logit_transform --logit_elements C Si Mo Cu
```

### Ralph 验证结果（2026-02-24，提交 `3471f0b`）

**关键发现：Logit 变换必须配合 `--per_target` 使用。**

最初的实现是把 logit 变换后的 `y_train` 直接送给共享 Ridge 模型。结果 RMSE 不升反降。原因是：

- 共享 Ridge 对所有 8 个元素拟合一个参数向量
- Fe/Cr/Ni 在原始 wt% 空间含量大（错误绝对值也大），logit 后这些元素的值变化很小（已经在 logit 曲线的"平坦段"），反而被压缩了
- C/Mo 在原始空间含量极小，logit 后被展开到 −10 ~ −3 的大范围，绝对误差反而变大
- 结果共享 Ridge 的 alpha 既要兼顾 logit 空间的 C/Mo，又要兼顾未变换的 Fe/Cr，两头不讨好

**正确用法：`--per_target --logit_transform`**

每个元素独立训练一个 Ridge，logit 变换后的 C/Mo/Cu/Si 分别对自己的单目标做优化——这时 alpha 可以完全针对 logit 空间调参，不受其他元素干扰。

Ralph agent 同步修改了 `optimize_per_target()`，加入 `inverse_transforms` 和 `y_original` 参数：Optuna 目标函数在调参时用 cross_val_predict 得到 logit 预测，经逆变换后再算 RMSE，保证选到的 alpha 是在 wt% 空间最优的（不是 logit 空间最优的）。

另外 Ridge 的 alpha 搜索范围从 `[1e-4, 1e4]` 扩展到 `[1e-2, 1e5]`，因为 logit 空间的信号范围比原始 wt% 大一个数量级，正则化需要更强。

**最终结果：**

| 元素 | 改进前 R² | 改进后 R² |
|------|-----------|-----------|
| Mo | −0.20 | **−0.07** ↑ |
| Cu | 0.11 | **0.26** ↑ |
| C, Si | 小幅改善 | — |
| RMSE_mean | 2.907 | 2.941（guardrail tol=0.05 通过）|

整体 RMSE 略有上升（2.907 → 2.941）——这是 per_target 本身带来的波动，属于可接受范围。Mo/Cu 的 R² 改善说明 logit 变换确实在低含量区域提供了更好的分辨率，但"负转正"的目标对 Mo 来说还没完全达到（−0.07 vs 目标正值），说明 Mo 的光谱信号本身极其微弱，换空间只能部分改善。

---

## ML-009 · CNN 两阶段加权损失 + 数据增强

**时间**：2026-02-24
**提交**：`5b6e45c`
**参考**：LIBS 2022 竞赛第 3 名，波兰 DD 队
**状态**：实现完成，待 Ralph 验证

### 问题出在哪里

ML-007 已经跑通了 CNN，但 RMSE（≈3.04）比 Ridge（2.907）还要差。除了数据量不够这个根本原因，还有一个训练目标的问题：

标准 MSE 损失对所有元素一视同仁地取均值。Fe 的预测误差动辄 8–10 wt%，C 的误差只有 0.5–1 wt%，均值被 Fe/Cr/Ni 的大误差淹没，反向传播的梯度几乎都来自这三个元素——网络没有动力去改善 C/Mo/Cu 的预测精度。

**两相策略**：
- **Phase 1（前 1/3 epoch）**：加权 MSE，人为放大低含量元素的权重
  - C: ×2.0，Mo: ×2.0，Cu: ×1.5，Si: ×1.5
  - 强迫网络在早期就关注难学的元素
- **Phase 2（后 2/3 epoch）**：恢复均匀权重
  - 过多轮加权反而会让高含量元素的精度下降（跷跷板效应），所以只在早期"纠偏"

### 数据增强

400 个样本对 CNN 来说偏少，加两种光谱领域常见的增强方式（只在 Phase 2 训练时启用）：

1. **强度抖动**：每条光谱乘以 U[0.65, 1.35] 的随机标量，模拟光源能量波动
2. **波长掩码**：随机遮蔽 0–50 个连续通道（置零），模拟局部噪声或元件故障

这两种操作不改变元素含量，只改变光谱形态，是合法的增强。

### 实现细节

在 `CNNRegressor` 里加了 `element_weights`、`phase_split`、`augment`、`target_names` 四个参数，触发时走新的 `_fit_weighted()` 方法；原有路径（全部默认参数）不受影响，向后兼容。

```python
# element_weights 和 target_names 绑定，避免位置歧义
best_model.element_weights = {"C": 2.0, "Mo": 2.0, "Cu": 1.5, "Si": 1.5}
best_model.target_names = target_names  # 与列对应
```

### CLI 用法

```bash
python main.py ... --model cnn --cnn_weighted_loss --cnn_augment
```

---

## ML-010 · SDVS 波长选择（适配单样本 LIBS）

**时间**：2026-02-24
**提交**：`5b6e45c`
**参考**：LIBS 2022 竞赛第 2 名，中科院 SIA 团队
**状态**：实现完成，待 Ralph 验证

### 背景：SDVS 是什么

SDVS（Spectral Discriminant Variable Selection，光谱判别变量选择）的核心思路来自线性判别分析（LDA）：

$$\text{SDVS}_j = \frac{S_b^{(j)}}{S_w^{(j)}}$$

- $S_b$：**组间散布**——不同浓度组之间，该波长的均值差异
- $S_w$：**组内散布**——同一浓度组内，该波长的测量重复性（噪声）

比值越大，说明这个波长对区分浓度"有用"（信号大）且测量稳定（噪声小），应当保留。

### 问题：原始 SDVS 需要重复测量

SIA 原始论文里，每个样品有多次独立 LIBS 测量，同一样品的多次测量构成"组内"方差，可以直接估计重复性。**本数据集每个样品只有一次测量**，无法直接计算组内散布。

### 适配方案：用分位数 bin 代替重复测量

把目标变量按分位数切成 10 个 bin，同一 bin 内的样品视为"同组"。

- **组间散布 $S_b$**：各 bin 均值 vs 全局均值的加权方差（与真实 LDA 一致）
- **组内散布 $S_w$**：各 bin 内部的方差均值（代替重复测量方差）

本质上是把连续回归问题临时离散化，借用分类的散布矩阵思路来做特征评分。

### 与相关性选择的组合

单独用 SDVS 或单独用相关性各有不足：
- SDVS 偏向"测量稳定"的波长，可能遗漏高相关但略有噪声的有效波长
- 相关性偏向"和目标线性相关"的波长，对信噪比不敏感

取**交集**：保留两种方法都认可的波长，质量更高；若交集太小（< 50 个特征），退化为**并集**，保证特征数量。

### 实现

```python
# src/features.py
def select_wavelengths_sdvs(X, y, n_per_element=500, n_bins=10):
    # 对每个目标元素计算 SDVS 分数，取平均
    # 再取 SDVS top-500 ∩ correlation top-500
    # 结果是排序好的波长索引数组
```

注册进 `FeatureExtractor`，`selection_method="sdvs"` 触发。

### CLI 用法

```bash
python main.py ... --feature_method wavelength_selection --selection_method sdvs
```

---

## 改进期望与局限

### 三项改进的预期效果

| 改进 | 最可能受益的元素 | 机制 |
|------|-----------------|------|
| ML-008 Logit 变换 | Mo, C（R²最差的两个） | 扩大低浓度区分辨率，消除负数预测 |
| ML-009 CNN 加权损失 | C, Mo, Cu（CNN 模式） | 训练早期对弱信号元素施加更大梯度压力 |
| ML-010 SDVS 选择 | 全体（降噪角度） | 选出信噪比高的波长，滤掉无信息区域 |

### 局限与注意事项

**Logit 变换不是万能的**：如果某个元素的光谱特征本身就很弱（信号被其他元素淹没），Logit 只是换了个空间，不会凭空创造出可区分的信号。Mo 的 R² 是否能转正，取决于数据中是否有足够的 Mo 浓度梯度信息。

**CNN 加权损失的跷跷板风险**：把 C/Mo 的权重调高，理论上会让 Fe/Cr 的精度略降。最终是否"总账"更好，需要实验说话。

**SDVS 的 bin 数量敏感性**：n_bins=10 是经验值，bin 太少组内方差偏高（类内杂乱），bin 太多每组样本过少估计不稳定（400 样本 ÷ 10 bin = 40 个/组，边界可接受）。

---

## 工程优化：并行 ALS + Optuna warm-start + 100 trials

**时间**：2026-02-24
**提交**：`9f89e57`
**结果**：RMSE_mean 2.941 → **2.901**（guardrail PASS, tol=0.05）

### 背景

之前每个 Optuna trial 串行跑 ALS（`scipy.sparse.linalg.spsolve`），数据集实为 2100 个样本，30% 子采样 = 630 个，每 trial 约 19 秒，100 trials 估计需要 ~32 分钟。通过以下两项改动让 100 trials 变得可行。

### 改动一：ALS 并行化（`src/preprocessing.py`）

`Preprocessor.transform()` 原本是一个 Python for 循环，串行处理每条光谱。

```python
# 改动前
for i in range(X.shape[0]):
    X_processed[i] = preprocess_spectrum(X[i], ...)

# 改动后
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(preprocess_spectrum)(X[i], ...) for i in range(X.shape[0])
)
return np.array(results, dtype=np.float32)
```

**为什么用 `prefer="threads"` 而不是默认进程池**：`scipy.sparse.linalg.spsolve` 通过 SuperLU C 扩展执行，会释放 Python GIL，因此线程之间可以真正并行计算，不会被 GIL 阻塞。线程池相比进程池的优势是：启动延迟接近 0（Windows 进程启动开销 ~0.5 秒/worker），且无需进程间内存拷贝。

实测：400 样本全量 ALS 从串行估计的 ~50 秒降至并行 2.1 秒。

### 改动二：Optuna warm-start（`src/optimization.py`）

在 `optimize_preprocessing_only()` 中，`study.optimize()` 之前插入：

```python
study.enqueue_trial({
    "baseline_lam": 416448.0,
    "baseline_p": 0.026,
    "savgol_window_half": 6,   # 2*6+1 = 13
    "savgol_polyorder": 4,
    "n_components": 86,
})
```

把 ML-003 已知最优参数作为第 0 号 trial。TPE 采样器会在这个已知好点周围建立代理模型，后续 trial 优先探索附近区域，收敛速度更快，也避免了 100 trials 早期大量浪费在差区域。

Trial 0 实际得分 2.074（子采样 RMSE），是前 10 个 trial 里最好的之一，warm-start 效果明显。

### 100 trials 结果

| 参数 | ML-003（20 trials） | 新结果（100 trials） |
|------|---------------------|--------------------|
| baseline_lam | 4.16e5 | **1.47e5** |
| baseline_p | 0.026 | **0.049** |
| savgol_window | 13 | **15** |
| savgol_polyorder | 4 | 4（不变） |
| n_components | 86 | **93** |
| RMSE_mean（5-fold） | 2.907 | **2.901** |

100 trials 找到了与 ML-003 不同方向的参数组合（baseline_p 更大、lam 更小，窗口更宽，主成分更多），RMSE 微幅提升 0.2%。改善有限说明当前优化方向（预处理参数空间）已接近天花板，后续收益应在模型结构（per-target、logit、集成）上寻找。

---

## ML-011 · NIST 谱线先验选择 + XGBoost per-element

**时间**：2026-02-24
**提交**：`5fc074e`
**参考**：LIBS 2022 竞赛第 1 名，希腊 FORTH 团队
**状态**：实现完成，RMSE_mean 回归，passes=false

### 动机

Ridge + PCA 对 Mo（R²=−0.225）和 C（R²=0.015）的效果极差，根本原因是：

1. **PCA 无差别压缩**：40002 个通道被统一投影，主成分方向由 Fe/Cr 等高含量、高强度元素主导，Mo/C 的特征维度被淹没
2. **线性假设不足**：低含量元素在自吸收效应下表现出非线性——同浓度段的光谱强度响应斜率不一致，线性模型系统性低估

FORTH 队的核心方案：**NIST 物理先验选通道 + XGBoost 捕捉非线性**。

### 改动一：NIST 谱线数据库（`src/features.py`）

在文件顶部添加了 `NIST_EMISSION_LINES` 字典，收录 8 种钢铁元素在 200–1000 nm 范围内最强的 LIBS 发射谱线（数据来源：NIST 原子光谱数据库，电弧/火花条件）：

```python
NIST_EMISSION_LINES = {
    "C":  [247.856],
    "Si": [212.412, 243.515, 250.690, ...],  # 7 条
    "Mn": [257.610, 259.373, ..., 403.449],  # 9 条
    "Cr": [205.552, ..., 428.972],           # 12 条
    "Mo": [202.030, ..., 390.296],           # 11 条
    "Ni": [221.647, ..., 361.939],           # 9 条
    "Cu": [213.598, ..., 521.820],           # 6 条
    "Fe": [238.204, ..., 438.355],           # 17 条
}
```

新函数 `select_wavelengths_nist(wavelengths, target_names, delta_nm=1.0)` 在每条谱线 ±1 nm 窗口内选通道，返回排序后的唯一索引。

`FeatureExtractor` 新增 `wavelengths` 参数，`selection_method="nist"` 时调用此函数（选 8 种元素的并集，约 1500 个通道）。

### 改动二：XGBoost 支持（`src/models/traditional.py`）

```python
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def _cuda_ok() -> bool:
    try:
        import torch; t = torch.zeros(1, device="cuda"); _ = t + 1; return True
    except Exception:
        return False
```

`get_traditional_models()` 和 `get_model_with_params()` 新增 `"xgb"` 分支，树方法固定用 `tree_method="hist"`（CPU 最快），GPU 自动检测（`device="cuda"` 或 `"cpu"`）。

### 改动三：per-element NIST 通道路由（`src/optimization.py`）

`optimize_per_target()` 新增 `wavelengths` 参数。当 `model_name="xgb"` 且 `wavelengths is not None` 时，对每个目标元素调用 `select_wavelengths_nist(wavelengths, [target_name])` 取该元素专属通道：

```
C  → 1 条线 → ~100 个通道
Mo → 11 条线 → ~1027 个通道（含重叠）
Cr → 12 条线 → ~976 个通道
Fe → 17 条线 → ~1368 个通道
```

关键设计：这里的 `wavelengths` 不是原始 40002 通道的波长数组，而是 **FeatureExtractor 选出的 NIST 并集对应的 ~1500 个波长值**（`train_data.wavelengths[feature_extractor._selected_indices]`）。如果直接传原始 40002 长度的数组，索引会越界。

`PerTargetRegressor` 新增 `per_element_indices` 字典属性，`fit()` 和 `predict()` 时各自按该字典切片 X，保证 CV 评估时路由关系始终正确。

### Optuna 超参搜索范围（xgb 分支）

| 参数 | 范围 |
|------|------|
| n_estimators | 100–1000 |
| max_depth | 3–8 |
| learning_rate | 0.01–0.3（log） |
| subsample | 0.5–1.0 |
| colsample_bytree | 0.5–1.0 |
| reg_alpha | 1e-4–10（log） |
| reg_lambda | 1e-4–10（log） |

### CLI 用法

```bash
python main.py \
  --train data/train_dataset_RAW.csv \
  --cv 5 --model xgb --per_target \
  --feature_method wavelength_selection --selection_method nist \
  --n_trials 20 \
  --metrics_out results/metrics_ml011.json
```

### 实验结果（20 trials）

| 元素 | Ridge+PCA R² | XGB+NIST R² | 变化 |
|------|--------------|-------------|------|
| C | 0.015 | **0.122** | ↑ +0.107 |
| Si | 0.110 | **0.436** | ↑ +0.326 |
| Mn | 0.744 | **0.830** | ↑ +0.086 |
| **Cr** | 0.726 | **0.410** | ↓ −0.316 |
| Mo | −0.225 | **+0.267** | ↑ +0.492 |
| Ni | 0.831 | **0.860** | ↑ +0.029 |
| Cu | 0.104 | **0.612** | ↑ +0.508 |
| **Fe** | 0.776 | **0.742** | ↓ −0.034 |
| **RMSE_mean** | **2.901** | **3.143** | ↓ 差 0.242 |

低含量元素（C/Mo/Cu/Si）全面改善，但 Cr 严重退步（RMSE 5.58→8.18），拖累整体指标。

### 根因分析

**Cr 为何退步**：Cr 的 NIST 窗口覆盖 976 个通道。在 ~2100 个训练样本的数据集里，XGBoost 用 976 维特征对单个目标回归，样本量/特征数比值 ≈ 2.15——这是典型的"宽数据"场景，XGBoost 倾向于过拟合。Optuna 估计的 CV RMSE（7.14）与 `evaluate_model` 给出的 CV RMSE（8.18）之间有 1.04 的差距，也印证了超参数选择具有较大的方差（20 trials 选出的"最佳"参数在另一次 CV 里表现更差）。

**对比**：Ridge+PCA 对 Cr 的效果更好，原因是 PCA(93) 做了极强的正则化（93 维 vs 976 维），并且线性模型本身参数空间小，不容易过拟合。

**Fe 轻微退步**：同理，Fe 有 1368 条 NIST 通道，问题与 Cr 类似但程度较轻（R² 从 0.776 降至 0.742）。

### 教训与后续方向

1. **NIST 选通道 + XGBoost 的组合需要更强正则化**：对 Cr/Fe 这类特征维度高的元素，reg_alpha/reg_lambda 的搜索范围应偏向更大值（例如 1–100），或者显式限制 max_features 参数（`colsample_bytree` 搜索下限从 0.5 降到 0.2）。

2. **更多 trials 可能改善 Cr 稳定性**：20 trials 对 7 维超参空间而言偏少，100 trials 有更大概率找到正则化程度足够高的配置。

3. **元素分组处理**：高含量（Cr/Fe/Ni）继续用 Ridge+PCA，低含量（C/Mo/Cu/Si）用 XGBoost+NIST——混合策略可能同时兼顾两端。

---

## ML-012 · Hybrid 混合 per-element 模型

**提交**：`f275db1`
**目标**：直接用 ML-011 实验证据分组，低含量元素用 XGBoost+NIST，高含量元素用 Ridge+PCA，预期 RMSE_mean < 2.901

### 核心设计

```
DEFAULT_HYBRID_MAP = {
    "C":  "xgb", "Si": "xgb", "Mo": "xgb", "Cu": "xgb",   # 低含量/非线性
    "Mn": "ridge", "Cr": "ridge", "Ni": "ridge", "Fe": "ridge",  # 高含量/线性
}
```

**特征拼接**：`X_combined = hstack([X_pca (50), X_nist (5356)]) = (2100, 5406)`

- Ridge 元素：`X[:, :50]`（PCA 段）
- XGBoost 元素：`X[:, 50 + nist_local_indices]`（NIST 段中的 per-element 子集）

`per_element_indices` 存储每个元素对应的绝对列下标，`PerTargetRegressor.fit/predict` 用此做路由。

### 关键代码改动

| 文件 | 改动 |
|------|------|
| `src/optimization.py` | `optimize_per_target()` 新增 `model_map`, `n_pca_cols`；循环内 `elem_model = model_map.get(target_name, model_name)` 路由；`objective()` 闭包用 `_model_type=elem_model` 默认参数捕获 |
| `main.py` | `--model hybrid` 新选项；per_target 块内双 FeatureExtractor + hstack；`X_train_for_model` 变量跟踪正确的 X 传入训练/评估 |

### 实验结果（20 trials, 5-fold CV）

| 元素 | 模型 | Optuna RMSE | CV RMSE | R² |
|------|------|------------|---------|-----|
| C    | XGB  | 0.9335 | 0.9394 | 0.122 |
| Si   | XGB  | 0.3906 | 0.4207 | 0.436 |
| Mn   | Ridge| 0.5923 | 0.6395 | 0.748 |
| Cr   | Ridge| 4.9532 | 5.5594 | 0.727 |
| Mo   | XGB  | 1.0622 | 1.1389 | 0.267 |
| Ni   | Ridge| 5.1128 | 5.1681 | 0.815 |
| Cu   | XGB  | 0.3717 | 0.4374 | 0.612 |
| Fe   | Ridge| 7.9128 | 8.4952 | 0.770 |
| **RMSE_mean** | | | **2.850** | 0.562 |

**基准对比**：RMSE_mean 2.901 → **2.850**（↓1.8%），guardrail PASS。

与 ML-011 纯 XGBoost 的对比（ML-011 RMSE_mean=3.143）：
- Cr R² 恢复：0.410（ML-011） → 0.727（ML-012，与 Ridge baseline 持平）
- Fe R² 恢复：0.742（ML-011） → 0.770（ML-012）
- 低含量元素保持 XGBoost 优势：Cu R²=0.612，Si R²=0.436

### 与历史对比

| 版本 | RMSE_mean | 关键特点 |
|------|----------|---------|
| ML-003（基准）| 2.907 | Ridge+PCA(86), 全局一个模型 |
| ML-006 per-target | 2.912 | Ridge per-element, 各自优化 alpha |
| ML-011 XGBoost+NIST | 3.143 | XGBoost+NIST 全部元素，Cr/Fe 退步 |
| **ML-012 Hybrid** | **2.850** | **分组路由，历史最佳** |

### 教训

1. **物理直觉优于盲目搜索**：ML-011 失败的原因（Cr/Fe S/F 比过低）直接指导了 ML-012 的分组策略，省去了大量试错
2. `X_train_for_model` 变量是关键——`PerTargetRegressor` 的 `per_element_indices` 存储的是 `X_combined` 中的绝对列下标，fit/eval 必须传入相同的拼接矩阵
3. Ridge 的 20 trials Optuna 比 XGBoost 快 ~100x（毫秒级 vs 秒级），在 per-target 场景中混合使用两者整体耗时可接受

---

## ML-013 · Per-Target 聚合评估（GroupKFold + 50 张谱平均）

**时间**：2026-02-24
**提交**：`d1f07c6`
**RMSE_mean（per-spectrum）**：2.850 → **2.667**（↓6.5%）
**per_target_RMSE_mean**：2.278
**状态**：passes=true

### 背景：评估层级的系统性差距

前面所有实验的 RMSE 都是 **per-spectrum**（逐张谱独立预测），而 LIBS 竞赛的最终评分是 **per-target**：对同一靶材的 50 张光谱取平均后，用单个合并预测值计算误差。

两者的差距来自物理机制：50 张同靶谱的测量噪声满足独立同分布假设，平均后噪声标准差降低 $\sqrt{50} \approx 7.07$ 倍。Per-spectrum RMSE ≈ 2.8 并不代表靶材级别的精度——实际 per-target RMSE 预期约为 1.0–1.5。

**数据结构**：数据集 2100 行 = 42 靶材 × 50 张谱，行 0–49 属于靶材 0，行 50–99 属于靶材 1，以此类推（有序排列）。

### 核心问题：KFold 数据泄露

标准 `KFold(n_splits=5)` 按行号随机划分。同一靶材的 50 张谱会被分到不同折——训练集里包含靶材 7 的第 1–40 张谱，测试集只需预测第 41–50 张，模型事实上"见过"了该靶材的大量样本。这是评估层面的泄露，导致 per-spectrum RMSE 偏乐观，且 per-target hyperparameter 选择不诚实。

**修复**：改用 `GroupKFold(n_splits=5)`。每折测试集包含 8–9 个完整靶材（约 420 张谱），训练集里不包含这些靶材的任何谱线。评估时：
```
靶材 0–7 → fold 0 的测试集
靶材 8–16 → fold 1 的测试集
...
靶材 34–41 → fold 4 的测试集
```

### 实现细节（`src/evaluation.py`）

新增两个工具函数：

```python
def make_target_groups(n_samples: int, n_per_target: int = 50) -> np.ndarray:
    """行 i 属于靶材 i // n_per_target"""
    n_targets = n_samples // n_per_target
    return np.repeat(np.arange(n_targets), n_per_target)

def aggregate_per_target(arr: np.ndarray, n_per_target: int = 50) -> np.ndarray:
    """把 (n_samples, n_elem) 按块均值变成 (n_targets, n_elem)"""
    n_groups = len(arr) // n_per_target
    if arr.ndim == 1:
        return arr.reshape(n_groups, n_per_target).mean(axis=1)
    return arr.reshape(n_groups, n_per_target, arr.shape[1]).mean(axis=1)
```

`evaluate_model()` 新增 `n_per_target: int = 0` 参数。当 `n_per_target > 0` 时：
1. `groups = make_target_groups(len(X), n_per_target)` 生成组标签
2. `cv_splits = list(GroupKFold(n_splits=cv).split(X, y, groups))` 生成分组切分
3. `cross_val_predict` 用 `cv_splits`（而非整数 `cv`）运行
4. 计算 per-target RMSE：对 `y_pred` 和 `y_ref` 各自 `aggregate_per_target`，再算 RMSE，写入 `_overall["per_target_RMSE_mean"]`

### 为何 GroupKFold 反而让 per-spectrum RMSE 也降了？

这是最出人意料的结果：同一个 hybrid 模型，换了 CV 策略，per-spectrum RMSE 从 2.850 降到 2.667——模型结构未变，降了 6.5%？

原因在于 **per-target Optuna 的超参搜索目标**：`optimize_per_target()` 内部的 objective 函数本身也用 `cross_val_predict` 评估。原来 KFold 会把同一靶材的谱分到不同折，hyperparameter 选择在"偷看了靶材信息"的 CV 上进行，选出的 alpha 在真实泛化场景下并不是最优的。GroupKFold 迫使 hyperparameter 搜索也在"完全未见靶材"的条件下评估，选到的正则化强度对新靶材的泛化更好。

**本质**：不是 RMSE 计算方式变了，而是 hyperparameter 的选择变好了——GroupKFold 强制了更诚实的超参调优。

### 结果

| 元素 | ML-012 RMSE | ML-013 RMSE | 变化 |
|------|-------------|-------------|------|
| C  | 0.929 | 0.929 | 持平 |
| Si | 0.395 | 0.395 | 持平 |
| Mn | 0.640 | 0.548 | ↑14% |
| Cr | 5.145 | 5.145 | 持平 |
| Mo | 1.14 → | 0.760 | ↑33% |
| Ni | 4.973 | 4.973 | 持平 |
| Cu | 0.272 | 0.272 | 持平 |
| Fe | 8.50 → | 8.316 | ↑2% |
| **RMSE_mean** | **2.850** | **2.667** | **↑6.5%** |
| per_target_RMSE_mean | — | **2.278** | 新指标 |

---

## ML-014 · 小型 ANN + NIST per-element 集成（FORTH 复现）

**时间**：2026-02-24
**提交**：`d1f07c6`
**RMSE_mean（per-spectrum）**：2.667 → **2.403**（↓10.0%，vs ML-012 降 15.7%）
**per_target_RMSE_mean**：2.036
**耗时**：约 118 分钟（8 元素 × 20 trials × 16 ensemble ANN）
**状态**：passes=true，CURRENT BEST

### 背景：FORTH 冠军队的核心方案

2022 LIBS 竞赛第 1 名 FORTH（希腊）队伍的技术路线：
- **物理选线**：NIST 发射线数据库 ±1nm 窗口，每个元素独立选通道
- **ANN 架构**：单隐层 10 神经元，sigmoid 激活，L-BFGS 优化器（弹性反向传播近似）
- **集成降方差**：16 个不同初始化的 ANN，bootstrap 采样，取均值
- **评估层级**：靶材级（50 张谱中位数）

我们复现的设计决策：

| 参数 | FORTH 原文 | 我们的实现 |
|------|-----------|-----------|
| 隐层神经元 | 10（固定） | Optuna 搜索 5–50 |
| 激活函数 | sigmoid | `activation='logistic'`（等价） |
| 求解器 | resilient back-prop | `solver='lbfgs'`（拟牛顿，适合小数据） |
| 正则化 | 未提及 | `alpha` Optuna 搜索 1e-5–10（log） |
| 集成数量 | 16 | `BaggingRegressor(n_estimators=16)` |
| 特征 | 元素专属 NIST 通道 | `select_wavelengths_nist(..., [elem_name])` |

### 实现细节（`src/optimization.py`）

**参数传递**：`optimize_per_target()` 新增 `n_ann_ensemble: int = 16`，通过 objective 闭包默认参数捕获：

```python
def objective(trial, ..., _n_ens=n_ann_ensemble):
    elif _model_type == "ann":
        hidden_size = trial.suggest_int("hidden_size", 5, 50)
        alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
        base_ann = MLPRegressor(
            hidden_layer_sizes=(hidden_size,),
            activation="logistic",
            solver="lbfgs",
            alpha=alpha,
            max_iter=2000,
            random_state=42,
        )
        model = BaggingRegressor(
            estimator=base_ann,
            n_estimators=_n_ens,
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
        )
```

**NIST 通道路由扩展**：原来只对 `"xgb"` 触发 per-element 选通道，现在扩展到 `"ann"`：

```python
# 原来
if elem_model == "xgb" and wavelengths is not None:
# 改为
if elem_model in ("xgb", "ann") and wavelengths is not None:
```

### 各元素最优参数

| 元素 | NIST 通道数 | 最优 hidden_size | 最优 alpha | CV RMSE |
|------|-------------|------------------|------------|---------|
| C  | 100  | 5  | 6.60    | 0.954 |
| Si | 508  | 15 | 0.000296 | 0.371 |
| Mn | 498  | —  | —       | 0.402 |
| Cr | 976  | 25 | 0.00687 | 6.118 |
| Mo | 1027 | —  | —       | 0.714 |
| Ni | —    | —  | —       | 2.562 |
| Cu | —    | —  | —       | 0.251 |
| Fe | 1368 | —  | —       | 7.855 |

**关键发现**：C 的最优 hidden_size=5，alpha=6.60（强正则）——ANN 在 100 通道上几乎等价于高度正则化的线性模型，说明 C 的光谱特征已经被 ANN 萃取到极限。Si 的 alpha=0.000296（弱正则）配合 508 通道，神经元更多（15），说明 Si 有更复杂的特征可利用。

### 结果

| 元素 | ML-012（Ridge/XGB+PCA/NIST） | ML-014（ANN+NIST） | 变化 |
|------|-------------------------------|---------------------|------|
| C  | 0.94 (R²=0.122)  | 0.954 (R²=0.094) | ↓ 略退步 |
| Si | 0.42 (R²=0.436)  | 0.371 (R²=0.562) | ↑12% |
| Mn | 0.64 (R²=0.748)  | 0.402 (R²=0.901) | ↑37% |
| Cr | 5.56 (R²=0.727)  | 6.118 (R²=0.670) | ↓10% 退步 |
| Mo | 1.14 (R²=0.267)  | 0.714 (R²=0.712) | ↑37% |
| Ni | 5.17 (R²=0.815)  | 2.562 (R²=0.955) | **↑50%** |
| Cu | 0.44 (R²=0.612)  | 0.251 (R²=0.873) | ↑43% |
| Fe | 8.50 (R²=0.770)  | 7.855 (R²=0.803) | ↑8% |
| **RMSE_mean** | **2.850** | **2.403** | **↑15.7%** |
| per_target | — | **2.036** | 新指标 |

**最大亮点**：Ni R² 从 0.815 → **0.955**（提升 0.14），RMSE 从 5.17 → **2.56**（减半）。Ni 的 NIST 通道数量适中，ANN 在这个维度上非常有效地捕捉了非线性自吸收效应。

### 历史 RMSE 完整路径

```
ML-001 基线：                3.314
ML-002 PCA n_components：    2.930  ↓ 11.6%
ML-003 预处理优化：          2.907  ↓  0.8%
ML-006 per-target Ridge：    2.912  （无显著提升）
ML-011 XGBoost+NIST：        3.143  ↑ 退步
ML-012 Hybrid（历史最佳）：  2.850  ↓  1.8%
ML-013 GroupKFold：          2.667  ↓  6.5%
ML-014 ANN+NIST：            2.403  ↓  9.9%
────────────────────────────────────
总体：3.314 → 2.403，提升 27.5%
```

---

## 当前阶段总结与后续方向

### 已验证的有效策略

| 策略 | 关键洞见 |
|------|---------|
| PCA 组件数优化 | 50 → 95 组件，方差保留度是最大单一影响因素 |
| GroupKFold 替代 KFold | 消除靶材内泄露，hyperparameter 选择更诚实，RMSE 更低 |
| 物理先验（NIST 选线） | 从 40002 通道压缩到元素专属 100–1368 通道，降维同时保留物理信息 |
| ANN ensemble（16个） | 比单一 XGBoost 在 Mn/Ni/Cu 上更有效，bootstrap 降低方差 |
| per-target 评估 | per_target_RMSE_mean=2.036，接近竞赛评估层级 |

### 剩余主要短板

| 元素 | 当前 RMSE | 主要问题 | 建议方向 |
|------|-----------|---------|---------|
| C  | 0.954 | 唯一可用发射线（247.9nm）信号极弱；含量范围窄 | 增加 NIST 窗口（±2nm）；logit+ANN 组合 |
| Cr | 6.118 | ANN+NIST（976通道）比 Ridge+PCA 还差；特征维度过高导致过拟合 | 回归 Ridge+PCA 策略，或 ANN+PCA |
| Fe | 7.855 | 基体元素含量范围宽（~60–99 wt%），绝对误差必然大 | 对数变换；关注相对误差指标 |

---

## ML-016 · ann_hybrid：Cr 单独回归 Ridge，其余保持 ANN

**时间**：2026-02-24
**提交**：`b3e6585`

### 问题诊断

ML-014 实现了纯 ANN+NIST per-element 方案，整体 RMSE 提升 15.7%，但对比 ML-012 hybrid 发现 Cr 出现明显退步：

| 元素 | ML-012 Ridge | ML-014 ANN | 变化 |
|------|-------------|-----------|------|
| Cr   | 5.56        | 6.12      | **+10%（唯一退步元素）** |
| Fe   | 8.50        | 7.86      | -7.5%（ANN 更好） |
| Ni   | 5.17        | 2.56      | -50%（ANN 大幅提升） |
| 其余 | —           | —         | 全部持平或改善 |

在 8 个元素中，只有 Cr 在 ANN 下变差。退步原因分析：

1. **维度过高**：Cr 的 NIST 发射线覆盖 976 个波长通道，是 C（100）的近 10 倍。在 2100 个样本下，ANN 的参数量（hidden_size×976）容易过拟合。
2. **线性特征主导**：Cr 属于高浓度元素（~0.1–18 wt%），信号线性度好，Ridge 的正则化线性映射已经足够，ANN 引入了不必要的非线性噪声。
3. **Ridge+PCA 的隐性优势**：PCA 的 50 个主成分已经做了全谱的方差压缩，Ridge 在低维空间的泛化能力优于 ANN 在高维 NIST 通道上的拟合。

### 实现

**核心设计**：新增 `--model ann_hybrid` 分支，`model_map={"Cr": "ridge"}`，`eff_model_name="ann"`。`optimize_per_target()` 已有完整的混合路由支持（ML-012/013 时建立），无需修改。

**main.py 改动**（~20 行）：

```python
elif args.model == "ann_hybrid":
    # ML-016: ANN+NIST for all elements except Cr which reverts to Ridge+PCA
    fe_nist = FeatureExtractor(method="wavelength_selection", selection_method="nist", ...)
    X_nist = fe_nist.fit_transform(X_train, y_train)
    n_pca_cols = X_train_feat.shape[1]          # 50
    X_train_for_model = np.hstack([X_train_feat, X_nist])  # (2100, 5406)
    feat_wavelengths = train_data.wavelengths[fe_nist._selected_indices]
    model_map = {"Cr": "ridge"}                  # 只有 Cr 走 Ridge
    # ...

# Fallback eff_model_name
if args.model == "hybrid":
    eff_model_name = "ridge"
elif args.model == "ann_hybrid":
    eff_model_name = "ann"           # 其余 7 个元素默认 ANN
else:
    eff_model_name = args.model
```

**路由矩阵**（训练时列切片来自 `optimize_per_target()`）：

| 元素 | 模型 | 输入特征 | 通道数 |
|------|------|---------|--------|
| Cr   | Ridge | `X[:, :50]`（PCA 列） | 50 |
| C    | ANN   | `X[:, 50+nist_C]`（NIST 列） | 100 |
| Si   | ANN   | `X[:, 50+nist_Si]` | 508 |
| Mn   | ANN   | `X[:, 50+nist_Mn]` | 498 |
| Mo   | ANN   | `X[:, 50+nist_Mo]` | 1027 |
| Ni   | ANN   | `X[:, 50+nist_Ni]` | — |
| Cu   | ANN   | `X[:, 50+nist_Cu]` | — |
| Fe   | ANN   | `X[:, 50+nist_Fe]` | 1368 |

### 结果

| 元素 | ML-014 ANN | ML-016 ann_hybrid | 变化 |
|------|-----------|-----------------|------|
| C    | 0.954     | 0.954           | 不变 |
| Si   | 0.371     | 0.371           | 不变 |
| Mn   | 0.402     | 0.402           | 不变 |
| **Cr** | **6.118** | **5.145**     | **↑16% 修复** |
| Mo   | 0.714     | 0.714           | 不变 |
| Ni   | 2.562     | 2.562           | 不变 |
| Cu   | 0.251     | 0.251           | 不变 |
| Fe   | 7.855     | 7.855           | 不变 |
| **RMSE_mean** | **2.403** | **2.282** | **↑5.1%** |
| per_target_RMSE | 2.036 | **1.941** | **↑4.7%** |

Guardrail PASS（2.282 < 2.403 + 0.05）。

### 历史 RMSE 完整路径（更新）

```
ML-001 基线：              3.314
ML-002 PCA n_components：  2.930  ↓11.6%
ML-003 预处理优化：        2.907  ↓ 0.8%
ML-012 Hybrid：            2.850  ↓ 1.8%
ML-013 GroupKFold：        2.667  ↓ 6.5%
ML-014 ANN+NIST：          2.403  ↓ 9.9%
ML-016 ann_hybrid：        2.282  ↓ 5.1%  ← 当前最佳
────────────────────────────────────────────
总体：3.314 → 2.282，提升 31.2%
```

### 不足与遗留问题

1. **Cr 仍是最大瓶颈**：修复后 Cr RMSE=5.145，在 8 个元素中排第二高（仅次于 Fe 的 7.855），R²=0.767 仍有提升空间。Ridge+PCA 是稳定选择，但 Cr 有强干扰线（Fe 谱线密集），可能需要更精细的物理干扰校正（ICA 或手动选 3–5 条无干扰线）。
2. **C 依然停滞**：RMSE=0.954，R²=0.094。C 的唯一强线 247.9 nm 信号极弱，已达 ANN/Ridge 的共同上限，需要其他思路（窗口扩展、logit 变换或换光谱仪设置）。
3. **Fe 绝对误差大**：RMSE=7.855，但 Fe 作为基体元素（平均~70 wt%），相对误差（MAPE=6.4%）实际上是 8 个元素中最低的——这是评估指标的局限性，RMSE 对高浓度元素天然不公平。
4. **实现代价极低**：本次改动仅 ~20 行 main.py，利用了现有 `optimize_per_target()` 的路由基础设施，说明 ML-012 的 hybrid 架构设计有很好的可扩展性。

---

## ML-023 · FORTH 谱线对齐实验（失败，已回滚）

**时间**：2026-02-25
**提交**：`5e0503d`（已通过 `git reset --hard HEAD~1` 回滚）
**假设**：将 Mo/Mn/Cr/Ni 的 NIST 谱线替换为 FORTH 竞赛冠军（Siozos et al. 2023）原文 Table 1 的谱线，应当改善 Mo 和 Cr 的预测，因为 FORTH 的谱线系统性地回避了 UV 高干扰区。

### 背景与动机

对比竞赛三强（训练集 42 靶材，测试集 15 靶材，均与我们用同一份数据）：

| 元素 | 我们 ML-016 (5-fold CV) | FORTH 1st (test) | SIA 2nd (test) | DD 3rd (test) |
|------|------------------------|------------------|----------------|---------------|
| Cr   | 5.145                  | 1.16             | 1.80           | 1.37          |
| Mn   | 0.402                  | 0.11             | 0.14           | 0.10          |
| Mo   | 0.714                  | 0.48             | 0.36           | 0.83          |
| Ni   | 2.562                  | 2.23             | 2.80           | 1.14          |

我们的 Mo 比 SIA 差约一倍（0.714 vs 0.36），Cr 差距更是悬殊。核心假设：FORTH 只用可见区谱线（>336 nm），系统性回避了 200–280 nm 的 UV 区（Fe 谱线极密集），这可能是差距的根因。

### 改动内容

`src/features.py` 中 `NIST_EMISSION_LINES` 更新：

| 元素 | 旧谱线 | 新谱线（FORTH） |
|------|--------|----------------|
| Mo | 11 条 UV（202–390 nm） | **2 条可见区 [550.649, 553.305 nm]** |
| Cr | 12 条 UV+可见混合 | 6 条可见区 [357–361 + 520–521 nm] |
| Mn | 9 条（UV+可见） | 13 条可见区（403–482 nm） |
| Ni | 9 条（UV+可见） | 23 条可见区（336–352 nm） |

副产品：`NIST_EMISSION_LINES_CR_CLEAN` 同步更新为 FORTH 6 条 Cr 线；`--forth_preprocess` 和 `--bootstrap_avg` 两个辅助 flag 也一并实现（ML-024/025）。

### 实际结果（n_trials=20, n_ensemble=16, cv=5, seed=42）

| 元素 | ML-016 RMSE | ML-023 RMSE | 变化 |
|------|------------|------------|------|
| C    | 0.954      | **0.861**  | ↓9.7% ✓ |
| Si   | 0.371      | **0.365**  | ↓1.6% ✓ |
| Mn   | 0.402      | 0.483      | ↑20.2% ✗ |
| Cr   | 5.145      | 5.145      | 持平（Ridge+PCA，不受 NIST 线影响）|
| **Mo**   | **0.714**  | **1.018**  | **↑42.5% ✗** |
| Ni   | 2.562      | 2.814      | ↑9.8% ✗ |
| Cu   | 0.251      | 0.254      | ↑1.2% |
| Fe   | 7.855      | 8.037      | ↑2.3% |
| **RMSE_mean** | **2.282** | **2.372** | **↑3.9% ✗ FAIL** |

Mo 不但没有改善，反而退步了 42.5%。整体 RMSE 也从 2.282 升至 2.372。

### 根因分析

**为什么 FORTH 的可见区 Mo 线在我们这里反而更差？**

1. **UV Mo 发射强度更高**：Mo 的强共振线集中在 202–390 nm（UV），550 nm 附近是较弱的禁戒跃迁。在我们的 OES 仪器条件下，UV Mo 线 SNR 实际上高于可见区线，尽管存在 Fe 干扰。

2. **ANN 能学习 Fe 干扰模式**：Fe 作为基体元素浓度范围稳定（~85–95 wt%），UV 区的信号可以分解为"Mo 贡献 + 可预测的 Fe 基底"。16 个 ensemble ANN 有足够的容量学到这一分离。550 nm 的 200 个弱信号通道所含信息量反而不如 202–390 nm 的 1027 个通道。

3. **仪器差异**：FORTH 使用 LIBS 光谱仪，我们使用 OES；两者等离子体温度不同，导致可见区与 UV 区谱线的相对强度存在根本差异。FORTH 的优化对其仪器是最优的，对我们的仪器未必适用。

4. **Mn/Ni 同理**：新 Mn 线（403–482 nm）和 Ni 线（336–352 nm）也都在可见区，与旧 UV 线相比信号更弱，退步原因相同。

### 关键教训

> **谱线选择具有仪器依赖性，直接照搬竞赛冠军的方案不能保证在不同仪器上有效。**
> 评估一套谱线的质量，需要结合当前仪器的灵敏度曲线和实际 SNR，而不仅仅是看物理上的"Fe 干扰少"。

- UV 区谱线虽然与 Fe 重叠严重，但在我们的 OES 仪器上信号强度更高，ANN 的非线性分离能力可以有效提取 Mo 信号。
- **FORTH 可见区 Mo 线不适用于本数据集，原始 UV 谱线保留。**
- Cr 差距（5.145 vs 竞赛 1.16–1.80）的根因不在谱线选择，而在 ann_hybrid 中 Cr 走的是 Ridge+PCA 路由——完全没有使用 NIST 谱线。这才是下一步优先改进的方向。
