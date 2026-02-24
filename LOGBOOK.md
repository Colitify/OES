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
**提交**：`5b6e45c`
**参考**：LIBS 2022 竞赛第 1 名，希腊 FORTH 研究所（Siozos et al. 2023）
**状态**：实现完成，待 Ralph 验证

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
