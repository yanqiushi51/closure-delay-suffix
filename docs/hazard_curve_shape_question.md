# Exit Hazard 曲线形态控制问题

## 背景

我们正在研究一个在线推理进程代理 `exit_hazard`。给定模型已经生成的 prefix，它输出当前推理是否接近收束、是否准备退出 reasoning 的信号。这个代理不是事后长度统计，而是在每个 token 位置都可以由 logits / probe 特征即时计算。

目前我们已经训练了一个增强版 `exit_hazard` 代理，并在 Qwen2.5-7B-Instruct 的 GSM8K 推理轨迹上验证：

- 代理整体具有较强的在线单调趋势，综合分数约为 `0.906`。
- 它与 token 长度本身的相关性很低，不是简单长度计数。
- token-level jump 分析显示，大的 hazard 跃升不只出现在最终答案标记附近，也常出现在建模完成、关键计算完成、检查/复核开始等语义阶段。
- 用 phrase-level CEM 优化出的自然语言 suffix 可以在保持正确率基本不变的情况下显著增加推理长度。例如在 100 条评测上，平均长度增加约 `+52` tokens，正确率保持 `0.96 -> 0.96`。
- 但这个强 suffix 的平均 hazard 反而上升，而不是下降。因此，过度思考攻击似乎不应该被建模为“压低 exit_hazard”，而应该被建模为“控制 exit_hazard 曲线形态”。

## 核心观察

设一条推理轨迹的在线代理序列为：

\[
h_1, h_2, \ldots, h_T
\]

其中 \(h_t = f(p_t)\)，\(p_t\) 是前 \(t\) 个 token 的 prefix。

我们现在的理解是：

- \(h_t\) 较低：模型仍处在开放探索、问题建模、未稳定收束的阶段。
- \(h_t\) 明显跃升：模型完成了某个局部推理阶段，例如关键变量关系确定、主要计算路径确定、答案候选基本形成。
- \(h_t\) 高位平台：模型已经有较强收束倾向，但仍可能继续做复核、重算、解释或长尾验证。
- 最终再次上升或保持高位：模型进入显式答案输出阶段。

因此，针对“诱导过度思考”的 suffix 优化，目标可能不是让 \(h_t\) 全程变低，而是让曲线表现出某种阶段结构，例如：

1. 前期不要过早出现陡峭跃升；
2. 中期允许缓慢上升，但避免快速进入最终退出状态；
3. 后期维持一个复核/验证平台段；
4. 最后仍允许模型正常给出答案，以保持正确性。

## 想请教的问题

我们希望把这个问题形式化为一个可优化的数学目标。请问下面哪种建模方式更合理？

### 1. Hazard 曲线的最优控制问题

是否可以把 \(h_t\) 看成一个在线 hazard / survival process，然后把 suffix 看成控制变量 \(u\)，目标是优化曲线形态：

\[
\min_u \mathcal{L}_{shape}(h_{1:T}(u)) + \lambda \mathcal{L}_{answer}(u)
\]

其中 \(\mathcal{L}_{answer}\) 保证答案正确或至少不破坏原本答案，\(\mathcal{L}_{shape}\) 不再是简单压低 hazard，而是控制跃升时间、斜率、平台长度和曲线面积。

例如：

\[
\mathcal{L}_{shape}
= a \cdot \mathrm{EarlyJump}
+ b \cdot \mathrm{SlopePenalty}
- c \cdot \mathrm{PlateauReward}
+ d \cdot \mathrm{PrematureExitPenalty}
\]

这种形式是否有理论上的合理性？如果有，哪些曲线统计量最值得控制？

### 2. 跃升延迟目标

如果大的 hazard jump 对应“局部收束事件”，那么过度思考是否可以被定义为延迟关键收束跃升：

\[
\tau_{jump} = \arg\max_t (h_t - h_{t-1})
\]

目标变成让 \(\tau_{jump}\) 尽量靠后，同时保持答案正确：

\[
\max_u \tau_{jump}(u) - \lambda \mathcal{L}_{answer}(u)
\]

但 \(\arg\max\) 不容易直接优化。是否可以用 softargmax、分位数 crossing time、first-passage time 等方式做可导近似？

### 3. 高位平台目标

我们的实验显示，强 suffix 往往不是让 hazard 下降，而是让模型在高 hazard 状态下继续复核和展开。是否可以把“过度思考”定义为高位平台段变长：

\[
\mathrm{PlateauLength}
= |\{t: \ell < h_t < r,\ t < \tau_{answer}\}|
\]

也就是说，模型已经接近可以退出，但被 suffix 诱导继续进行检查、重算和解释。这是否比“低 hazard 持续时间”更符合过度思考的定义？

### 4. 曲线分布匹配 / detector-guided shaping

我们是否应该先收集两类轨迹：

- 正常简洁推理轨迹；
- 过度思考或长尾复核轨迹；

然后训练一个曲线级 detector \(D(h_{1:T})\)，再优化 suffix 让生成轨迹的 hazard 曲线更像过度思考分布：

\[
\max_u D(h_{1:T}(u)) - \lambda \mathcal{L}_{answer}(u)
\]

或者使用 MMD / Wasserstein distance 做曲线分布匹配。这种方式是否比手写 shape loss 更稳健？

### 5. Teacher-forced 曲线和 free-generation 曲线的差异

目前 suffix 优化阶段常常用 teacher-forced baseline response 来估计代理曲线，因为这样便宜且稳定；但最终攻击效果要在 free generation 上验证。请问是否有必要把优化改成两阶段：

1. 用 teacher-forced shape loss 快速筛选候选 suffix；
2. 对 top-k suffix 做真实 free-generation 评估，再用 CEM / bandit / Bayesian optimization 更新候选分布。

这个两阶段框架是否是处理分布偏移的合理方式？

## 我们希望得到的指导

我们不想再把目标写成“让 hazard 越低越好”。更准确的问题是：

> 在正确性约束下，如何从数学上定义一种合理的 `exit_hazard` 曲线形态目标，使 suffix 优化能够稳定诱导过度思考、长尾复核或延迟退出？

希望老师帮助判断：

1. `exit_hazard` 应该被理解为 exit readiness、semantic closure，还是 survival hazard？
2. 对过度思考攻击来说，最合理的曲线目标是 delayed jump、controlled slope、long plateau，还是 distribution matching？
3. 哪种目标更容易给出数学解释，并且适合后续实验验证？
