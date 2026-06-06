# 问题：如何用在线代理刻画并操纵模型的推理退出行为

## 1. 我们真正想研究的对象

我们的任务不是单纯让模型输出更长，也不是让模型回答错误。我们真正关心的是：

> 能否找到一个在线可计算的推理状态代理，并利用它构造 suffix，使模型在保持答案正确的同时，更倾向于继续推理、复核或长尾展开。

也就是说，我们希望研究的是模型的 **reasoning exit behavior**：

- 模型什么时候认为自己已经接近完成？
- 模型什么时候从开放推理进入答案收束？
- 什么样的 suffix 会推迟退出、诱导复核、或者制造过度思考？
- 这些行为能否通过一个在线代理变量被量化并优化？

因此，最终攻击目标可以写成：

\[
\text{induce overthinking} \quad \text{subject to answer correctness}
\]

其中正确性可以由单独的 answer loss 或最终评测保证。代理本身主要负责刻画推理过程，而不是直接评判答案质量。

## 2. 我们当前的代理：exit_hazard

对一条生成轨迹，设 prefix 为：

\[
p_t = (x, y_1, \ldots, y_t)
\]

其中 \(x\) 是题目和 prompt，\(y_1,\ldots,y_t\) 是模型已经生成的 token。

我们定义一个在线代理：

\[
h_t = f(p_t)
\]

其中 \(h_t\) 只依赖当前 prefix，可以在生成过程中即时计算。

当前实现中的 `exit_hazard` 不是简单长度指标，也不是只检测 `Final answer`、`\boxed{}` 之类的表层标记。它主要来自两类信号：

1. **logit/probe 信号**：比较模型在当前 prefix 下更倾向于继续推理，还是进入答案/结论表达。
2. **累积与形态信号**：跟踪这些倾向随 token 推进的变化，例如 margin、runmax、正负累积等。

直观上，我们目前把 \(h_t\) 解释为：

> 当前 prefix 下，模型进入答案收束或退出 reasoning 的 readiness。

它更接近 **exit readiness / semantic closure**，而不是“输出还剩多少 token”的直接预测。

## 3. 为什么它能服务于攻击

过度思考攻击的关键不是直接把长度写进优化目标，因为长度是事后变量，且容易引入无意义灌水。我们希望操纵的是生成过程中的内部倾向：

\[
\text{suffix} \rightarrow \text{reasoning state trajectory} \rightarrow \text{longer but still correct reasoning}
\]

`exit_hazard` 在这个链条中承担中间变量：

\[
u \rightarrow h_{1:T}(u) \rightarrow \text{generation behavior}
\]

其中 \(u\) 是要优化的 suffix。

如果一个 suffix 真的诱导了过度思考，它不一定会让 \(h_t\) 全程降低。我们的实验反而显示，强 suffix 往往会让模型在“已经接近收束”的状态下继续做复核和展开。因此，更合理的攻击逻辑是：

> 不是让模型永远不知道答案，而是让模型已经知道答案后仍然继续检查、重算和解释。

这对应的不是低 hazard，而可能是 **高 hazard 平台段变长**。

## 4. 当前实验现象

我们在 Qwen2.5-7B-Instruct + GSM8K 上得到几个现象：

1. 增强版 `exit_hazard` 代理具有较强的在线趋势，综合分数约为 `0.906`。
2. 代理与 token 长度相关性很低，因此不是简单长度计数。
3. token-level jump 分析显示，大的 hazard 跃升经常对应语义阶段变化，例如：
   - 问题建模完成；
   - 关键变量关系确定；
   - 主要计算路径确定；
   - 开始复核或重算；
   - 最终答案表达。
4. phrase-level CEM 搜到的自然语言 suffix 可以在保持正确率基本不变的情况下增加推理长度。100 条评测中，平均长度约增加 `+52` tokens，正确率保持 `0.96 -> 0.96`。
5. 但是这个强 suffix 的平均 hazard 不是下降，而是上升。

这说明：

\[
\text{overthinking} \neq \text{low hazard everywhere}
\]

更可能是：

\[
\text{overthinking} = \text{delayed exit or prolonged post-closure reasoning}
\]

也就是模型已经进入某种收束状态，但被 suffix 诱导继续执行检查、复述、重算或验证。

## 5. 我们现在需要重新定义优化目标

原始想法是：

\[
\min_u \sum_t h_t(u) + \lambda \mathcal{L}_{answer}(u)
\]

但这个目标可能是错的。因为它会鼓励模型始终处于不收束状态，而这未必产生有效的过度思考，也可能破坏答案稳定性。

更合理的目标应该控制曲线形态：

\[
\min_u \mathcal{L}_{shape}(h_{1:T}(u)) + \lambda \mathcal{L}_{answer}(u)
\]

其中 \(\mathcal{L}_{answer}\) 保证答案不被破坏，\(\mathcal{L}_{shape}\) 则描述我们想诱导的推理行为。

我们目前考虑的形态目标包括：

### 5.1 延迟过早收束

定义第一次达到高 hazard 的时间：

\[
\tau_c = \min\{t: h_t \ge c\}
\]

如果 \(\tau_c\) 太早，说明模型过早进入答案收束状态。攻击可以尝试最大化：

\[
\tau_c
\]

或者惩罚早期 crossing：

\[
\mathrm{EarlyExitPenalty}
= \sum_{t < \alpha T} \max(0, h_t - c)
\]

### 5.2 控制跃升斜率

定义局部跃升：

\[
\Delta h_t = h_t - h_{t-1}
\]

如果前中期出现很大的正跃升，可能表示模型快速完成了主要推理并准备退出。攻击可以惩罚早期大跃升：

\[
\mathrm{JumpPenalty}
= \sum_{t < \alpha T} \max(0, \Delta h_t - m)
\]

### 5.3 拉长高位复核平台

如果过度思考的本质是“已经接近完成但继续复核”，则可以奖励高位平台段：

\[
\mathrm{PlateauLength}
= |\{t: \ell \le h_t \le r,\ t < \tau_{answer}\}|
\]

这比单纯压低 hazard 更符合我们的实验现象。

### 5.4 曲线分布匹配

另一种方式是不手写 shape loss，而是收集两类轨迹：

- 正常简洁推理；
- 过度思考/长尾复核推理。

然后训练曲线级判别器：

\[
D(h_{1:T}) \rightarrow [0,1]
\]

表示一条 hazard 曲线像不像过度思考轨迹。优化 suffix 时使用：

\[
\max_u D(h_{1:T}(u)) - \lambda \mathcal{L}_{answer}(u)
\]

这相当于学习“过度思考曲线分布”，而不是人为指定某一种曲线形态。

## 6. 我们想请教老师的核心问题

我们现在想问的不是“有没有一个完美单调代理”，而是下面这个更具体的问题：

> 如果 `exit_hazard` 可以近似刻画模型的 reasoning exit readiness，那么在正确性约束下，应该如何定义一个数学上合理、实验上可优化的曲线形态目标，使 suffix 能稳定诱导过度思考、长尾复核或延迟退出？

具体想请教：

1. `exit_hazard` 应该被形式化为 exit readiness、semantic closure，还是 survival analysis 里的 hazard rate？
2. 对过度思考攻击来说，目标应该是：
   - 延迟第一次高 hazard crossing；
   - 抑制早期大跃升；
   - 拉长高位平台段；
   - 还是学习过度思考曲线分布？
3. “高 hazard 但继续生成”是否可以被看作过度思考的核心表征？也就是模型已经具备退出条件，但 suffix 诱导它继续复核和展开。
4. 如果 teacher-forced 曲线和 free-generation 曲线存在分布偏移，是否应该采用两阶段优化：
   - teacher-forced shape loss 快速筛选候选 suffix；
   - free-generation 评估更新 CEM / bandit / Bayesian optimization？
5. 有没有更合适的数学工具来描述这个问题，例如 optimal control、survival process、first-passage time、change-point detection、distribution matching 或 imitation learning？

## 7. 我们希望最终形成的攻击框架

理想情况下，攻击框架应该是：

\[
\text{suffix } u
\rightarrow
\text{hazard curve } h_{1:T}(u)
\rightarrow
\text{controlled reasoning state}
\rightarrow
\text{longer correct reasoning}
\]

其中：

- `exit_hazard` 是在线推理状态代理；
- `shape loss` 是攻击优化目标；
- `answer loss` 或最终 correctness eval 保证答案不崩；
- 输出长度只是最终攻击效果的外部验证，不是代理本身。

因此，我们希望老师帮助判断：

> 这个“用在线代理曲线控制推理退出行为”的问题定义是否成立？如果成立，最合理的曲线目标应该怎么定义？
