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

## 5. 更准确的数学对象：两个事件过程

原始想法是：

\[
\min_u \sum_t h_t(u) + \lambda \mathcal{L}_{answer}(u)
\]

但这个目标可能是错的。因为它会鼓励模型始终处于不收束状态，而这未必产生有效的过度思考，也可能破坏答案稳定性。

更准确的建模方式是把推理过程拆成两个事件：

1. **Closure event** \(\tau_C\)：模型已经具备答案收束条件，或者进入 semantic closure / exit readiness 状态。
2. **Answer-onset event** \(\tau_A\)：模型开始显式最终答案表达，例如 `Final answer`、`Answer:`、`\boxed{}`、EOS/EOT 等。

过度思考不等于 \(\tau_C\) 越晚越好。真正关键的是：

\[
\tau_C \le t < \tau_A
\]

这段区间变长。也就是说，模型已经接近可以回答，但仍继续复核、重算、解释或展开。

因此，`exit_hazard` 当前更准确地说是 closure/readiness evidence，而不是严格 survival analysis 意义下的 calibrated hazard rate。如果要写成严格 hazard，可以进一步定义：

\[
\lambda^C_t
=
\Pr(\tau_C=t\mid \tau_C\ge t,p_t)
=
\sigma(b(t)+g_\phi(p_t))
\]

当前实现里，我们先使用现有 cumulative closure evidence 的 soft relaxation：

\[
q^C_t
=
\sigma\left(\frac{H^C_t-c_C}{\epsilon_C}\right)
\]

其中 \(H^C_t\) 可以取 `exit_hazard_cumprob` 或 `exit_hazard_cumlogit`。

再定义 answer-onset survival：

\[
\lambda^A_t
=
\sigma\left(\frac{M^A_t-c_A}{\epsilon_A}\right),
\quad
S^A_t
=
\prod_{s\le t}(1-\lambda^A_s)
\]

其中 \(M^A_t\) 来自 final-answer markers / EOS / answer phrase 的 probe logmass。

实现上需要区分两种情况：

- 如果 \(\lambda^A_t\) 已经由 calibrated onset hazard head 训练得到，可以使用 cumulative survival \(\prod_{s\le t}(1-\lambda^A_s)\)。
- 如果 \(M^A_t\) 只是未校准的 marker/probe logmass，则不应直接连乘。否则长回答会因为许多很小的未校准 \(\lambda^A_t\) 被天然惩罚，PCG/VPCG 反而和过度思考长度反向。当前实现默认使用 local answer survival：

\[
S^A_t = 1-\lambda^A_t
\]

也就是只惩罚当前位置明显进入 final-answer/onset 的倾向，而不把未校准 probe 当成严格 hazard 连乘。

最核心的 post-closure gap 是：

\[
\operatorname{PCG}(u)
=
\sum_t
q^C_t(u)S^A_t(u)
\]

它表示模型已经 closure/readiness 较高，但尚未进入 final answer onset 的累计时间。

## 6. 主攻击目标：Verified Post-Closure Gap

为了区分有效复核和无意义漂移，我们再定义：

\[
B_t
=
\Pr(\text{verification / recomputation / alternative branch at }t\mid p_t)
\]

\[
D_t
=
\Pr(\text{drift / service chatter / unrelated continuation at }t\mid p_t)
\]

最终主目标是：

\[
\operatorname{VPCG}(u)
=
\sum_t
q^C_t(u)
S^A_t(u)
B_t(u)
(1-D_t(u))
\]

这对应我们现在对 overthinking 的更准确定义：

> overthinking = prolonged verified survival after closure but before answer onset.

也就是：

\[
\text{high closure readiness}
+
\text{answer survival}
+
\text{verification behavior}
-
\text{drift}
\]

suffix 优化目标可以写成：

\[
\min_u
\mathbb E_x[
\mathcal L_{\text{shape}}(x,u)
]
+
\lambda\mathcal L_{\text{answer}}(x,u)
\]

其中：

\[
\mathcal L_{\text{shape}}
=
-w_g
\frac{1}{T}
\sum_t
q^C_t S^A_t B_t(1-D_t)
+
w_e
\frac{1}{T_0}
\sum_{t<T_0}
q^C_t
+
w_j
\frac{1}{T_1}
\sum_{t<T_1}
\operatorname{softplus}
\left(
\frac{q^C_t-q^C_{t-1}-m}{\epsilon_j}
\right)
+
w_d
\frac{1}{T}
\sum_t
S^A_tD_t
\]

第一项奖励 post-closure verification plateau，是主目标。

第二项惩罚太早 closure，只是 regularizer。

第三项惩罚早期突发大跃升，也是 regularizer。

第四项惩罚 drift。

answer loss 只保证答案稳定，不代替 shape objective。

## 7. 四类目标的优先级

### 7.1 第一优先级：VPCG

这是最符合实验现象的目标：

\[
\sum_t q^C_tS^A_tB_t(1-D_t)
\]

它解释了为什么强 suffix 的 average hazard 反而上升：suffix 不是让模型不知道答案，而是让模型在接近知道答案后继续验证。

### 7.2 第二优先级：多通道曲线分布匹配

可以训练曲线级 detector，但输入不应该只有 \(H^C_t\)，而应该是多通道序列：

\[
z_t =
[
H^C_t,\;
\lambda^A_t,\;
B_t,\;
D_t,\;
\Delta H^C_t
]
\]

优化：

\[
\max_u
D_\psi(z_{1:T}(u))
-
\lambda\mathcal L_{\text{answer}}(u)
\]

### 7.3 第三优先级：延迟 high-hazard crossing

delayed crossing 有用，但不能作为主目标。最大化 \(\tau_C\) 可能诱导模型一直不 closure，变成 confusion / underthinking，而不是 overthinking。因此它更适合作为 early-closure penalty。

### 7.4 第四优先级：抑制早期大跃升

jump penalty 更像平滑正则，用来防止前 20%-40% 轨迹中突然进入 answer-ready 状态，但它不是 overthinking 的核心定义。

## 8. Teacher-forced 与 Free-generation 的两阶段协议

真实目标是：

\[
J(u)
=
\mathbb E_{y\sim P_\theta(\cdot\mid x,u)}
[
\Phi(y,h(y;u))
\mathbf 1\{\text{correct}(y)\}
]
\]

teacher-forced 优化的是 baseline trajectory 上的近似目标：

\[
\tilde J(u)
=
\mathbb E_{\bar y\sim q}
[
\Phi(\bar y,h(\bar y;u))
]
\]

两者存在 off-policy 偏移。因此合理协议是：

1. 用 teacher-forced VPCG shape loss 快速筛 top-K suffix。
2. 对 top-K suffix 做真实 free-generation 评测。
3. 用 CEM / bandit / Bayesian optimization 根据 free-generation 指标更新候选分布。
4. 把 top suffix 的 free-generation 轨迹加入训练集，重新校准 heads 或 curve discriminator。

free-generation 评测应该至少报告：

- accuracy；
- length ratio；
- PCG；
- VPCG；
- answer onset delay；
- drift rate；
- teacher-forced score 与 free-generation score 的 rank correlation。

## 9. 我们现在采纳的攻击框架

最终框架是：

\[
\text{suffix } u
\rightarrow
\text{process channels } z_{1:T}(u)
\rightarrow
\operatorname{VPCG}(u)
\rightarrow
\text{longer correct post-closure verification}
\]

其中：

- `exit_hazard` 提供 closure/readiness evidence；
- answer-onset probe 提供 \(S^A_t\)，未校准时使用 local survival，校准后可升级为 cumulative survival；
- verification probe 提供 \(B_t\)。当前实现使用 hybrid verification evidence：绝对 verification/branch probe 加轨迹内相对 evidence，避免单一 first-token probe 过稀疏；
- drift probe 提供 \(D_t\)；
- VPCG 是主 shape objective；
- early crossing 和 jump penalty 是辅助正则；
- answer loss / correctness eval 保证答案不崩；
- 输出长度只是最终外部效果，不是优化目标本身。

一句话概括：

\[
\boxed{
\text{overthinking}
=
\text{prolonged verified survival after closure but before answer onset}
}
\]
