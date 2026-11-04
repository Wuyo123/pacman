# 方向 B 完备研究计划与 Agent 执行交接方案：基于路口地标 MiTA 交叉注意力信念聚合与死胡同安全屏蔽的 Pac-Man CTF 协同博弈学习（Belief-MiTA CTF）

**文档编号**：`AGENT-SPEC-PACMAN-BELIEF-MITA-20261104`  
**整理日期**：2026-11-04  
**用途**：直接交付给下游研发/实验执行 Agent 的完备技术规范、实验方案与止损协议（遵循 `experiment-plan` 与 `experiment-design` 技能规范）。  
**硬件与预算硬约束**：**100% 本地闭环（0 元租卡费用）**，全部数据采集、神经网络训练与锦标赛对局评测均在本地 RTX 3050 Ti Laptop（4GB 显存）与 CPU（纯 Python 环境）上完成；每周投入约 20 小时；目标会议级别为 CCF-B / CCF-C（IEEE CoG、AAMAS、ECAI、PRICAI、ICTAI）。

## 一、 执行 Agent 必读：赛道定位与核心设计原则（Phase 0: Context & Principles）

接手本项目的 Agent **必须严格遵守**以下学术定位与工程纪律，避免将项目做成简单的“课程大作业报告”：

1. **严禁做成全图可见的简单吃豆游戏**：
   * 必须严格采用 [UC Berkeley CS188 Pac-Man Capture the Flag (CTF) 官方锦标赛规则](https://berkeleyai.github.io/cs188-website/projects.html)（2v2 红蓝对抗 + **仅视野范围 5 格内可见、全图仅提供带均匀噪声 $\pm 6$ 的曼哈顿距离观测**）。
   * 论文的核心学术落脚点是：**非完备信息下的混合协作-对抗多智能体博弈（Partially Observable Mixed Cooperative-Competitive Game）中的“联合信念状态估计（Joint Belief Estimation）”与“安全屏蔽引导博弈树搜索（Shielded Neural-Guided MCTS）”**。
2. **坚持“离线全局专家模仿学习 + 辅助信念监督”，拒绝盲目端到端 MARL**：
   * 在稀疏奖励的 CTF 中直接从零跑 QMIX / MAPPO 极难收敛且策略脆弱。本项目采用**全知视角离线搜索专家（Full-Observability Oracle Search Expert）**生成高水平攻防演示轨迹，通过**多任务模仿学习（同时监督“敌方真实位置信念热力图”与“己方双智能体协同动作”）**训练轻量级注意力网络，本地 RTX 3050 Ti 在 1–2 小时内即可完成全部训练。
3. **坚持“神经先验 + 拓扑安全屏蔽 + 协同 MCTS”三结合**：
   * 纯神经网络在对抗极值状态下容易犯低级致命错误（如在敌方幽灵逼近时钻入只有唯一出口的死胡同吃豆）。必须结合图论预计算的**死胡同深度安全屏蔽器（Dead-End Safety Shield）**与轻量协同 MCTS，确保对阵人类顶尖规则智能体时具备压倒性胜率。

## 二、 核心问题锚点与方法论创新设计（Method Specification）

### 1. 现有 Pac-Man CTF 方法的真实瓶颈（Problem Anchor）

* **传统粒子滤波 + 搜索基线的割裂与退化**：
  现有高分开源方案（如 [lzheng1026 的粒子滤波 + Alpha-Beta](https://github.com/lzheng1026/Pacman-Capture-The-Flag) 或 [Jaden Stock 等人的共享粒子滤波 MCTS](https://jadenstock.files.wordpress.com/2017/03/573_ai_final_report.pdf)）面临两大痛点：
  1. **多模态粒子坍缩与拓扑盲目性**：当敌方潜入者在迷宫分支路口隐蔽移动时，普通粒子滤波缺乏对“豆子消失事件”、“地图咽喉要道（Chokepoints）”以及“己方双智能体历史观测交叉几何约束”的深层时空推理，取均值或众数点会导致向错误走廊扑空。
  2. **双智能体联合搜索深度受限与死胡同陷阱**：双智能体每步有 $|\mathcal{A}|^2 = 5 \times 5 = 25$ 个联合动作，加之敌方响应，[协同 MCTS](https://github.com/RuilinM/MCTS-in-Pacman-Capture-The-Flag-environment) 在单步 $50\sim 100\,\text{ms}$ 内只能展开 $2\sim 3$ 层，无法看穿长死胡同（Dead-End Corridor）的围剿陷阱。

### 2. 本文方法：`Belief-MiTA CTF` 三大核心模块与数学公式

#### 模块一：基于 MiTA 思想的路口地标交叉注意力信念-策略网络（Chokepoint-Landmark MiTA Network）

地图网格通常包含 $W \times H \approx 32 \times 16 = 512$ 个单元格。若对过去 $L=8$ 帧的全图网格特征序列做全局自注意力，序列长度达 $4096$，无法满足单步毫秒级实时搜索需求。我们引入 [MiTA Attention (NeurIPS 2026, arXiv:2602.01219)](https://arxiv.org/abs/2602.01219)（[官方实现 QishuaiWen/MiTA](https://github.com/QishuaiWen/MiTA)）的“地标查询（Landmark Queries）+ 可复用变形 Top-$k$ 路由与窄 MLP 压缩”机制：

1. **地图拓扑地标提取（Topological Chokepoint & Cluster Landmarks）**：
   每局开始前，用 Tarjan 割点/桥算法与豆群聚类在地图上自动提取 $M=16$ 个关键拓扑地标（包括中线咽喉路口、能量豆位置、防守豆群中心），编码为地标查询向量 $Q_{\text{land}} \in \mathbb{R}^{M \times d}$（$d=64$）。
2. **双智能体时空观测序列编码（Spatio-Temporal Observation Tokens）**：
   将过去 $L=8$ 步己方两个智能体的位置、带噪敌方距离环（Noisy Distance Rings，即满足 $|d_{\text{manhattan}}(v, \text{agent}_i) - \tilde{d}_{i, e}| \le 6$ 的可行网格掩码）、刚刚被吃掉的豆子坐标差编码为特征序列 $X_{\text{obs}} \in \mathbb{R}^{N_v \times d}$（$N_v$ 为可行走网格总数，约 $200\sim 300$）。
3. **MiTA 变形 Top-$k$ 路由与全局压缩（Deformable Belief Aggregation）**：
   * **稀疏路由专家**：$M$ 个路口地标查询 $Q_{\text{land}}$ 在全图可行网格 $K_{\text{obs}}$ 上执行 Top-$k$（$k=12$）激活路由，自动聚焦于“符合带噪距离交集且靠近咽喉要道”的高疑点区域，形成 $M$ 个变形局部信念专家 $E_{\text{deform}} \in \mathbb{R}^{M \times d}$。
   * **全局压缩专家**：用窄线性投影将全图历史状态压缩为 $M_c=4$ 个全局态势摘要 Token $E_{\text{comp}} \in \mathbb{R}^{M_c \times d}$。
   * **双头协同读出（Dual-Head Readout）**：
     - **Head 1（显式敌方信念热力图预测头）**：输出敌方两个智能体在全图可行网格上的概率分布 $\hat{B}^{(e_1)}, \hat{B}^{(e_2)} \in \Delta^{N_v}$。
     - **Head 2（双智能体协同动作先验与价值头）**：以己方两个智能体特征为 Query，交叉注意力读取 $[E_{\text{deform}}; E_{\text{comp}}]$，输出联合动作先验（分解为条件或边际先验 $P_\theta(a_1 \mid s), P_\theta(a_2 \mid s, a_1) \in \Delta^5$）以及当前局面胜率估值 $V_\theta(s) \in [-1, 1]$。

#### 模块二：全知视角离线专家采集与多任务模仿学习（Oracle Demonstration & Multi-Task IL）

* **全知专家（Full-State Oracle Planner）构造**：
  在离线数据生成阶段，允许专家访问真实敌方位置 $(x_{e_1}^*, x_{e_2}^*)$，结合双鬼协同围堵最短路割点算法（防守）与带安全距离约束的 Alpha-Beta / MCTS（进攻），对阵多样化对手池（包括 [COMP90054 规则智能体](https://github.com/infinityglow/COMP90054-Pacman-Contest-Project)、MCTS 智能体、随机扰动智能体）采集 $1,000$ 局高质量轨迹（约 $250,000$ 个状态步）。
* **多任务模仿损失函数（Multi-Task Imitation Loss）**：
  同时监督动作先验、局面价值与**真实敌方位置热力图**（以真实敌方坐标为中心的高斯平滑核 $B^*$ 为软标签，迫使 MiTA 注意力真正学会从带噪距离推演敌方隐状态）：
  $$\mathcal{L} = \text{CE}(P_\theta(a_1, a_2), a_{\text{oracle}}^*) + \lambda_{\text{bel}} \sum_{e \in \{1, 2\}} D_{\text{KL}}\big(B^{*(e)} \,\|\, \hat{B}_\theta^{(e)}\big) + \lambda_{\text{val}} \big(V_\theta(s) - z_{\text{outcome}}\big)^2$$

#### 模块三：死胡同拓扑安全屏蔽引导的协同 MCTS（Dead-End Shielded Cooperative MCTS）

在线对战时，每步决策预算严格限制在 $50\,\text{ms}$ 内：

1. **死胡同拓扑安全屏蔽器（Dead-End Topology Shield）**：
   借鉴 [Safe RL via Shielding (AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/11797/11656)，在开局初始化时用双连通分量（Biconnected Components）预计算每个死胡同单元格 $u$ 的唯一逃生入口 $g(u)$ 及逃生深度 $d_{\text{exit}}(u)$。在进攻状态下，若智能体进入动作 $a$ 后的逃生入口距离 $d(u, g(u))$ 大于等于信念分布 $\hat{B}$ 中敌方非恐慌幽灵到达 $g(u)$ 的最小步数减去安全裕度 $\delta=2$：
   $$d(u, g(u)) + \delta \ge \min_{v: \hat{B}(v) > \tau} d_{\text{maze}}(v, g(u))$$
   则安全屏蔽器直接将动作 $a$ 的先验概率置零（Hard Shielding），彻底杜绝“钻死胡同被瓮中捉鳖”。
2. **先验裁剪协同 MCTS**：
   在屏蔽后的合法动作集上，仅保留神经网络先验概率最高的 Top-3 动作组合（将联合分支因子从 $25$ 压降至 $\le 6$），以 $\hat{B}_\theta$ 采样敌方假想位置展开轻量 PUCT 搜索，叶节点直接调用 $V_\theta(s)$ 估值而无需慢速随机 Rollout。

## 三、 论文核心主张映射表（Phase 1: Freeze the Paper Claims）

| 主张编号 | 核心主张内容（Claim） | 为什么重要（Why It Matters） | 必须排除的竞争解释（Anti-Claim to Rule Out） | 最小充分证据标准（Minimum Convincing Evidence） | 关联实验块 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C1（主主张：综合对抗胜率与跨地图泛化）** | 在标准及随机生成的 Pac-Man CTF 地图上，`Belief-MiTA CTF` 在相同单步时间预算（$50\,\text{ms}$）内，对阵经典粒子滤波+Alpha-Beta、协同 MCTS 以及竞赛级 FSM 规则智能体均取得显著超过 $65\%$ 的对战立胜率。 | 直接证明“模仿学习 + MiTA 信念注意力 + 安全屏蔽 MCTS”在非完备信息博弈中的实战统治力。 | **反驳 1**：智能体只是死记硬背了 `defaultCapture` 单张地图的路线，换到未见随机地图（`RANDOM[seed]`）就会崩溃。 | 在默认地图与 20 张未见随机生成地图（`RANDOM` Layouts）上，对三大类基线对手的综合胜率均 $\ge 65\%$（双侧二项/符号检验 $p < 0.01$）。 | **Block 1, Block 2** |
| **C2（辅主张：MiTA 信念聚合与安全屏蔽的独立贡献）** | (1) 结合辅助信念损失的路口地标 MiTA 注意力在敌方位置估计精度（L1 距离误差与 NLL）上显著优于独立/共享粒子滤波；(2) 死胡同安全屏蔽与先验剪枝使有限预算下的死亡被抓次数下降 $\ge 40\%$。 | 拆解证明深度学习模块（MiTA 信念估计）与符号安全模块（Shielded MCTS）各自发挥了不可替代的作用。 | **反驳 2**：只要加上死胡同屏蔽规则，用普通粒子滤波 + MCTS 也能赢，不需要 MiTA 注意力网络；或者纯反应式网络不配 MCTS 也能赢。 | 单变量消融证明：(a) 保持相同 Shielded MCTS 时，用 `Belief-MiTA` 替代粒子滤波胜率显著提升 $\ge 12\%$；(b) 去掉安全屏蔽或去掉 MCTS 均导致胜率明显回落。 | **Block 3, Block 4** |

## 四、 四大实验块与单变量隔离矩阵（Phase 2 & 3: Experiment Blocks）

### Block 1（B1）：多对手池锦标赛主基准实验（Main Tournament Matrix）

* **检验主张**：**C1**（综合胜率、平均净胜分与跨地图泛化）。
* **测试地图集**：
  1. **已知标准图（Seen Layouts）**：`defaultCapture`、`mediumCapture`、`alleyCapture`、`crowdedCapture`。
  2. **未见随机拓扑图（Unseen Generalization Layouts）**：使用官方生成器 `-l RANDOM[seed]` 固定 20 个未在训练中出现的地图种子。
* **对手基线池（Opponent Pool，涵盖全部四大主流流派）**：
  1. `Baseline-Heuristic`：官方强化进攻/防守基线 + 经典最短路策略。
  2. `PF-AlphaBeta`：复现 [lzheng1026](https://github.com/lzheng1026/Pacman-Capture-The-Flag) 的粒子滤波敌方位置估计 + 深度 3 Alpha-Beta 极小化极大博弈搜索。
  3. `Hybrid-MCTS`：复现 [neeilnandal](https://github.com/neeilnandal/pacman-mcts-agent-v2) 与 [RuilinM](https://github.com/RuilinM/MCTS-in-Pacman-Capture-The-Flag-environment) 的粒子滤波 + 双智能体协同 UCT 搜索（相同 $50\,\text{ms}$/步预算）。
  4. `Champion-FSM`：复现 [COMP90054 竞赛高分智能体 (infinityglow)](https://github.com/infinityglow/COMP90054-Pacman-Contest-Project) 的四模态状态机 + BFS 死胡同规避策略。
  5. `Pure-MARL (QMIX)`：在相同环境奖励下训练的 CTDE 强化学习基线。
* **评价指标**：胜率（Win Rate %，平局计 0.5 胜）、平均净胜分差（Average Score Margin）、进攻端单次潜入平均带回豆数、防守端平均成功拦截步数。每对组合在每类地图上交换红蓝半场各打 50 局（共 100 局配对比赛）。

### Block 2（B2）：非完备信息下的敌方定位精度专项评测（Belief Tracking Accuracy）

* **检验主张**：**C2**（验证路口地标 MiTA 交叉注意力相比传统粒子滤波的状态估计优势）。
* **测试设计**：在 100 局标准对局（仅在智能体视野外、即依赖带噪距离观测的时间步）上，同步运行以下 4 种信念估计器而不干预走子，对比其预测分布 $\hat{B}$ 与真实敌方坐标 $v^*$ 的误差：
  1. `Independent-PF`（单智能体独立粒子滤波，粒子数 $P=500$）。
  2. `Shared-PF`（双智能体共享测量更新粒子滤波，[Jaden Stock 2017](https://jadenstock.files.wordpress.com/2017/03/573_ai_final_report.pdf)）。
  3. `Conv-LSTM-Belief`（无地标路由的普通卷积循环信念网络）。
  4. `Belief-MiTA`（本文路口地标压缩 + Top-$k$ 变形路由信念网络）。
* **评价指标**：真实敌方位置处的负对数似然（NLL，$-\log \hat{B}(v^*)$）、Top-3 预测区域命中率（Hit@3 %）、期望曼哈顿距离误差（Expected L1 Error）。

### Block 3（B3）：单变量隔离消融矩阵（Single-Variable Isolation Matrix）

* **检验主张**：**C2**（在 10 张代表性地图 × 对阵最强对手 `Champion-FSM` 与 `Hybrid-MCTS` 共 100 局比赛上，严格每次仅变动一个组件）：

| Run ID | 隔离变量（Factor Varied） | 变量取值（Factor Value） | 固定配置（Fixed Config） | 检验目的与预期结果 |
| :--- | :--- | :--- | :--- | :--- |
| **B0 (Baseline MCTS)** | 信念与引导机制 | `Shared-PF` + 均匀先验 MCTS（无网络、无死胡同屏蔽） | $50\,\text{ms}$/步搜索预算 | 传统博弈树搜索基准 |
| **B1 (+DeadEnd Shield)** | 安全屏蔽器 | `Shared-PF` + **开启死胡同拓扑屏蔽** + 均匀先验 MCTS | $50\,\text{ms}$/步搜索预算 | 隔离验证死胡同拓扑屏蔽单独带来的防暴毙收益 |
| **B2 (+Pure Policy IL)** | 神经网络引导 | 开启网络动作先验与价值头，但**关闭辅助信念头（$\lambda_{\text{bel}}=0$）** | 开启死胡同屏蔽 + MCTS | 证明若不显式监督敌方信念分布，纯动作模仿在部分可观测下易迷失 |
| **B3 (Full Grid Attn)** | 注意力机制类型 | 用普通全网格注意力替代 **MiTA 路口地标路由注意力** | 开启辅助信念头 + 屏蔽 + MCTS | 验证 MiTA 地标压缩与 Top-$k$ 路由在降低推理耗时（留给 MCTS 更多模拟次数）及提升路口聚焦上的优势 |
| **B4 (Reactive Only)** | 在线树搜索 | `Belief-MiTA` + 死胡同屏蔽，但**关闭在线 MCTS（纯单步贪心输出）** | 相同网络权重 | 证明在线轻量协同 MCTS 对处理短视距战术博弈的必要性 |
| **B5 (Full Belief-MiTA)** | 完整系统 | 路口地标 `Belief-MiTA` + 死胡同安全屏蔽 + 先验裁剪协同 MCTS | $50\,\text{ms}$/步搜索预算 | 达到最高胜率与最低被抓死亡率 |

### Block 4（B4）：注意力地标热力图与死胡同逃生定性案例分析（Interpretability & Case Study）

* **检验主张**：直观展示新颖注意力机制的可解释性。
* **可视化产出**：
  1. **路口地标注意力与信念热力图（Figure 3）**：展示当敌方吃掉一颗远处豆子或进入带噪距离交集时，`Belief-MiTA` 的 Top-$k$ 路由如何瞬间点亮敌方必经的 2 个咽喉路口地标，引导两只己方 Ghost 分走上下两条走廊完成合围。
  2. **死胡同屏蔽前后对比轨迹（Figure 4）**：展示进攻 Pac-Man 在深走廊吃豆时，安全屏蔽器如何根据敌方信念概率提前 3 步阻断继续深入的贪心动作并安全返航。

## 五、 执行里程碑與本地资源预算（Phase 4: Run Order & Milestones）

全程在**本地 RTX 3050 Ti Laptop + CPU** 上执行，**0 元租卡费**，总计计算时间约 **6–8 小时**：

| 里程碑 | 阶段目标 | 具体执行动作（Runs） | 预注册决策门控（GO / STOP Gate） | 预计本地耗时 |
| :--- | :--- | :--- | :--- | :--- |
| **M0：环境搭建与小样本门控（Sanity & Gate）** | 跑通 CS188 CTF 环境、死胡同预计算与 200 局专家轨迹采集，在 `defaultCapture` 上做快速闭环测试 | 采集 200 局 Oracle 对局，在 3050 Ti 上训练 20 Epoch `Belief-MiTA`，测试单步推理耗时及对阵 `Baseline-Heuristic` 的 20 局胜率 | **GO**：单步网络推理耗时 $\le 8\,\text{ms}$（为 $50\,\text{ms}$ MCTS 留出 $>40\,\text{ms}$ 裕度），且对基础基线胜率 $\ge 75\%$；**STOP**：若推理超过 $15\,\text{ms}$，缩减通道维度 $d$ 从 64 至 48 | 2 小时 |
| **M1：全量专家采集与多任务网络训练（Training）** | 在多样化地图上采集 1,000 局专家数据并完成主网络与消融变体训练 | 运行多进程 CPU 采集 1,000 局轨迹，在 3050 Ti 上训练 `Belief-MiTA`、`Conv-LSTM`、`Full-Grid-Attn` 及无信念头变体 | 验证集敌方位置预测 Hit@3 $\ge 65\%$，动作 Top-3 准确率 $\ge 80\%$ | 2.5 小时 |
| **M2：多对手锦标赛与信念精度评测（Tournament）** | 执行 Block 1（5 类对手 × 标准/随机图）与 Block 2（信念追踪误差对比） | 多进程并发运行锦标赛对局，统计胜率、净胜分与双侧检验 $p$ 值 | 确认对 `PF-AlphaBeta`、`Hybrid-MCTS` 和 `Champion-FSM` 胜率均 $\ge 60\%$（$p < 0.01$） | 2 小时 |
| **M3：单变量消融与热力图绘制（Ablation & Plots）** | 执行 Block 3（B0–B5）与 Block 4 可视化渲染 | 运行消融对局并导出典型帧的注意力权重矩阵绘制热力图 | 完成全部正文图表生成 | 1.5 小时 |

## 六、 实验追踪表（Phase 5: EXPERIMENT_TRACKER）

| Run ID | 里程碑 | 实验块 | 目的 | 对比模型 / 变体 | 地图与样本量 | 核心指标 | 状态 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PM-001** | M0 | Gate | 死胡同屏蔽与推理延迟门控 | `Belief-MiTA` 原型 vs `Baseline` | `defaultCapture` (20 games) | ms/step, Win Rate % | **TODO** | 本地 3050 Ti 快速验证 |
| **PM-002** | M1 | 数据/训练 | 1,000 局全知专家采集与多任务训练 | `Belief-MiTA` 及消融网络 | 10 Seen Layouts (250k steps) | Val Action Acc, Belief NLL | **TODO** | 本地 3050 Ti 训练约 1.5h |
| **PM-003** | M2 | B1 主表 | 已知地图锦标赛（對阵 5 大基线） | `Belief-MiTA` vs 5 Baselines | 4 Standard Layouts (100 games each) | Win Rate %, Score Margin | **TODO** | 对应论文 Table 1 上半部 |
| **PM-004** | M2 | B1 泛化 | 未见随机地图零样本泛化锦标赛 | `Belief-MiTA` vs 5 Baselines | 20 Unseen `RANDOM` Layouts | Win Rate %, Score Margin | **TODO** | 对应论文 Table 1 下半部 |
| **PM-005** | M2 | B2 信念 | 敌方隐位置追踪精度专项对比 | `Ind-PF, Shared-PF, ConvLSTM, Belief-MiTA` | 100 局盲区帧序列 | NLL, Hit@3 %, L1 Error | **TODO** | 对应论文 Table 2 |
| **PM-006** | M3 | B3 消融 | 单变量隔离消融（B0–B5） | 6 组受控变体 | 对阵 `Champion-FSM` & `Hybrid-MCTS` | Win Rate %, 被抓死亡次数 | **TODO** | 对应论文 Table 3 |
| **PM-007** | M3 | B4 可视化 | 地标路由注意力与死胡同逃生渲染 | `Belief-MiTA` 完整模型 | 典型合围帧与死胡同帧 | 注意力权重热力图 | **TODO** | 对应论文 Figure 3 & 4 |

## 七、 方向 B 核心必读文献与代码库清单（Recommended Reading List）

1. **[UC Berkeley CS188 Pac-Man Capture the Flag 官方框架](https://berkeleyai.github.io/cs188-website/projects.html)** 与三大开源强基线代码库：
   * **[COMP90054 Top-8 Tournament Agent (infinityglow)](https://github.com/infinityglow/COMP90054-Pacman-Contest-Project)**：必读其死胡同检测（Dead-End Detection）与攻防状态切换实现。
   * **[Particle Filter + Alpha-Beta Agent (lzheng1026)](https://github.com/lzheng1026/Pacman-Capture-The-Flag)**：必读其粒子滤波器更新与极小化极大博弈搜索代码。
   * **[Hybrid MCTS Pac-Man Agent (neeilnandal)](https://github.com/neeilnandal/pacman-mcts-agent-v2)** 与 **[Multi-Agent MCTS (RuilinM)](https://github.com/RuilinM/MCTS-in-Pacman-Capture-The-Flag-environment)**：必读其双智能体 MCTS 节点扩展接口。
2. **[MiTA Attention: Efficient Fast-Weight Scaling via a Mixture of Top-k (NeurIPS 2026)](https://arxiv.org/abs/2602.01219)**（[官方代码 QishuaiWen/MiTA](https://github.com/QishuaiWen/MiTA)）：**核心注意力机制来源**，重点参考其如何将长序列 Token 压缩并路由至少量地标查询，用于构建本文的路口地标信念聚合器。
3. **[Safe Reinforcement Learning via Shielding (Alshiekh et al., AAAI)](https://ojs.aaai.org/index.php/AAAI/article/view/11797/11656)** 与 **[Probabilistic Shielding for Safe RL (2025, arXiv:2503.07671)](https://arxiv.org/html/2503.07671v1)**：**安全屏蔽理论权威出处**（两篇论文均以 Pac-Man 作为核心验证场景！），为你的死胡同拓扑安全屏蔽器提供坚实的理论形式化定义。
4. **[Human-level performance in 3D multiplayer games with population-based reinforcement learning (DeepMind FTW, Science 2019)](https://deepmind.google/blog/capture-the-flag-the-emergence-of-complex-cooperative-agents/)** 与 **[Disentangling Successor Features for Coordination in MARL (DISSC, arXiv:2202.07741)](https://arxiv.org/html/2202.07741)**：**CTF 博弈论文写作标杆**，用于引言与相关工作中论证夺旗博弈的学术价值与技术挑战。
5. **[Reflections on Pacman AI Competition (Matthias Bachfischer, 2021)](https://bachfischer.me/posts/2021/02/reflections_on_pacman_ai_competition/)** 与 **[Monte Carlo Tree Search for Pacman CTF (Jaden Stock et al.)](https://jadenstock.files.wordpress.com/2017/03/573_ai_final_report.pdf)**：**技术细节复盘文献**，详细分析了共享粒子滤波与在线规划在 CTF 中的优缺点。
