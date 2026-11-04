# 核验发现

- 原计划：C:/Users/25497/Desktop/G TO C交接/b_belief_mita_ctf_20261104.md，2026-11-04。
- MiTA 原论文及官方仓库真实，作者 Qishuai Wen 等；arXiv:2602.01219，当前 v6。本文提出的拓扑查询是待验证的领域改造，不能由原视觉实验直接证明收益。
- Liu 与 Stock，Monte Carlo Tree Search for Pacman Capture the Flag，2017，报告 §2.1 已用双队友共享两个敌方 PF；不是所有 PF 都只保留均值。https://jadenstock.files.wordpress.com/2017/03/573_ai_final_report.pdf
- ConstrainedZero (Moss et al., IJCAI 2024) 已做信念空间的策略/价值/失败概率学习和在线 MCTS；需要对照区分新增价值。https://www.ijcai.org/proceedings/2024/0746.pdf
- Risk-aware shielding of Partially Observable Monte Carlo Planning policies (Mazzi et al., AIJ 2023) 已做 belief 风险规则与 POMCP shielding。https://doi.org/10.1016/j.artint.2023.103987
- 当前屏蔽使用逐格概率阈值而非累计风险，漏掉低单格概率但总质量大的危险区域；先验置零也不等于删除搜索合法动作。
- 原计划 C1 的 65% 与 M2 的 60% 不一致；平局记半胜后不能直接套普通二项检验；比赛/时间步有相关性。
- 原计划同时引入 oracle、网络、shield、MCTS、QMIX，多处消融不隔离；6–8 小时没有吞吐测量支撑。
- 源码核验：game.py 按 agentIndex 轮流行动；capture.py 当前行动者产生 sonar，任一队友 5 格内即可显露敌人，因此共享历史必须保留时间戳。
- 独立无平局、n=100、p0=0.5 的双侧精确检验：60 胜 p=0.05688793364，65 胜 p=0.00351764172；实际有配对/平局时应另定检验。
- 若按每图每对手100局，24×5×100=12000局；若己方每局250次决策均用满50ms，仅己方串行决策时间为41.7小时，不能当成并行墙钟下界。
- neeilnandal README 仅支持 MCTS-inspired 进攻/返航与反应式防守，不支持原计划直接标为共享 PF 协同 UCT；infinityglow README 最高排名自述为开发期间8/88，非冠军。
- 全文评审、最近工作比较、修订后诊断顺序和执行决定已写入 PLAN_REVIEW_20261104.md。

## 用户要求原计划优先后的执行证据

- 2026-11-04正式M0采集：200/200局、120000己方时刻、119胜81平0负、约488.9秒；这是全知专家vs官方baseline，不能用作部分可观测学习模型的结果。
- 修复过专家在终局前向负时间搜索的实现错误，失败测试→通过；首批数据只作为工程记录保留，未进入正式训练。
- 29项测试通过后启动d64、20epoch GPU训练和20局独立评测的连续批次，源文件哈希在runs/m0_original/manifest.json。
- 最后代码审查另记两个边界：benchmark初始化不检查deadline；当网络>15ms且实战动作超时，gate先返回INVALID_EXECUTION，未自动触发原定d48。当前批次开始时仍有约92分钟且已有CPU约3ms原型诊断，先保留冻结运行不热改代码；如实际触发这些分支，保留原始结果并按原计划补做d48/资源记录，不把无效执行当科学负结果。
