# Belief-MiTA CTF 原计划执行

用户要求先执行[原计划](ORIGINAL_PLAN_20261104.md)，未通过后再考虑调整。[评审意见](PLAN_REVIEW_20261104.md)保留，未替代本轮方法。

当前阶段与真实结果见 `RUN_LEDGER.json`；实现细节、种子、超参数与门槛见 [执行协议](EXECUTION_PROTOCOL.md)。

## 环境

- 工作目录：`C:/D/PacMan`，WSL路径 `/mnt/c/D/PacMan`。
- 已验证解释器：Ubuntu `/opt/lagat-h-20261022/venv/bin/python`，Python3.11、PyTorch2.8.0+cu126；共享环境未改动。
- GPU：RTX3050Ti Laptop 4GB；训练CUDA，部署设备由真实前向延迟选择。
- 引擎：`vendor/capture`，infinityglow/COMP90054-Pacman-Contest-Project，固定commit083e06c77425d6de87850efda55e4a5e1075da9a。

## 验证

在WSL、项目目录运行：

```sh
/opt/lagat-h-20261022/venv/bin/python -m unittest discover -s tests -v
```

## 数据与结果

- `runs/m0_preterminalfix/`：首个专家终局截断实现错误的中断批次，仅作工程记录，未用于训练。
- `runs/m0_original/data/`：200局正式oracle数据，100种子各交换红蓝两局，逐局真值与可见输入分开。
- `runs/m0_original/collection_summary.json`：完整采集分母与结果。
- `runs/m0_original/manifest.json`：训练/评测开始时的源码及原计划哈希、引擎版本、设备。
- `runs/m0_original/training/`：逐epoch记录、最后一轮模型与训练/验证来源。
- `runs/m0_original/evaluation/`：真实帧延迟、20局完整记录、原门控判定。

代码与已封存的模型、数据和结果已保存到公开仓库 [Wuyo123/pacman](https://github.com/Wuyo123/pacman)。计算在本机进行，未租用远程计算资源。原M0的20局只决定是否推进M1，不支持论文级显著性或跨图泛化主张。

## 2026-11-05 修复续行

原M0在修复GC延迟后得分70%，未达到75%门槛；后续三轮对照也未建立改进，见[改进结果](IMPROVEMENT_RESULT_20261105.md)。M1–M3未运行。

[问题定位](ROOT_CAUSE_ANALYSIS_20261105.md)发现同格携豆覆盖和公开状态缺失。现已新增版本化37通道输入，保持旧18通道/模型兼容，并加入实际搜索访问深度和观测复用统计；66项测试及CUDA端到端试跑通过。

[输入修复对照](INPUT_REPAIR_PROTOCOL_20261105.md)由Luna执行，同一批200局数据、两臂各20epoch和20局同期评测，90分钟硬预算。批次已因资源截止结束：V0完成20轮，V1完成15轮后在第16轮中断，未开始评测，效果未决；见[完整结果](INPUT_REPAIR_RESULT_20261105.md)。全知教师与顺序协同标签问题保留为下一阶段，不混入本轮。

用户随后授权[续跑](INPUT_REPAIR_CONTINUATION_PROTOCOL_20261105.md)，现已完成：两臂均20轮、33760次更新，40局评测/重放全部通过；原输入V0得分65%，修复V1为45%，未建立改进。原90分钟资源中断记录保持不变；[续跑结果与搜索深度证据](INPUT_REPAIR_CONTINUATION_RESULT_20261105.md)已单独封存，920确认集和M1–M3未运行。
