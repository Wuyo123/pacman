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

所有路径均为本地结果；没有上传或租用计算资源。原M0的20局只决定是否推进M1，不支持论文级显著性或跨图泛化主张。
