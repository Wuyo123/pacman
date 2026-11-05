# 输入修复对照续跑协议

用户于2026-11-05明确授权“继续跑完这轮”。这是原输入对照的预算延长，独立记账，不改写原90分钟批次的`INCONCLUSIVE_RESOURCE`结论，也不作为第二次独立实验。

## 冻结范围

- 原数据：`runs/input_repair_20261105/data`，200局、180/20按游戏划分、120000时刻。核对文件hash，不重采、不改标签。
- V0：直接使用已完成20轮的`v0/training/model.pt`，不重训。
- V1：使用原`v1/training/model.pt`中15轮完成、第16轮部分更新的权重和Adam状态，精确续至总20轮，不重复已完成更新。
- seed20261105、d64、batch64、Adam 3e-4、损失、37通道定义、模型、shield与搜索算法保持不变。
- 原输出只读；续跑输出至`runs/input_repair_continuation_20261105`。正式续跑自启动起最多45分钟，含训练、评测和重放核验。再遇资源截止仍保存未决结果，不自动再延时。

## 恢复正确性

原checkpoint缺少DataLoader generator状态，但保存了已完成epoch及当前epoch已更新batch数。训练模型没有随机dropout，shuffle使用独立显式generator。以同长度dummy数据集、同构DataLoader和相同seed，完整迭代已完成各轮，恢复每轮base seed和shuffle消耗；真实当前轮跳过已完成batch，仅训练其余部分。

若当前轮全部batch已经更新而验证中断，仅重跑验证，不再更新该轮。恢复前核对checkpoint进度、data hash、配置与Adam step；恢复后各参数step应等于`20 * ceil(train_samples / 64)`。CPU回归须验证中途保存/恢复与不中断训练的最终模型、Adam状态和batch次序一致；不声称CUDA逐bit确定性。

第16轮历史训练损失累加器未保存，其续跑日志仅表示剩余样本，或标记validation-only；不得伪造整轮平均损失。最终20轮验证和正式对局才用于效果比较。

## 评测与收口

两臂各20局，共40计划记录，使用原定9700000–9700009红蓝交换的相同配对种子；此前这40局均未运行。CPU推理、周期GC移至局间、完整动作50ms和batch1前向p95<=8ms门槛不变。无崩溃、非法动作、超时且完整20局，才可用于有效比较。

V1得分率`(胜+0.5平)/20 >= 75%`且比同期V0至少高5个百分点，才记为`INPUT_REPAIR_IMPROVEMENT_SCREEN_MET`。两臂执行必须有效，所有有效trace独立重放通过；否则按执行无效或资源未决报告。未满足改善条件则记`NO_IMPROVEMENT_ESTABLISHED`。保留失败分母，不能用损失、局部统计或历史结果替代胜率。

预留920确认集不动，不扩展种子、不挑checkpoint、不调参数。此次只有一个训练seed和20局/臂的小样本screen，不支持显著性、跨地图泛化或注意力模块独立贡献的主张。教师信息与顺序协同标签未在本轮改变。

结果保存原/续跑预算、来源及checkpoint哈希、有效更新数、各臂原始40记录与配对差、时延、搜索计数和重放核验，再封存Git。
