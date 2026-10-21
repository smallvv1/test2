# NDP-LRD：局部关系蒸馏候选设计

## 目的与边界

NDP-LRD 是独立研究性候选，不接入正式训练 processor。本设计保持
`L_base = L_CE + L_Triplet + L_Bio + 0.1 L_GPC` 不变，不改变 GPC selector、
matched-budget、lambda、EVA02 或数据划分。参考状态必须在训练开始前冻结，不能使用
update400 后验激活标签选择关系。

## 数学定义

对 anchor `i` 的固定合法邻居集合 `P_i`（同 PID）和 `N_i`（异 PID），将当前特征
归一化后定义：

`l_cur(j) = cosine(z_i,z_j)/tau`，`q_cur = softmax(mask(l_cur))`；

参考分支同理得到 `q_ref`，但参考特征完全 `detach`。正身份概率质量为
`Q_+(q) = sum_{j in P_i} q(j)`，参考优势为

`A_ref = logmeanexp(l_ref(P_i)) - logmeanexp(l_ref(N_i))`。

可靠 mask：

`R_i = [A_ref > rho] AND [Q_+(q_ref) >= q_min] AND external_mask_i`。

在线激活门控：

`G_i = stopgrad(R_i AND [Q_+(q_ref)-Q_+(q_cur) > epsilon_Q])`。

候选损失：

`L_LRD = mean_{i:G_i=1} KL(q_ref(i) || q_cur(i))`。

无激活关系时返回与当前 anchor 计算图相连的零；参考分支不接收梯度。缓存邻居
版本中梯度只经当前 anchor（以及任何未 detach 的当前邻居）传播；实时重编码版本
可让当前邻居共同获得梯度，但计算成本更高。

`epsilon_Q` 的单位是概率质量，不得把 cosine 域 `0.02` 直接当作该阈值。所有
`tau/rho/q_min/epsilon_Q` 必须在训练前冻结，并由 train-only 诊断协议确定。

## 为什么不是自适应 Margin Triplet

单正单负时，局部关系只有二类 softmax，LRD 退化为一个平滑二类关系约束，不能宣称
独立机制。多邻居时，LRD 同时约束正身份总质量和正/负邻居内部概率形状，梯度可分配
给全部合法邻居，而 Batch-Hard/自适应 Triplet 主要由选中的 hardest pair 决定。

严格反例已写入单测：两组当前多邻居 logits

- `P=[3,2], N=[2,1]`；
- `P=[4,1], N=[3,0]`；

两者的 `logmeanexp(P)-logmeanexp(N)` 均为约 `1`，因此只看自适应 margin 的标量
无法区分；但相对于同一参考分布的 LRD KL 梯度不同。该差异来自多邻居概率结构，
不是把 margin 阈值改名。

## 与 VLADR 的关系

VLADR 风格全局关系蒸馏是本候选的思想来源，但 NDP-LRD 只对固定身份语义的局部
正/负邻域工作，并引入可靠性与单侧身份质量下降门控；不强制保持参考模型所有关系，
避免把参考模型中的错误排序也蒸馏下来。它也不等价于全局含对角线关系 KL。

## 预注册控制组

- C0：固定 Bio+GPC-v1。
- C1：相同邻居集合和预算的普通局部 Triplet。
- C2：参考自适应 Margin Triplet。
- C3：相同邻居集合的全局/局部 KL 关系蒸馏，不使用单侧质量门控。
- C4：可靠性与单侧质量门控的 NDP-LRD。

所有控制需匹配新增邻居集合、重编码次数、有效更新数，并保留 A/A 噪声对照。

