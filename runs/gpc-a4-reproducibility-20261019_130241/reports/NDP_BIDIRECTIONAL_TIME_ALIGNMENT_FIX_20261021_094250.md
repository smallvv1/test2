# NDP 双侧边界时间对齐修复

## 状态

`PASS_WITH_LIMITED_COVERAGE`。本报告取代旧的 28 行双侧结果；旧报告保留为历史记录，语义标记为 `SUPERSEDED`。

## 修复内容

旧实现通过下一条 transition 的 `target_margin` 计算变化，导致当前区间被错配。新实现对同一行直接使用：

```text
M_prev   = row.prev_margin
M_target = row.target_margin
Delta_M  = M_target - M_prev
```

同时只保留合法的单负、互反图像关系；A→B 与 B→A 保留为独立有向关系，不合并梯度或翻转标签。

## 实际结果

| Seed | 有向图像对 | 修复后行数 | A改善/B恶化 | A恶化/B改善 | 双改善 | 双恶化 | A侧翻转 | B侧翻转 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1234|7|35|2|5|17|11|2|1|
|4321|7|35|8|7|10|10|1|0|

每个 seed 均覆盖 `0→20、20→50、50→100、100→200、200→400` 五个区间。旧结果每 seed 只有 28 行，因此不是完整区间统计。

## 证据与命令

执行：

```text
/usr/local/bin/python3.10 -u scripts/ndp_bidirectional_boundary_audit.py \
  --rows .../future_boundary_fixed/future_boundary_seed1234.csv .../seed4321.csv \
  --groups .../fixed_diagnostic_groups.json .../fixed_diagnostic_groups.json \
  --output-dir outputs/ndp_future_boundary_20261021_094250/bidirectional_fixed
```

结果：

`outputs/ndp_future_boundary_20261021_094250/bidirectional_fixed/bidirectional_boundary_summary.json`

源码 SHA256：`ndp_bidirectional_boundary_audit.py` = `bfb298939a5584db0a7a27b3980f0398c60b57a67b3a1f18259cdc97e9f18056`。

回归测试 `tests/test_ndp_bidirectional_boundary.py` 验证 0→20 不会读取 20→50 的目标值，并覆盖全部五个区间；本轮总测试 8/8 通过。

## 限制

每个 seed 只有 7 个有向图像对、6 个 identity pair，不能作为全训练集双侧边界普遍性证据。该修复只证明统计口径正确，不证明 NDP 训练收益。
