# 动态高反候选提取设计细则

## 1. 设计原则

本任务不应该被做成庞大、冗余、状态复杂的系统。

但“系统简单”不等于“不讨论细节”。我们需要把每个关键细节提前说清楚，避免实现时反复歧义、反复试错。本文档的作用是明确边界和取舍，而不是提前堆砌模块。

第一版追求：

```text
简单
有效
高效
可观测
可调试
资源可控
```

第一版避免：

```text
常驻多层地图
复杂状态机
全局 micro-component 两两 grouping
无限 raw cache
把所有未来可能性一次性实现
```

核心准则：

```text
先用最小机制覆盖当前真实任务
每个新增机制必须解决已经观察到的问题
设计细节写清楚，但实现按阶段收敛
```

## 2. 当前任务事实与假设

当前阶段的任务条件：

```text
地图已经是高反射点地图
场地内没有长期稳定且强混淆的高反干扰物
反光板静止
板子约 40 x 40 cm
高反贴集中在中间约 20 x 25 cm 区域
高反贴通常 3-5 条
反光条宽度可能只有约 1.5 cm
LiDAR 10 Hz
车辆运动不剧烈
在线识别优先，离线全图重识别不是第一版目标
任意两块需要同时识别的真实板，在 map-space high-ref support 上应有足够分离距离
```

这些条件说明：我们需要的是工程上稳定的高反区域提取，不是通用三维目标检测系统。

第一版边界：

```text
只保证分离良好的单板 proposal
近距离双板、ROI 明显重叠、两个板被同一个 ROI 同时收进来的场景，属于 unsupported / degraded case
遇到该类情况必须打日志和 debug marker，不应静默当成普通负样本
```

## 3. 总体方案

第一版采用：

```text
immutable map snapshot
-> active fine voxel refs
-> temporary sort/group by mid key
-> explicit discrete bridge rule
-> seed regions
-> ROI completion through mid-cell group lookup
-> ROI-level dedup
-> candidate-record association
-> cooldown / BPU admission budget
-> loose evidence / geometry checks
-> raw-window lookup using consistent map pose
-> fixed-K old-contract BPU evaluation
-> score aggregation
-> verified-board-level dedup
-> output
```

其中：

```text
fine voxel:
  保存高反空间证据

mid cell:
  临时搜索结构，负责降复杂度和有限桥接

proposal seed:
  表示“这里附近值得展开”

ROI completion:
  把 seed 周围可能属于同一块板的高反证据收集完整

BPU:
  最终判别，不承担地图搜索
```

重要语义：

```text
mid-cell connected region != 完整反光板
mid-cell connected region = proposal seed
ROI query != enumerate fine voxel grid
candidate center != physical board center
```

## 4. 常驻数据结构

### 4.1 Fine High-Reflectivity Evidence Map

这是第一版唯一常驻空间地图。

每个 active fine voxel 保存紧凑证据：

```text
voxel key
representative position
max intensity
evidence count
last_seen tick/stamp
support sum，可选
```

`representative position` 第一版定义为：

```text
voxel center in map frame
```

不要使用“最新命中点”作为候选几何的代表点。这样 seed、ROI、center、PCA 和 footprint 不会因为同一个 voxel 内部的最新点抖动而变化。

不保存：

```text
所有历史 raw points
多帧完整点云
复杂 per-voxel raw cache
```

当前 fine voxel baseline 可以继续用 `0.003 m`，但它不是最终锁死参数。

需要测试：

```text
同一反光贴在 map 坐标下的累计散布
不同 voxel_size 下的地图点数量
不同 voxel_size 下的候选召回
不同 voxel_size 下的 CPU / 内存
```

如果保留 `0.003 m`，active 判断不能要求单个 3 mm voxel 跨多帧稳定命中。因为定位误差和扫描角度会让同一物理高反区域散到多个 fine voxel 中。

因此 mid-cell 聚合时主要看：

```text
局部总 evidence
活跃 fine voxel 数
support sum，可选
```

`support sum` 只表示累计支持量，不表示 distinct frame count。第一版不把“不同帧支持数”作为 active 条件，因为多个 fine voxel 的 frame support 简单相加会重复计算同一帧。

### 4.1.1 空间量化合同

fine voxel 和 mid cell 不要求严格父子层级。

原因：

```text
fine_voxel_size_m = 0.003
mid_cell_size_m = 0.05 或 0.08
```

二者通常不是整数倍关系。因此不得把 mid key 写成 fine key 的 parent。

第一版统一采用：

```text
fine_key = floor((p_map - grid_origin_map) / fine_voxel_size_m)
mid_key  = floor((p_map - grid_origin_map) / mid_cell_size_m)
```

其中：

```text
p_map:
  点在当前 map frame / map_epoch 下的坐标

grid_origin_map:
  全系统统一的量化原点
```

所有空间量化都必须使用数学 floor。

禁止：

```text
对负坐标使用 C++ 整数除法截断来代替 floor
```

否则地图跨过原点时，负坐标 voxel 会归属错误。

### 4.1.2 Evidence 生命周期合同

evidence 的更新、饱和和淘汰策略由 mapper 定义，candidate extractor 只读取，不自行改变 evidence 生命周期。

当前 mapper 侧合同：

```text
同一 scan/frame 对同一 fine voxel 的 evidence 增量最多一次
evidence count 有饱和上限
last_seen 表示最近一次实际写入 evidence 的时间或 tick
TTL 开启时由 mapper 淘汰过期 voxel
TTL=-1 时不按时间淘汰，但仍受 map capacity 约束
```

candidate 的 `last_seen` 必须来自 ROI 内 fine voxel 的实际 `last_seen` / evidence version，不能因为 full snapshot 再次遍历到该区域就刷新。否则静态旧候选会永远“新鲜”，TTL 和 cooldown 都会失效。

但 BPU admission 不能只依赖 `max last_seen`。

原因：

```text
静态板每帧继续被扫到
同一批已饱和 voxel 的 last_seen 仍会更新
空间 support 没有变化
候选却会被 cooldown 后反复送 BPU
```

因此 candidate record 需要使用：

```text
support_revision
```

它只在以下情况变化时递增：

```text
ROI 内出现新的 active fine voxel
ROI 内有效 support set 发生变化
累计 capped evidence 跨过预定义有效增量
ROI 的空间 support 明显扩展
```

同一个已饱和 voxel 再次被看到，不应单独触发 `support_revision`。

### 4.2 Bounded Raw Observation Ring

raw ring 只用于在线 BPU 验证，不能变成第二张地图。

必须有限：

```text
raw_ring_horizon_sec
raw_ring_max_frames
raw_ring_max_points_per_frame
```

保存字段尽量少：

```text
xyz
intensity
stamp / frame id
map_epoch
frame map-space AABB
frame occupied mid-cell keys，可选
T_map_lidar_used_for_mapping，如果点保存在 LiDAR/sensor frame
deskewed/map-space 标记
```

raw ring 坐标合同必须明确：

```text
如果 raw ring 保存 map-space deskewed points:
  xyz 必须已经在当前 map frame / map epoch 下

如果 raw ring 保存 LiDAR/sensor-frame points:
  必须同时保存 T_map_lidar_used_for_mapping
  ROI 查询前必须使用该 transform 投到 map frame
```

第一版优先保存已经去畸变并投到 `reflective_odom` 的 high-ref points，这样 map ROI 与 raw ring 查询天然在同一坐标系下。若后续为了恢复更接近旧训练输入而保存 sensor-frame window，也必须保存对应 pose/transform。

raw ring 的作用：

```text
当 map proposal 刚出现时，仍能取到近期原始观测窗口
按旧 BPU 输入契约构造识别输入
```

raw ring 不保证能验证历史地图中所有板子。

raw ring horizon 必须覆盖：

```text
模型所需窗口长度
+ proposal 生成延迟
+ 调度延迟
```

否则 proposal 出来时，构造旧 BPU 输入所需的最早窗口可能已经被淘汰。

raw ring 查询不能简单对每个 proposal 扫描所有 ring points。第一版至少使用 frame-level 粗筛：

```text
ROI
-> 先用 frame map-space AABB / occupied mid-cell keys 筛可能相关的 raw frames
-> 只对候选 raw frames 做点级 ROI 检查和 BPU 输入构造
```

这不是把 raw ring 做成第二张地图，只是给窗口查找设置明确上界。

raw ring 的生命周期也必须稳定。

如果 BPU job 已经选择了某个 raw frame，则 job 必须持有：

```text
frame 的稳定副本或 shared_ptr
或带 generation 校验的 pinned ring slot
```

禁止只保存 ring 下标后异步使用。否则 ring 滚动覆盖后，BPU 构造输入时可能读到错误 frame。

`raw_ring_max_points_per_candidate` 不属于 ring 插入阶段的随意截断参数。候选级点数上限应放在 BPU 输入构造阶段，并复用旧 pipeline 的确定性采样、padding、mask 和 meta 规则。

### 4.3 Candidate Record

candidate record 只做轻量记录，不做复杂状态机。

建议字段：

```text
candidate id
proposal_center_map
support_bbox_map
roi_map
support_revision
map_epoch
last_seen stamp
last_eval stamp
score
verified flag
unverified reason
last_evaluated_support_revision
last_evaluated_ring_revision
retry_count
```

第一版状态只需要：

```text
unverified
verified
stale
suppressed，可选
```

其中 `unverified reason` 必须区分：

```text
raw_unavailable
insufficient_points
not_yet_evaluated
budget_deferred
bpu_below_threshold
ambiguous_template
deferred_overflow
```

这些原因不能混成 reject。

需要有上限：

```text
max_candidate_records
candidate_ttl_sec
bpu_cooldown_sec
```

但第一版不要引入复杂 LOCKED / LOST / READY 状态机。

admission 规则：

```text
verified:
  除非 support_revision 明显变化，否则不重复 BPU

bpu_below_threshold:
  标记 suppressed，直到 support_revision 变化才允许再评估

raw_unavailable:
  仅在出现新的可用 raw window / ring revision 时重试

budget_deferred:
  保留等待，不视为 reject
```

这些是轻量字段语义，不是复杂状态机。

support_revision 的 Phase 2 固定算法基于轻量 support signature：

```text
support_key_hash:
  hash(ROI 内 fine voxel key 集合)

support_voxel_count:
  ROI 内 fine voxel 数

support_evidence_bucket:
  floor(E_ROI / support_evidence_bucket_size)

support_bbox:
  ROI 内 support bbox
```

`support_revision` 增加，当且仅当：

```text
1. support_key_hash 改变，
   且 support_voxel_count 或 support_bbox 变化超过容忍范围；
或
2. support_evidence_bucket 相比上次 BPU 评估时提升；
或
3. support_bbox 任一边界扩展超过 support_expand_margin。
```

candidate record association 规则：

```text
仅在同一 map_epoch 中关联。

对每个 proposal:
  在已有 record 中找 ROI IoU 最大者；
  若 IoU >= candidate_association_iou_threshold，则关联到该 record；
  否则创建新 record。

若多个 record 同时满足:
  选 support overlap 最大者；
  再以中心距离作 tie-break。
```

record 过期规则：

```text
若连续 candidate_ttl_sec 内无关联 proposal:
  state = stale
  stale record 优先被容量淘汰
```

`max_candidate_records` 满时，淘汰顺序：

```text
stale
-> unverified low-evidence
-> suppressed
-> oldest verified
```

禁止随机淘汰或只按插入顺序淘汰。

## 5. Snapshot 处理策略

### 5.1 第一版使用 Full Snapshot

第一版允许每次处理 full map snapshot。

原因：

```text
当前地图规模小
便于验证 proposal recall
便于调参
不用提前改 mapper-extractor 增量协议
```

但需要明确：full snapshot 是第一版验证路径，不一定是最终在线路径。

这里的 snapshot 不等于复制整张 `unordered_map`，也不等于 extractor 直接遍历 mapper 的 live hash map。

第一版并发合同：

```text
mapper 在受保护读窗口内导出紧凑 FineVoxelRef[]
snapshot = FineVoxelRef[] + snapshot_id + map_epoch
extractor 只持有 FineVoxelRef[]，不保存 mapper 内部 voxel 的指针或引用
mapper 解锁后可以继续更新自己的 hash map
```

禁止：

```text
每次复制完整 fine hash map
extractor 持有 live hash map iterator / pointer / reference
```

原因：

```text
复制完整 map 会制造接近整张地图大小的额外内存峰值
live hash map 在插入、rehash、TTL 淘汰时会让 iterator / pointer / reference 失效
```

### 5.1.1 Snapshot 调度合同

第一版不使用复杂任务队列，但必须禁止 snapshot 积压。

固定规则：

```text
extract_snapshot_period_sec:
  extractor 实际处理 snapshot 的最小间隔

max_pending_snapshots = 1
```

若 extractor 正在处理上一份 snapshot：

```text
新 snapshot 不排队
只保留最新 snapshot，覆盖旧 pending snapshot
记录 coalesced_snapshot_count
```

因此运行时上界是：

```text
最多一个正在运行
最多一个待处理
```

snapshot 超过容量时：

```text
fine_ref_count > max_snapshot_fine_refs
```

必须：

```text
记录 snapshot_overflow
不静默截断
本次进入 degraded / deferred 路径
```

第一版单线程 ROS 回调实现时，可以用：

```text
subscriber queue size = 1
extract_snapshot_period_sec 节流
```

来近似满足“不积压”。后续若改成异步 worker，必须显式实现上述 pending 覆盖语义。

需要记录：

```text
active fine voxel count
extract runtime p50 / p95
scratch memory peak
proposals per snapshot
BPU calls per snapshot
```

当这些指标接近资源预算，再引入 dirty-region。

### 5.2 Map Epoch 合同

`map_epoch` 表示当前 evidence map、raw ring map-space 索引、candidate record 所属的同一坐标时代。

发生以下任一事件时，必须切换 epoch：

```text
LIO / 外部里程计重置
map frame 重定位
回环或全局校正
reflective_odom 原点重置
mapper 清图并重新建图
```

第一版 epoch 改变后的简单规则：

```text
不允许跨 epoch 查询 raw ring
清空或失效旧 candidate record
清空 raw ring，或标记旧 ring frame 不可用于当前 map
新 epoch 重新积累候选
```

若第一版系统保证 `reflective_odom` 永远连续、不重定位、不重置，也必须把它作为运行时不变量记录到日志里。

### 5.3 Dirty-Region 是后续触发项

dirty-region 不是第一版必做。

后续触发条件：

```text
extract p95 超过 snapshot 周期预算
active fine voxel 数持续增长到 full scan 明显吃力
BPU calls per snapshot 超预算
scratch memory 接近内存目标
```

目标 dirty-region 流程：

```text
本周期更新的 fine voxels
-> dirty mid-cell keys
-> 加一圈邻域
-> 局部临时 mid 聚合
-> 局部 seed / ROI / proposal 更新
```

它只维护小的 dirty key set，不常驻完整 mid map。

## 6. Mid-Cell 聚合

### 6.1 目的

mid cell 不负责识别板。

它只负责：

```text
降低搜索复杂度
把相近高反证据聚成局部 seed
提供有限桥接能力
```

### 6.2 尺度

建议第一版：

```text
mid_cell_size_m = 0.05 - 0.10
```

这个尺度来自：

```text
板上反光贴间距
反光区域整体 20 x 25 cm
希望有限 d_bridge_cells 能桥接小断裂
不希望大范围串联
```

mid cell size 是关键参数，优先测试 `0.05 m` 和 `0.08 m`。

### 6.3 临时实现方式

优先：

```text
遍历 active fine voxels
-> 生成 FineVoxelRef
-> 使用统一 grid_origin_map 和数学 floor 计算 mid_key
-> 按 mid_key 写入 scratch vector
-> sort by mid_key
-> group / aggregate
-> 生成 MidCellEntry ranges
```

建议临时布局：

```text
FineVoxelRef {
  mid_key
  fine_key 或 fine_index
  position
  evidence
  intensity
  last_seen
}

MidCellEntry {
  mid_key
  begin/end range in FineVoxelRef[]
  evidence_sum
  active_fine_count
  bbox
}
```

暂不常驻：

```text
mid hash map
coarse hash map
octree
```

这样更容易控制内存峰值。

### 6.4 Active 判断

mid cell active 不看单个 fine voxel 是否稳定，而看局部聚合：

```text
evidence_sum = sum(min(E_v, E_cap_for_aggregation))
active fine voxel count
max intensity
support sum，可选，不代表 distinct frame count
```

Phase 1 的精确规则：

```text
fine voxel eligible for snapshot =
  mapper 未淘汰
  AND evidence_count > 0
  AND intensity >= min_intensity

mid_active =
  evidence_sum >= mid_active_evidence_threshold
  AND active_fine_count >= mid_active_min_fine_voxels
```

其中：

```text
evidence_sum = sum(min(E_v, evidence_aggregation_cap))
```

`max_intensity` 和 `support_sum` 第一版只作为日志字段，不作为 hard gate。

第一版 active 阈值要宽松，目标是保证 proposal recall。

## 7. Seed Region

第一版必须选择一个明确的离散桥接规则，不能写成“膨胀可选”。

```text
active mid cells 按 Chebyshev 网格距离连接
d_inf(cell_i, cell_j) <= d_bridge_cells
```

得到 seed region。

第一版默认：

```text
d_bridge_cells = 1
```

也就是普通 26 邻域连通，不先做一格膨胀。这样桥接能力明确，链式串联风险最低。

如果真实测试发现同一块板被过度拆分，再实验性测试：

```text
d_bridge_cells = 2
```

但每次运行只能有一个明确规则，日志中必须打印 `mid_cell_size_m` 和 `d_bridge_cells`。

### 7.1 桥接规则语义

桥接规则只解决小断裂，不负责复杂 grouping。

桥接尺度由以下离散量决定：

```text
mid_cell_size_m
d_bridge_cells
```

不要采用“先膨胀一格，再对膨胀集合做 26 邻接”的模糊流程。这个流程可能让原始 active cells 在 `d_inf <= 3` 时被连在一起，实际桥接尺度会比直觉更大。

如果桥接后出现超大链式 region：

```text
不要直接整体 reject
```

应局部回退：

```text
减小 d_bridge_cells 重新连通
提高 active evidence 阈值重新切
按局部 evidence peak 切分
```

第一版可以先简单实现：

```text
如果 region 超过 max span:
  若 d_bridge_cells > 1，则回退到 d_bridge_cells = 1 重新连通
  仍超大则标记 deferred_overflow，并打日志和 debug marker
```

`overflow region` 不得静默视为无候选，也不得计入“正常过滤掉的假候选”。

Phase 1 中必须统计：

```text
overflow_seed_count
overflow_seed_span
overflow_seed_evidence
overflow debug marker
```

在线 Phase 2 可以暂时不把 overflow region 直接送 BPU，但必须保留为可观测的未决结果：

```text
reason = deferred_overflow
```

只有真实测试证明 overflow 经常包含真实板时，再增加最小恢复机制，例如在 overflow region 内按局部 evidence peak 产生少量 seed。第一版不提前实现完整 split。

## 8. ROI Completion

seed 不等于完整候选。需要 ROI completion。

### 8.1 ROI 定义

ROI completion margin 应来自板图案物理先验：

```text
m_roi >= D_reflective_pattern_max + epsilon_map
```

其中：

```text
D_reflective_pattern_max:
  同一块板上两个独立高反区域之间的最大距离

epsilon_map:
  定位误差 + voxel 离散误差 + 累计散布余量
```

第一版可以先使用固定参数：

```text
roi_completion_margin_m
```

建议初始值围绕你的高反图案范围设定，例如覆盖 20 x 25 cm 高反区域并留余量。

### 8.2 ROI 形状

第一版固定使用 map-frame AABB ROI：

```text
roi_map = expand(seed_support_bbox_map, roi_completion_margin_m)
```

也就是：

```text
roi_min = seed_bbox_min - margin
roi_max = seed_bbox_max + margin
```

fine evidence 点属于 ROI 的判断为逐轴闭区间：

```text
roi_min.x <= p.x <= roi_max.x
roi_min.y <= p.y <= roi_max.y
roi_min.z <= p.z <= roi_max.z
```

第一版不使用 sphere / radius search。

不要过早用严格 2D 板平面，因为 seed 点可能太少，平面不稳定。

参数命名统一为：

```text
roi_completion_margin_m
```

不要继续叫 `radius`，避免实现者误以为要做球形查询。

### 8.3 ROI 收集内容

ROI 内重新收集：

```text
fine evidence voxels
近期 raw ring 中落入 ROI 的点/窗口
```

这一步是 deterministic completion，不是通用 grouping。

### 8.4 ROI 查询方式

ROI 查询禁止按 fine voxel 网格逐格枚举。

错误方式：

```text
ROI bbox
-> 枚举其中所有 0.003 m fine voxel keys
-> 对每个 key 查 hash map
```

这个复杂度会随 ROI 体积和 fine voxel 尺寸爆炸。例如 `0.5 m` 立方体配 `0.003 m` voxel，潜在 key 数量可达百万级。

正确方式：

```text
ROI
-> 枚举 ROI 覆盖的少量 mid-cell keys
-> 在 MidCellEntry[] 中查找这些 mid-cell
-> 读取每个 MidCellEntry 对应的 FineVoxelRef 连续 range
-> 只检查真实存在的 fine voxel refs 是否落入 ROI
```

也就是：

```text
ROI query = enumerate mid cells -> retrieve fine refs
ROI query != enumerate fine voxel grid
```

实现上可以用：

```text
sorted MidCellEntry[] + binary search
或本次 snapshot 临时 mid_key -> range 索引
```

临时索引只服务本次 snapshot，不是常驻 mid map。

## 9. Proposal 初筛

第一版初筛保持宽松。

使用：

```text
min_map_support_voxels
min_map_support_evidence
max_candidate_span_m
max_proposals_per_snapshot，可选
```

不要使用：

```text
严格整板尺寸
严格 aspect ratio
过早平面硬拒绝
```

原因：

```text
地图只看到高反贴，不一定看到底板轮廓
单个 seed 可能只来自一条反光贴
点少时平面拟合不稳定
```

## 10. 平面和二维几何

平面检查只在点足够时启用。

平面和二维 footprint 只使用：

```text
ROI 内 FineVoxelRef 的 voxel center / representative position
```

raw ring 点只用于 BPU window 构造，不能和 map evidence 点混在同一次 PCA 或 footprint 统计中。

结果分三类：

```text
valid_planar
insufficient_evidence
non_planar
```

如果点集近似线状，例如只有一条 1.5 cm 窄反光条，PCA 形式上也会给出一个法向，但该法向没有稳定几何意义。

设 PCA 特征值为：

```text
lambda1 >= lambda2 >= lambda3
```

只有当二维支撑足够，也就是 `lambda2` 足够大，并且 `lambda3 / lambda2` 足够小时，才可以输出 `valid_planar`。

若 `lambda2` 过小，说明点集 rank deficient / line-like，应归为：

```text
insufficient_evidence
```

而不是 `valid_planar`，也不是 `non_planar`。

规则：

```text
insufficient_evidence != reject
non_planar 只有在点数足够且残差明确很差时才 reject
```

如果平面 valid，再计算局部二维 footprint：

```text
width
height
span
occupancy
```

二维几何用于辅助过滤和日志诊断，不要第一版就做复杂模板匹配。

第一版中 `non_planar` 建议先作为日志和软抑制条件。只有真实测试证明非平面误检很多，且不会伤害真实板召回时，再升级为硬 reject。

### 10.1 Candidate Center 语义

第一版候选中心不是物理板中心。

候选模块输出的中心应理解为：

```text
proposal_center_map = 高反 evidence / ROI support 的中心
```

它不等于：

```text
physical_board_center
```

因为地图中通常只看到中间 20 x 25 cm 高反图案，而不是完整 40 x 40 cm 底板。

字段命名建议：

```text
proposal_center_map
support_bbox_map
roi_map
```

如果后续需要输出物理板中心或完整板姿态，应在 BPU 确认模板身份后，结合模板图案、局部平面和图案在板上的相对位置额外恢复。不要在候选提取阶段默认已经得到物理板中心。

## 11. BPU 输入

现有 BPU 模型的输入契约更接近：

```text
有限时间窗内的原始 candidate 点云
```

而不是：

```text
跨长时间、多视角、定位融合后的 map voxel 点
```

因此第一版验证路径：

```text
map proposal
-> 查找 raw ring 中仍可用的原始窗口
-> 用该窗口的原始高反点和当时 pose 做 ROI 关联
-> 回到旧 candidate preprocessing
-> 每个窗口按旧训练一致的 point / mask / meta 规则构造输入
-> BPU 打分
```

禁止把以下路径作为主验证：

```text
map ROI 内所有 fused voxel
-> 直接拼成点云
-> 送旧 BPU
```

map-space ROI 的作用只是找到相关 source frame / source window。真正送 BPU 的输入必须尽量等价于旧 pipeline 的输入分布。

### 11.1 BPU 输入等价性回归测试

Phase 2 接 BPU 前必须做回归测试。

对同一 bag、同一时间窗、同一真实候选，比较：

```text
旧 pipeline 直接构造的 BPU 输入
vs
raw ring + ROI 查找后构造的 BPU 输入
```

至少比较：

```text
有效点数
mask
candidate meta
PCA / 中心化结果，若旧 pipeline 使用
采样后的点坐标
BPU score
```

要求二者在规定容差内一致。若不一致，优先修复输入构造，而不是调 BPU threshold。

### 11.2 BPU 任务语义

必须明确当前 BPU 是哪种任务。

模式 A：固定模板验证。

```text
S_j = score(T_fixed, W_j)
```

用于判断当前候选是否匹配一个固定目标模板。

模式 B：多模板 / 多板号检索。

```text
S_i = Agg_j score(T_i, W_j)
template_id = argmax_i S_i
```

此时 BPU 调用量上界接近：

```text
N_proposal * K_max_windows * N_template
```

第一版必须在配置和日志中写明使用模式 A 还是模式 B。若使用模式 B，还必须定义 template 调度、top-k 规则、最终 score threshold 和 margin 规则。

第一版实现时必须实际选定一种模式，不能只保留两个抽象选项。否则以下内容都无法冻结：

```text
BPU inference budget
score threshold
template margin
candidate 输出语义
```

Phase 2 实现前必须读取当前 BPU 模型输入列表，确认 template 是：

```text
运行时输入
固定常量
或由不同 .bin 表示
```

未确认前不得实现 mode B。

当前阶段若只需要验证一个固定目标板，Phase 2 优先冻结为：

```text
mode A: fixed-template verification
```

### 11.3 Raw Window 不可用

如果找不到 raw window：

```text
candidate = unverified
reason = raw_unavailable
不 reject
不作为负样本
```

这是必须规则。

原因：

```text
full snapshot 能发现历史地图里的板
bounded raw ring 只保存最近窗口
历史板可能已经没有 raw window
```

### 11.4 Candidate Record 与 BPU Admission

proposal 进入 BPU 前必须先关联 candidate record。

流程：

```text
ROI-level dedup
-> candidate record association
-> cooldown / new-evidence check
-> BPU admission budget
-> raw-window lookup
-> BPU
```

`new evidence` 应来自 ROI 内 fine voxel 的：

```text
support_revision
或 evidence version 的有效变化
```

不能因为 full snapshot 再次扫描到该 ROI 就认为有 new evidence，也不能因为同一批已饱和 voxel 的 `last_seen` 更新就反复触发 BPU。

Phase 2 一接入 BPU，就必须有硬预算：

```text
max_bpu_evals_per_snapshot
或 max_bpu_evals_per_sec
```

预算单位必须明确为真实 BPU inference 次数，也就是：

```text
template-window pair inference
```

多模板模式下：

```text
N_infer = sum_over_candidates(K_c * N_template_c)
```

`max_bpu_evals_per_snapshot` 限制的是 `N_infer`，不是 proposal 数，也不是 raw window 数。

超过预算时：

```text
按 evidence / raw-window quality 排序
优先处理高优先级 proposal
剩余 proposal 标记 deferred
不 reject
```

这属于资源保护，不是复杂状态机。

第一版还必须二选一：

```text
BPU 同步执行
```

或：

```text
BPU 异步执行，但 max_inflight_bpu_jobs 有硬上限
```

若采用异步，job 必须携带：

```text
candidate_id
map_epoch
support_revision
ring_generation
```

BPU 返回时，只有这些版本仍匹配当前 candidate record，结果才能写回；否则丢弃旧结果。

### 11.5 Raw Frame 与 BPU Window

raw frame 不等于 BPU window。

Phase 2 必须定义：

```text
window_length_frames / window_length_sec
window_stride_frames / window_stride_sec
window anchor rule
window 内候选点如何按旧 pipeline 构造
```

如果旧 pipeline 使用固定窗口，例如：

```text
W = 10 frames
S = 5 frames
```

则 Phase 2 必须从 raw ring 中恢复同一套窗口集合：

```text
map proposal
-> 找相关 source frame
-> 生成旧定义下的 W-frame window
-> 调旧 candidate preprocessing
-> 构造 BPU 输入
```

不能在 proposal 出现后任意挑若干 frame 拼成输入。

若 `raw_ring_max_points_per_frame` 触发截断，必须满足：

```text
截断确定性
保留旧 preprocessing 所需字段
被截断 frame 带 raw_truncated 标记
等价性回归不能把 truncated frame 当作正常通过
```

## 12. 多窗口评分

如果一个 candidate 有多个 raw windows，不要无限取最大值。

需要固定：

```text
K_max
window selection rule
score aggregation rule
score threshold
```

第一版可以简单：

```text
每个 candidate 最多 K_max 个窗口
按窗口内有效高反点数或 ROI 覆盖排序
取 top K
score aggregation = median
```

不要用无限 max：

```text
S = max(S_i)
```

因为窗口越多，越容易偶然高分。

固定模板验证模式下：

```text
candidate_score = median(score(W_1 ... W_K))
```

多模板检索模式下：

```text
template_score_i = median(score(T_i, W_1 ... W_K))
template_id = argmax_i template_score_i
candidate_score = template_score_template_id
```

如果有多个模板分数接近，需要 margin 规则：

```text
best_score - second_score >= bpu_template_margin
```

否则标记为：

```text
ambiguous_template
```

不要把低 margin 的结果当成稳定板号输出。

如果窗口数量不足：

```text
可用窗口数 < min_bpu_windows_per_candidate:
  进入 insufficient_windows
不要直接负判
```

## 13. 去重

去重要轻量，不做复杂 grouping。

### 13.1 BPU 前 ROI 去重

目的：

```text
多个 seed 扩出的 ROI 覆盖同一块板
避免重复跑 BPU
```

规则可以很简单：

```text
ROI overlap 很大
或 shared mid-cell support 很多
或来自同一个 seed neighborhood
```

中心距离只能作为辅助条件，不能单独决定去重。

原因：

```text
两块板可能在同一视野里相距不远
同一块板的高反 support center 会随视角和 ROI completion 抖动
```

如果使用中心距离，必须配合：

```text
roi_iou >= roi_dedup_iou_threshold
或 shared_support_ratio >= roi_dedup_shared_support_threshold
```

被合并时保留：

```text
evidence 更强
raw window quality 更好
last_seen 更新
```

### 13.2 BPU 后 Board 去重

verified candidate 之间如果：

```text
同一 template_id，若有板号
且 bbox/footprint 重叠
或 support overlap 明显
```

保留分数更高或 evidence 更强的一个。

如果没有可靠 template_id，只能使用更保守的规则：

```text
中心距离 < board_dedup_distance_m
且 ROI/footprint overlap 明显
```

`board_dedup_distance_m` 不能大于场地中相邻两块真实板可能出现的最小距离。否则会把真实相邻板合并。

第一版不做复杂 NMS。

## 14. 在线与离线边界

### 14.1 在线模式

第一版优先在线：

```text
候选刚进入地图
raw ring 仍有对应窗口
可以用旧 BPU 验证
```

### 14.2 离线全图模式

离线全图重新识别不是第一版目标。

如果后续需要，需要额外方案：

```text
保存 candidate source snippets
保存 bag 并支持回放查窗口
训练 map-domain verifier
几何/模板 map-domain 验证
```

不能假设 bounded raw ring 能验证所有历史板。

## 15. 参数清单

参数必须分级管理，不能把所有参数都当成可 sweep 的自由变量。

### 15.1 固定实现合同

这类参数/规则原则上不用于调参：

```text
grid_origin_map
数学 floor 量化规则
d_bridge_cells = 1，第一版默认
ROI 查询方式 = mid-cell group -> FineVoxelRef range
BPU admission 版本规则 = map_epoch + support_revision + ring_generation
BPU task mode，第一版必须实际选定
```

### 15.2 物理先验派生

这类参数来自板尺寸、图案尺寸和场地布置：

```text
roi_completion_margin_m
max_candidate_span_m
board_dedup_distance_m
min_board_separation_m，任务假设/日志项
```

### 15.3 资源保护

这类参数用于保证 CPU、内存和 BPU 调用有上界：

```text
raw_ring_horizon_sec
raw_ring_max_frames
raw_ring_max_points_per_frame
max_bpu_windows_per_candidate
max_bpu_evals_per_snapshot
max_inflight_bpu_jobs，若 BPU 异步
max_candidate_records
candidate_ttl_sec
bpu_cooldown_sec
```

### 15.4 允许在 validation 上搜索的少数参数

第一版主要调这些：

```text
fine_voxel_size_m
evidence_saturation_count
evidence_aggregation_cap
mid_cell_size_m
mid_active_evidence_threshold
roi_completion_margin_m
min_map_support_voxels
min_map_support_evidence
min_raw_window_points
min_bpu_valid_points
bpu_score_aggregation_rule
bpu_score_threshold
roi_dedup_distance_m
roi_dedup_iou_threshold
roi_dedup_shared_support_threshold
```

这些参数要能解释来源：

```text
板尺寸
高反图案最大间隔
地图误差
资源预算
BPU 速度
```

不要把大量参数变成自由调参空间。

其中 `d_bridge_cells` 是离散 mid-cell 桥接距离，不是“膨胀次数”的模糊写法。日志中必须打印：

```text
mid_cell_size_m
d_bridge_cells
roi_completion_margin_m
```

否则后面很难复现实验。

`min_*` 参数必须拆开，不得用一个 `min_candidate_points` 同时表示不同阶段：

```text
min_map_support_voxels:
  ROI 内 active FineVoxelRef / voxel center 数

min_map_support_evidence:
  ROI 内 capped evidence 总量

min_raw_window_points:
  某个 raw window 内可用于旧 preprocessing 的原始高反点数

min_bpu_valid_points:
  送入 BPU 后 mask 中的有效点数
```

## 16. 评价指标

BPU 前必须先验证 proposal 本身。

主要指标：

```text
proposal recall:
  每块真实板是否至少被一个 ROI 覆盖

proposal coverage:
  ROI 覆盖的真实板高反 support 点数 / 该真实板高反 support 总点数

proposals per snapshot
duplicates per board
false proposals per snapshot
unique false proposal locations
extract runtime p50 / p95
peak scratch memory
raw window availability ratio
BPU calls per snapshot
overflow_seed_count
overflow_affected_recall
T_first_proposal
T_verified
```

`proposal recall` 不能只看“有没有碰到一点”。建议用两个层级：

```text
covered:
  coverage >= coverage_min_for_recall

well_covered:
  coverage >= coverage_min_for_bpu
```

这样可以区分：

```text
候选提取完全漏检
候选提取碰到了但 ROI 不够完整
BPU 输入窗口质量不足
```

false proposal 也要区分：

```text
per snapshot false proposals:
  看在线负载和 BPU 调用压力

unique false proposal locations:
  看地图里是否存在长期稳定误检区域
```

overflow region 不得计入 successful proposal recall。

指标拆分：

```text
proposal recall:
  只统计产生了有效 ROI 的真实板

overflow_affected_recall:
  真实板落在 overflow region 中、但没有形成合法 proposal 的比例
```

否则会出现：

```text
真实板在 overflow marker 里
日志看起来“被看到”
但系统无法真正进入 raw lookup / BPU
```

在线时延必须单独统计：

```text
T_first_proposal:
  从板第一次进入有效观测，到第一次出现覆盖充分 ROI 的时延

T_verified:
  从板第一次可观测，到 BPU 输出 verified 的时延
```

否则可能出现：

```text
最终地图 proposal recall 很高
但在线模式下板经过视野时来不及验证
```

判断原则：

```text
如果 proposal recall 不够，先改 seed/ROI completion
如果 proposal 太多，先改 ROI 去重和宽松筛选
如果 BPU 调用太多，先加预算和 cooldown
如果 CPU/内存超，再考虑 dirty-region
```

参数调优协议：

```text
candidate extractor 的参数只在 validation bag / trajectory 上调
test map 必须独立构建
不得把同一连续采集轨迹的相邻窗口拆到不同集合后再做 map-level 评估
```

因为地图会融合跨帧证据，map-level 评估比单窗口评估更容易发生时间邻近泄漏。

Ground Truth manifest 必须独立保存，至少包含：

```text
bag / trajectory id
map_epoch
board instance id
board 的人工标注高反 support
或 board 的已知 map-space pose + 设计模板映射
板首次达到有效观测条件的时间
```

`T_first_proposal` 的起点不是板第一次进入 LiDAR FOV，而是第一次达到“可用于当前系统的最小有效观测条件”。

## 17. 实现阶段

### Phase 1: 候选覆盖验证

必做：

```text
FineVoxelRef / MidCellEntry 临时数组
临时 mid-cell 聚合
明确 d_bridge_cells 的 seed region
通过 MidCellEntry range 做 ROI completion
宽松筛选
overflow seed 可观测化
ROI debug marker
proposal recall / coverage 日志统计
```

Phase 1 不实现：

```text
raw ring window 重建
BPU 输入构造
BPU 模式选择
candidate record support_revision admission
复杂 record 淘汰
```

这些属于 Phase 2 前置冻结项。

目标：

```text
真实板基本都被 ROI 覆盖
proposal 数量可控
```

BPU 不是本阶段重点。

### Phase 2: 接现有 BPU

必做：

```text
candidate record association
support_revision 驱动的 cooldown / new-evidence check
BPU inference budget，单位为 template-window pair
ROI -> raw ring window lookup
frame AABB / mid-key coarse filtering
旧 BPU 输入构造
BPU 输入等价性回归测试
K_max window selection
固定 score aggregation
raw_unavailable 不 reject
轻量去重
```

### Phase 3: 资源优化

只有测试证明需要时再做：

```text
dirty-region
更强 cooldown
进程合并
内存压缩
```

### Phase 4: 离线能力

只有项目需要离线全图重识别时再做：

```text
candidate snippets
bag replay lookup
map-domain verifier
```

## 18. 当前共识

采用：

```text
fine evidence map
+ bounded raw ring
+ temporary mid-cell seed
+ ROI completion
+ loose filtering
+ existing BPU verification
+ light dedup
```

暂不采用：

```text
常驻 mid/coarse maps
通用 micro-component grouping
复杂状态机
无限 raw cache
离线全图 verifier
```

这不是因为这些机制永远没用，而是因为当前任务不需要一开始就承担它们的复杂度。只有当真实测试暴露具体问题时，再针对性增加。

## 19. 第一版实现合同汇总

为了避免实现时重新产生歧义，第一版必须遵守以下合同：

```text
1. ROI 查询不枚举 fine voxel grid
   只通过 mid-cell group 查真实存在的 FineVoxelRef。

2. seed bridge 使用明确的 d_bridge_cells
   不使用“膨胀一格 + 26 邻接”这种含糊规则。

3. raw ring 必须保存坐标合同
   明确 raw points 是 map-space 还是 sensor-space；
   若是 sensor-space，必须同时保存 T_map_lidar_used_for_mapping。

4. snapshot 和 raw ring 必须有生命周期合同
   extractor 持有 FineVoxelRef[]，不持有 mapper live hash map 引用；
   BPU job 持有稳定 raw frame 副本、shared_ptr 或 pinned slot。

5. map_epoch 改变后不跨 epoch 查询
   清空或失效 raw ring 和 candidate record，新 epoch 重新积累。

6. candidate record 必须参与 BPU admission
   cooldown 和 new-evidence 依据 support_revision / evidence version 的有效变化，
   不是依据 full snapshot 是否再次扫到，也不是依据饱和 voxel 的 last_seen 更新。

7. BPU 一接入就有调用预算
   预算单位是 template-window pair inference；
   超预算的 proposal deferred，不 reject。

8. BPU 输入必须回归验证等价性
   raw ring + ROI 找回的输入应与旧 pipeline 输入在点数、mask、meta、采样点和 score 上接近。

9. overflow region 不静默丢弃
   标记 deferred_overflow，发布 debug marker，并进入日志统计。

10. candidate center 是高反 support center
   不是完整物理板中心。

11. snapshot 不积压
   max_pending_snapshots = 1；
   忙时只保留最新 snapshot，旧 pending snapshot 合并/覆盖。

12. mid active 使用固定布尔规则
   evidence_sum >= threshold
   AND active_fine_count >= threshold。

13. ROI 形状固定
   第一版采用 seed support bbox 的 map-frame AABB expand；
   不同时保留 sphere / AABB 两种语义。

14. candidate association 与 support_revision 使用固定算法
   定义 record 匹配条件、support signature、evidence bucket、
   revision 更新条件和 record 淘汰顺序。

15. raw frame 与 BPU window 分离
   明确定义旧 pipeline 的 window 长度、步长、锚点和重建规则。

16. Phase 2 前冻结 BPU 实际模式
   确认当前 .bin 的 template 输入能力；
   固定 mode A 或 mode B、K_min、K_max、score aggregation。
```

这些合同不是额外系统，而是让简单方案保持确定、可复现、可测试的边界。
