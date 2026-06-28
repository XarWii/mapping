# 动态高反精细建图系统落地计划

本文档记录当前动态高反建图路线的阶段性共识、已观察到的实验事实、下一版系统骨架，以及后续验证顺序。

它不是最终算法说明书，而是接下来实现和讨论的工程边界文档。

## 1. 当前目标

当前目标不是重写完整 LIO，也不是引入回环。

系统输入固定为：

```text
Livox MID-360S raw point cloud
Livox MID-360S internal IMU
Super-LIO /lio/odom as black-box external odom
```

系统输出目标为：

```text
高反点精细地图
可诊断的 raw / candidate / mature 分层地图
可用于后续候选提取和 BPU 识别的稳定 map evidence
```

核心约束：

```text
不估计全局位姿
不做回环
不让点云反向修改 /lio/odom
优先建立可观测、可切换的最小完整系统骨架
```

## 2. 当前重要事实

### 2.1 硬件外参和同步

MID-360S 的 LiDAR 与 IMU 为同设备内置组合，硬件时间同步和官方外参默认可信。

同一套设备运行 FAST-LIO 效果很好，因此当前不把硬件级 LiDAR-IMU 同步或标定错误作为主线嫌疑。

但这不等于软件层时间合同和坐标链一定正确。仍需保护以下边界：

```text
header stamp / timebase / offset_time 的参考语义
/lio/odom pose 表达的是 T_OI、T_OL 还是 T_OB
T_IL / T_LI 使用方向
外参是否漏乘或重复乘
PoseAt(t) 是否真正 bracket point time
scan start / scan end 是否被混用
```

结论：

```text
硬件同步/外参：默认可信
软件时间合同/坐标链：必须用日志和单元测试保护
```

### 2.2 坐标系合同

本文档后续统一使用：

```text
O: /lio/odom 的 parent frame，也是当前 reflective local map frame
I: MID-360S internal IMU frame
L: MID-360S LiDAR frame
```

不默认把 `/lio/odom` 的 parent frame 称为全局 `map`。它在实际系统中可能叫：

```text
world
odom
camera_init
lio_odom
```

当前不做回环和全局地图拼接，因此 reflective map 只在同一个连续 `O` frame / odom segment 内有效。

固定变换合同：

```text
/lio/odom.pose = T_OI
T_OL = T_OI * T_IL
p_O = T_OL * p_L
```

其中 `T_IL` 表示：

```text
p_I = T_IL * p_L
```

外参只允许应用一次。

### 2.3 /lio/odom frame 判断

已查看 Super-LIO 发布代码，`/lio/odom` 更符合 IMU pose：

```text
T_OI
```

因此当前 mapper 配置采用：

```yaml
pose_input_frame: "imu"
```

切换为 IMU pose 后，旋转残差显著下降，说明之前确实存在 pose frame 解释问题。

### 2.4 已做 A/B 实验

旧配置中：

```text
translation residual avg ~= 0.54 m
rotation residual avg ~= 0.139 rad
```

改为 `pose_input_frame=imu` 并增强 pose 权重后：

```text
translation residual avg ~= 0.16-0.18 m
rotation residual avg ~= 0.001-0.002 rad
```

后续实验：

```text
hard_pose_anchor:
  视觉无明显改善
  translation residual 仍约 0.17 m

关闭 IMU translation deskew:
  肉眼略有改善
  map update 比例略升

external previous/current odom interpolation:
  肉眼继续略有改善
  AABB 异常略少
  但仍不是质变
```

当前判断：

```text
内部 IMU 平移传播不应作为第一版主 deskew 轨迹
外部 odom 插值方向更合理
临时 previous/current 插值不是严格 PoseAt(t)
点云噪声和 map evidence 缺失仍会污染最终地图
```

## 3. 必须先修正的诊断语义

当前日志中的 `0.16-0.18 m` translation residual 不能直接等价为“轨迹真的漂了 18 cm”。

它可能混入：

```text
LiDAR 原点与 IMU 原点杆臂
correction 前状态
correction 后状态
不同 reference time
不同 frame 下的位置
```

尤其需要检查它是否接近：

```text
|t_IL|
```

如果 residual 两端实际比较的是 `p_OL` 与 `p_OI`，那么它稳定接近杆臂长度是合理的，不能被解释为 IMU 平移传播漂移。

在 residual 语义审计通过前，不再用这个数评价 trajectory 改动是否有效。

下一版必须把日志分为两类。

### 3.1 坐标链不变量

该类指标用于验证外参、frame 和原点关系，不叫 trajectory residual。

```text
d_IL_O = p_OL - p_OI
d_IL_O should equal R_OI * t_IL
|d_IL_O| should equal |t_IL|
```

对应日志字段：

```text
lever_arm_norm_config
lever_arm_norm_observed
lever_arm_consistency_error
```

### 3.2 同参考点轨迹误差

只有比较同一个物理参考点、同一个时刻、同一个坐标系，才叫 trajectory residual。

```text
predicted_vs_odom_translation_error_OI
predicted_vs_odom_rotation_error_OI
post_anchor_translation_assertion_error_OI
post_anchor_rotation_assertion_error_OI
```

其中 `OI` 表示比较对象都是 IMU origin 在 odom parent frame `O` 下的 pose。

禁止继续使用含糊字段名：

```text
translation residual
rotation residual
```

旧字段若保留，只能作为 legacy debug，并必须注明比较两端。

下一版日志还必须明确区分：

```text
pre_correction_residual_OI
post_correction_residual_OI
predicted_scan_end_vs_odom_OI
lidar_origin_vs_imu_origin_delta
```

hard anchor 后必须检查断言：

```text
|p_OI_post - p_OI_odom| < epsilon_p
angle(R_OI_post^T R_OI_odom) < epsilon_R
```

若断言成立，而旧 residual 仍显示约 0.17 m，则旧 residual 是诊断定义问题，不是 anchor 失败。

## 4. 第一版主架构

第一版采用最小完整骨架，而不是同时开启所有复杂策略。

主链路：

```text
raw Livox scan
-> parse absolute point time
-> pre-deskew hard gate
-> derive effective reflective [t_s, t_e]
-> OdomCoverage / scan readiness
-> PoseAt(t_i)
-> per-point deskew to map
-> scan-local observation aggregation
-> map evidence update
-> raw / candidate / mature debug publish
```

核心原则：

```text
主轨迹 = 外部 /lio/odom 的 OdomBuffer::PoseAt(t)
点云只更新地图证据，不反向改 pose
IMU 平移传播不作为第一版生产主路径
```

建议参数化切换：

```yaml
trajectory_mode:
  odom_poseat
  external_previous_current_debug
  legacy_imu_single_anchor_debug
  imu_rotation_debug
  imu_full_debug

map_mode:
  raw_direct_debug
  candidate_mature
```

默认第一版：

```yaml
trajectory_mode: odom_poseat
map_mode: candidate_mature
```

debug 发布由独立开关控制，不混入 `map_mode` 枚举。

`trajectory_mode` 必须是互斥状态机，而不是多个布尔开关叠加。`odom_poseat` 模式下：

```text
禁止 legacy ApplyPoseCorrection 参与 deskew
禁止 hard_pose_anchor 参与 deskew
禁止 IMU translation deskew 参与 deskew
```

旧 IMU/correction 路径只允许作为 debug 模式显式启用。

## 5. Trajectory 层合同

### 5.1 OdomBuffer::PoseAt(t)

`PoseAt(t)` 是第一版唯一主 deskew 轨迹接口。

输入：

```text
t_query = point_time + odom_time_offset_sec
```

输出：

```text
valid
mode: exact / interpolated / invalid
left_stamp
right_stamp
interpolation_gap
odom_segment_id
pose T_OI(t)
```

禁止外推。

只有当存在：

```text
t_left <= t_query <= t_right
```

时，才允许返回 exact 或 interpolated。

第一版采用 scan-level all-or-nothing：

```text
eligible point =
  finite
  && point_time valid
  && reflectivity >= threshold
  && raw range valid

t_s = eligible points 的最小绝对时间
t_e = eligible points 的最大绝对时间

若没有 eligible point:
  SCAN_EMPTY，直接结束，不进入 WAIT_ODOM

CheckCoverage(t_s + odom_time_offset_sec,
              t_e + odom_time_offset_sec,
              max_odom_bracket_gap_sec) 必须 valid
否则整帧 WAIT_ODOM 或 SKIPPED，不更新任何地图状态
```

第一版不做 point-level partial drop。否则早段或晚段系统性缺失时，很难区分是高反可见性变化、时间字段错误，还是 odom buffer 覆盖不足。

### 5.2 OdomCoverage 与 scan readiness

端点 `PoseAt(t_s)`、`PoseAt(t_e)` 有效还不够。第一版需要 scan 级覆盖接口证明整个有效扫描时间区间连续可用：

```cpp
OdomCoverageResult CheckCoverage(
    double t_begin,
    double t_end,
    double max_gap_sec);
```

返回字段至少包括：

```text
valid
segment_id
max_gap_in_interval
has_future_odom
crossed_jump
reason
```

scan readiness：

```text
READY =
  all eligible points time-valid
  && OdomCoverage(q_s, q_e) valid
  && single odom_segment_id
  && max_gap_in_interval <= max_odom_bracket_gap_sec

q_s = t_s + odom_time_offset_sec
q_e = t_e + odom_time_offset_sec
```

scan 必须按时间顺序处理：

```text
pending scan FIFO
只处理队首 scan
队首 READY -> 处理
队首 SKIPPED_* -> 记录后弹出
队首 WAIT_ODOM -> 后续 scan 不得越过它更新地图
```

第一版 scan 状态：

```text
SCAN_EMPTY:
  没有 eligible high-ref point，直接结束。

WAIT_ODOM:
  当前 buffer 最新 odom 尚未覆盖 q_e，scan 保留在 pending queue 等待。

SKIPPED_ODOM_GAP:
  已有相关 odom，但区间内 gap > max_odom_bracket_gap_sec，不再等待。

SKIPPED_ODOM_JUMP:
  有效时间区间跨越 odom segment / jump，直接丢弃。

SKIPPED_TIME_INVALID:
  point_time / timebase / offset_time 无法构成有效时间区间，直接丢弃。

DROPPED_TIMEOUT:
  WAIT_ODOM 超过最大等待时间，或 pending queue 超容量，直接丢弃并记录。
```

### 5.3 外部 odom 插值公式

两个 odom 样本：

```text
(t_k, p_k, R_k)
(t_{k+1}, p_{k+1}, R_{k+1})
```

若：

```text
t_k <= t <= t_{k+1}
```

则：

```text
alpha = (t - t_k) / (t_{k+1} - t_k)
p(t) = (1 - alpha) * p_k + alpha * p_{k+1}
R(t) = Slerp(R_k, R_{k+1}, alpha)
```

若 `/lio/odom` 为 `T_OI`，则 LiDAR pose：

```text
T_OL(t) = T_OI(t) * T_IL
```

每个点：

```text
p_O = T_OL(t_i) * p_L
```

外参只应用一次。

### 5.4 时间偏移入口

保留显式参数：

```yaml
odom_time_offset_sec: 0.0
```

查询使用：

```text
PoseAt(point_time + odom_time_offset_sec)
```

符号约定：

```text
odom_time_offset_sec > 0:
  用更晚时刻的 external odom 解释当前 LiDAR point

odom_time_offset_sec < 0:
  用更早时刻的 external odom 解释当前 LiDAR point
```

第一版固定为 `0.0`，不根据单次 RViz 观察调参。后续若要 sweep，使用固定 bag 和固定指标比较。

若 residual 与速度强相关，优先怀疑 odom 时间偏移。

若 residual 与角速度强相关，优先检查时间偏移、姿态插值和 frame 链。

## 6. 点云去噪与 Scan-Local Observation

第一版不做激进删除。

原则：

```text
先压缩重复证据，不删除稀疏真实观测
先标签化，不强行硬删
```

### 6.1 硬合法性 gate

deskew 前必须检查：

```text
finite
point_time valid
reflectivity >= threshold
raw lidar range valid
```

range 使用原始 LiDAR frame：

```text
r = |p_L|
```

不要使用 map frame 下的点模长做 range gate。

Livox tag 第一版只统计和打标签。只有确认某些 tag 含义明确无效后，才加入 hard reject。

### 6.2 Scan-local micro-voxel aggregation

deskew 到 map 后，对同一 scan 内的高反点做 micro-voxel 聚合。

目标不是删除真实点，而是让同一 scan 内同一区域最多形成一个 observation。

每个 scan-local observation 保存：

```text
mean position
raw_point_count
mean intensity
max intensity
tag histogram
raw range min / max / mean
point time span
quality flags
```

单点 observation 保留，但标为 weak 或 isolated。

### 6.3 标签化质量

第一版 observation 标签：

```text
STRONG
WEAK
ISOLATED
TAG_SUSPECT
```

示例：

```text
同一 micro-voxel 多个原始点:
  STRONG

单点但 tag、强度、距离正常:
  WEAK

单点且 tag 可疑、距离远、附近没有局部点:
  ISOLATED

tag 含义不确定:
  TAG_SUSPECT
```

标签不直接决定删除。第一版最多只影响 observation weight，不影响 mature 升级门槛。

第一版进一步收缩为：

```text
标签只做诊断；
最多只影响 observation weight；
mature 升级门槛不随标签变化。
```

建议初始权重：

```text
STRONG: w = 1.0
WEAK: w = w_weak
ISOLATED: w = w_isolated
TAG_SUSPECT: w = w_tag_suspect
```

`support_scan_count` 仍然只表示不同 scan 命中次数，不按标签设置不同成熟门槛。

### 6.4 Scan-level support 上限

同一 scan 对同一 map state 的 support 最多增加一次。

这是第一版最重要的去噪规则之一：

```text
同一 scan 内重复点不能把 evidence 快速刷高
```

## 7. Map Evidence 层

地图状态单位不再是 raw point，而是 scan-local aggregate observation。

### 7.0 Scan voxel 与 map hash voxel

第一版明确使用两层概念：

```text
scan_voxel:
  只用于同一 scan 内把重复高反点压缩成一个 observation

map_hash_voxel:
  只用于快速查找附近地图状态，不直接决定唯一归属
```

真实关联不能只查询同一个 map hash voxel。体素边界两侧的近邻点应能关联到同一个状态。

邻域大小不能写死为 26 邻域，而必须由最大关联半径和 map hash voxel 尺寸决定：

```text
r_query = max(r_accept, r_conflict, r_candidate)
k = ceil(r_query / map_hash_voxel_size) + 1

查询中心体素周围 (2k + 1)^3 个 hash voxel，
或查询所有与半径 r_query 球相交的 hash voxel。
```

然后从其中的 candidate / mature / conflict state 中按距离走确定决策树：

```text
1. 查询 r_conflict 内最近 mature M。

2. 若存在 M 且 d(z, M) <= r_accept:
  update mature

3. 若存在 M 且 r_accept < d(z, M) <= r_conflict:
  M 的 conflict ring 独占该 observation
  update / create M 的 conflict candidate
  不允许普通 candidate 抢占

4. 若 r_conflict 内没有 mature:
  查询 FREE_CANDIDATE
  若最近 FREE_CANDIDATE 距离 <= r_candidate:
    update candidate
  否则:
    create FREE_CANDIDATE

```

约束：

```text
r_accept < r_conflict
r_candidate 独立配置，不默认等于 r_accept 或 r_conflict
```

candidate 逻辑身份：

```text
FREE_CANDIDATE:
  owner_mature_id = none

CONFLICT_CANDIDATE:
  owner_mature_id = specific mature state id
```

只要 observation 进入 mature 的 conflict ring，就不允许普通 candidate 抢占。

同一个 map state 在同一个 scan 中最多执行一次 support 增加，也最多执行一次状态融合。

实现可采用：

```text
先完成 observation -> state 关联
同一 state 在同一 scan 收到的多个 observation 做 scan-level merge
scan 结束后每个 state 只提交一次更新
```

或维护：

```text
last_update_scan_id
pending_scan_aggregate
```

state 的 mean position 更新后若跨过 hash voxel 边界，必须重新索引。

容量约束：

```text
max_free_candidates_per_hash_voxel
max_mature_states_per_hash_voxel
```

候选超限时优先淘汰：

```text
低 support
低 weight
最久未见
RMS 最大
```

第一版状态：

```text
CANDIDATE
MATURE
CONFLICT_CANDIDATE
EXPIRED
```

### 7.1 Candidate

新 observation 周围没有可接受的 candidate 或 mature 时，创建 candidate。

candidate 保存：

```text
mean position
RMS / scatter
support_scan_count
first_seen_time
last_seen_time
last_seen_scan_id
sensor_origin_first
sensor_origin_last
max_sensor_baseline
view direction bins
quality flags
weight sum
```

`support_scan_count` 必须是不同 scan 数量。

### 7.2 Mature

candidate 升级条件第一版采用保守规则：

```text
support_scan_count >= min_mature_scan_count
last_seen_time - first_seen_time >= min_mature_duration_sec
RMS <= max_mature_rms_m
```

第一版不强制多视角基线，以免误伤高反稀疏观测。

但从第一版开始记录并 debug 输出：

```text
max_sensor_baseline
view direction bins
sensor_origin_first / last
```

若后续发现稳定多径大量成熟，再把 `max_sensor_baseline >= min_mature_baseline_m` 作为可配置升级条件，而不是重做状态结构。

### 7.3 Conflict Candidate

对已有 mature 点 `mu_m`，新 observation `z` 若满足：

```text
r_accept < |z - mu_m| <= r_conflict
```

则不直接融合 mature，也不立即生成普通新结构，而进入：

```text
CONFLICT_CANDIDATE
```

第一版规则：

```text
每个 mature 周围最多一个 conflict candidate
conflict candidate 使用更短 TTL
不直接发布
不自动拖动 mature
重复出现时标记 persistent conflict
不立即升级成第二个 mature
```

若已有 conflict candidate `C`，新 conflict observation 仍属于同一 mature `M`，但距离 `C` 也很大：

```text
若接近已有 C:
  update C

若不接近已有 C:
  conflict_multimodal_count += 1
  只记录为 conflict outlier
  不创建第二个 C
  不移动已有 C
```

这样做优先避免：

```text
轨迹跳层 -> 稳定长成第二层成熟地图
```

### 7.4 Mature 更新原则

mature 只在 observation 被接受时更新：

```text
|z - mu_m| <= r_accept
```

更新采用有上限权重：

```text
W_new = min(W_old + w, W_max)
```

异常点不能拖动 mature 均值。

### 7.5 Odom segment 切换策略

若检测到：

```text
ODOM_JUMP
odom time reset
new odom_segment_id
parent frame change
```

第一版采用 automatic segment break，不保留“人工 reset 或新开 segment”二选一。

检测到新 `odom_segment_id` 后：

```text
1. 当前跨 segment 的 scan -> SKIPPED_ODOM_JUMP
2. 清空 active CANDIDATE
3. 清空 active CONFLICT_CANDIDATE
4. 冻结旧 active MATURE map，不再更新
5. 新建 active map_epoch / map_segment_id
6. 新 segment 从空 map evidence 开始
7. 发布明确的 map reset / epoch change 事件
```

当前系统不做回环，也不做全局位姿优化，因此不能假设 jump 后旧地图能自动搬到新坐标系。

active map 输出采用全量替换语义：

```text
/reflective/mature_map
```

每次发布代表当前 active segment 的完整成熟地图快照。

segment 切换时：

```text
先发布空 mature map
再发布新的 map_epoch
随后开始发布新 segment 的 snapshot
```

自定义消息或独立状态话题必须携带：

```text
map_epoch
map_segment_id
```

否则 candidate extractor、decision 或 BPU 上游无法知道同一 topic 中的点已经不是同一张地图。

ODOM_JUMP 检测由 `OdomBuffer` 统一负责，包括：

```text
timestamp regression
parent frame change
explicit source reset
translation jump
rotation jump
```

连续两条 odom 的 jump gate：

```text
|p_{k+1} - p_k| > max_external_velocity_mps * dt + external_position_margin_m

or

angle(R_k^T R_{k+1}) >
  max_external_angular_velocity_rps * dt + external_rotation_margin_rad
```

阈值作为参数，但 segment id 的递增、跨段 scan 的处理和 active map epoch 的切换必须集中在 `OdomBuffer` / mapper segment 管理边界内，不能散落在多个模块。

## 8. Debug Topic 与日志

第一版必须保留多层输出，避免 mature map 看起来干净后掩盖 trajectory 问题。

`raw` 只表示滚动 debug cloud，不表示长期 raw map。

```text
raw debug cloud:
  当前 scan 或最近 N 秒 / N 帧的原始高反 deskew 观测

candidate map:
  当前存活候选状态

mature map:
  当前稳定地图状态

conflict map:
  当前存活冲突候选与冲突统计
```

建议 topic：

```text
/reflective/deskewed_high_ref_cloud_raw
/reflective/scan_observation_debug
/reflective/candidate_map
/reflective/mature_map
/reflective/conflict_map
/reflective/map_snapshot
```

日志频率不宜过高，但种类要完整。

每 1-2 秒输出一次：

```text
scan id
raw points
high-ref points
valid point_time count
PoseAt valid / invalid / gap stats
scan-local observation count
STRONG / WEAK / ISOLATED / TAG_SUSPECT count
candidate insert / update
mature promoted / update
conflict created / persistent
expired count
nearest mature residual stats
CPU / RSS / PSS
```

`odom_poseat` 模式必须记录：

```text
scan_time_begin / end
PoseAt valid ratio
odom bracket gap P50 / P95 / max
odom segment id
point timestamp span
odom query time span
T_OI input
T_OL used
lever-arm consistency
scan pose delta
```

`imu_*_debug` 模式才额外记录：

```text
IMU predicted end state
IMU predicted vs odom end residual
scan-local correction twist
post-correction endpoint assertion
residual projection on velocity direction
residual projection on angular velocity direction
```

## 9. 与 Super-LIO / FAST-LIO 的对比边界

不对比完整滤波器。

只对比共同子任务：

```text
Livox 点时间解析
offset_time / timebase 使用
tag 处理
LiDAR-IMU 外参方向
deskew 后点云
高反点投影位置
局部地图融合效果
```

FAST-LIO 在同设备上效果很好，说明硬件同步和官方外参可信。但 mapper 仍必须证明自己没有在软件层误用时间或坐标链。

### 9.1 BPU 输入隔离原则

Map Evidence 不等于 BPU 的真实观测输入。

```text
mature map:
  用于候选发现、空间证据和 ROI 引导

BPU input:
  后续仍应从真实观测窗口重新提取
```

不要直接把经过多帧均值、体素聚合和 mature 滤波后的地图点当成 BPU 输入点云，否则容易造成训练/部署分布不一致。

## 10. 实现顺序

### Phase A0：坐标链与 residual 语义审计

```text
固定 O / I / L 坐标合同
计算 |t_IL|
统一 residual 两端为 T_OI
hard anchor 后做 post-anchor assert
记录 residual 与速度、角速度、姿态的相关性
明确旧 residual 是否混入杆臂
```

### Phase A1：时间合同与 scan readiness

```text
明确 header / timebase / offset_time
计算 scan 有效 t_s / t_e
定义 OdomBuffer 等待策略
定义同 segment 约束
定义最大 bracket gap
第一版采用整帧 all-or-nothing
```

### Phase B：OdomBuffer::PoseAt(t)

```text
纯 external T_OI 插值
无外推
逐点 bracket
返回 bracket 诊断信息
只应用一次 T_IL
legacy correction 与 odom_poseat 互斥
全点云仅 debug，高反点作为生产路径
```

### Phase C：Scan-local Observation

```text
finite / time / intensity / range gate
tag 统计和标签
map-frame micro-voxel aggregation
STRONG / WEAK / ISOLATED / TAG_SUSPECT
scan-level support 上限
```

### Phase D：Map Evidence

```text
CANDIDATE
MATURE
CONFLICT_CANDIDATE
TTL / expiry
odom segment 切换策略
mature-only publish
raw / candidate / mature / conflict debug publish
```

### Phase E：再讨论 IMU 增强

只有在 PoseAt 基线和 map evidence 都稳定后，再比较：

```text
odom-only
odom + IMU rotation debug
双端 anchor trajectory
IMU full debug
```

## 11. 第一版验收标准

第一版不是要求地图立刻完美，而是要求系统能回答“问题在哪一层”。

最低验收：

```text
PoseAt(t) 无外推
每帧能统计 odom bracket gap
raw high-ref map 可看
scan-local observation 可看
candidate map 可看
mature map 可看
conflict 可统计
CPU/RSS 可记录
日志能区分 trajectory、点云噪声、evidence 三类问题
```

若 mature map 变干净但 raw map 仍明显分层，说明 evidence 在抑制污染，但 trajectory 仍需查。

若 raw/candidate/mature 都分层，优先查 PoseAt、时间合同和坐标链。

若 raw map 毛刺多、mature map 稳定，说明点云噪声主要由 evidence 层解决。

### 11.1 固定评估协议

不同版本必须用固定协议比较，避免只凭单次 RViz 观感。

固定条件：

```text
同一 rosbag
同一起始时刻
同一 reflectivity threshold
同一 range gate
同一外参配置
odom_time_offset_sec 固定
若干固定 ROI：
  一处墙面/门框
  一处反光板
  一处高反噪声区域
```

每版输出：

```text
high-ref retained count
scan-local observation count
candidate count
mature count
mature conflict rate
nearest-mature residual P50 / P95
固定 ROI 内跨 scan centroid RMS
固定 ROI 内 component spatial thickness
CPU / RSS / PSS
```

`map update ratio` 只能作为辅助指标，且必须明确定义：

```text
update_ratio =
  accepted_to_existing_state / valid_scan_local_observations
```

在新区域、高反稀疏区域或 candidate 初期，update ratio 天然可能很低，不能单独作为质量主指标。

## 12. 当前推荐默认实验配置

下一版基线建议：

```yaml
pose_input_frame: "imu"
trajectory_mode: "odom_poseat"
odom_time_offset_sec: 0.0
max_odom_bracket_gap_sec: 0.15
map_mode: "candidate_mature"
publish_raw_debug: true
publish_candidate_debug: true
publish_mature_debug: true
publish_conflict_debug: true
```

`max_odom_bracket_gap_sec` 不是固定真值。它必须由 `/lio/odom` timestamp 间隔分布确定：

```text
P50
P95
P99
max
```

阈值应能容忍正常 jitter，但拒绝丢包断裂，不按单次 RViz 视觉效果直接调。

legacy/debug 参数只在对应 debug mode 下允许生效：

```yaml
trajectory_mode: "legacy_imu_single_anchor_debug"
hard_pose_anchor: true

trajectory_mode: "imu_full_debug"
deskew_use_imu_translation: true

trajectory_mode: "external_previous_current_debug"
deskew_use_external_pose_interpolation: true
```

其中 `deskew_use_external_pose_interpolation` 是临时 previous/current 实验开关，后续应由严格 `OdomBuffer::PoseAt(t)` 替代。

在 `trajectory_mode: "odom_poseat"` 下，以上 legacy/debug 布尔开关不得参与生产 deskew。
