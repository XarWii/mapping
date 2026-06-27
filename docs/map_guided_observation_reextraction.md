# Map-Guided Observation Re-Extraction Implementation

本文档只记录动态建图路线中识别输入的具体实现方案。

## 1. 目标链路

动态建图路线采用两域结构：

```text
Livox raw scan
-> high-ref filtering
-> IMU + external odom deskew
-> A. map-domain high-ref evidence map
-> B. observation-domain bounded source ring

map-domain candidate ROI
+ observation-domain source windows
-> ROI-guided old candidate extraction
-> existing BPU preprocessing
-> existing BPU model
-> aggregated RecognitionScores
```

地图域只负责空间发现、候选编号、ROI、当前位置和几何支撑。

观测域只负责保存真实观测窗口，并为旧 BPU 输入契约重建候选点云。

## 2. 当前动态路线的替换点

当前动态路线为：

```text
reflective_candidate_extractor
-> ReflectiveCandidate.cloud_sensor
-> reflective_bpu_scorer
-> RecognitionScores
```

第一版改为：

```text
dynamic_reflective_mapping
-> ReflectiveObservationFrame

reflective_candidate_extractor
-> ReflectiveCandidates with roi/support fields

reflective_map_decision
-> ReflectiveRecognitionRequest

reflective_observation_bpu_scorer
-> RecognitionScores
```

`reflective_bpu_scorer_node` 不再用于动态建图 launch。它保留在代码中，但动态建图 route 不启动它。

`ReflectiveCandidate.cloud_sensor` 在动态建图 route 中不再作为 BPU 输入使用。第一版保留该字段以兼容消息定义和 RViz/debug，但 scorer 不读取它。

## 3. Map Domain

### 3.1 Mapper 输出

`dynamic_reflective_mapping_node` 继续发布：

```text
/reflective/map_snapshot
/reflective/rolling_map
/reflective/deskewed_high_ref_cloud
```

其中：

```text
/reflective/map_snapshot:
  用于 candidate extractor

/reflective/rolling_map:
  用于 RViz

/reflective/deskewed_high_ref_cloud:
  用于调试，不作为 BPU 权威输入
```

新增：

```text
/reflective/observation_frame
```

该话题发布每帧可用于重建 BPU 输入的真实观测源。

### 3.2 Candidate Extractor 输出

`reflective_candidate_extractor_node` 继续从 `/reflective/map_snapshot` 中提取候选。

候选输出语义为：

```text
candidate_id:
  map epoch 内稳定的空间候选 ID

center_map:
  map frame 下高反 support 的中心

support_bbox_map:
  候选真实 map support 的 AABB

roi_map:
  support_bbox_map 按 roi_completion_margin_m 扩张后的 AABB

support_revision:
  同一 candidate_id 下 support 发生实质变化时递增

cloud_map:
  map-domain support/debug 点云

cloud_sensor:
  动态建图 route 中置空或仅用于兼容，不作为 BPU 输入
```

`candidate_id` 只表示空间候选，不表示物理板号，也不表示 BPU template id。

### 3.3 消息字段调整

`ReflectiveCandidate.msg` 增加以下字段：

```text
geometry_msgs/Point support_min_map
geometry_msgs/Point support_max_map
geometry_msgs/Point roi_min_map
geometry_msgs/Point roi_max_map
uint32 support_revision
```

字段含义：

```text
support_min_map / support_max_map:
  ROI completion 后实际纳入 candidate 的 map support AABB

roi_min_map / roi_max_map:
  用于 observation lookup 的 map-space ROI AABB

support_revision:
  candidate record 内部版本号
```

`support_revision` 由 `reflective_candidate_extractor_node` 独占维护。它表示 map support 相对 extractor 内部 `revision_anchor` 的变化，不表示 BPU 是否已经评估过。

scorer / decision 维护独立状态：

```text
last_scored_support_revision[candidate_id]
```

当：

```text
candidate.support_revision > last_scored_support_revision[candidate_id]
```

时，decision 才把该 candidate 视为“有新 support 可重新入 BPU”。scorer 完成评分后，decision 更新 `last_scored_support_revision`。

`support_revision` 更新规则：

```text
同一 candidate_id 第一次出现:
  support_revision = 1
  revision_anchor = current support summary

后续 snapshot 若相对 revision_anchor 满足任一条件:
  active voxel 数正增长 >= support_revision_min_voxel_growth
  capped evidence 正增长 >= support_revision_min_evidence_growth
  support bbox 任一方向向外扩张 >= support_revision_min_bbox_expand_m
  active mid-cell support 集合 Jaccard 变化率 >= support_revision_key_change_ratio

则:
  support_revision += 1
  revision_anchor = current support summary

否则:
  support_revision 保持不变
```

support 减少、TTL 淘汰、容量淘汰、视角变稀疏导致的 evidence/voxel 下降，不触发 `support_revision` 增加。

active mid-cell support 集合 Jaccard 变化率定义：

```text
r_change = 1 - |S_current ∩ S_anchor| / |S_current ∪ S_anchor|
```

其中 `S` 使用 candidate ROI 内 active mid-cell key 集合，不使用 fine voxel key 集合。

同一 `map_epoch` 内 `candidate_id` 不复用。candidate record 失效后，其 id 不能分配给新的空间对象。

第一版参数：

```yaml
support_revision_min_evidence_growth: 20
support_revision_min_voxel_growth: 12
support_revision_min_bbox_expand_m: 0.03
support_revision_key_change_ratio: 0.35
```

### 3.4 Snapshot 与 Request 版本字段

`ReflectiveCandidates.msg` 增加：

```text
uint64 candidate_snapshot_id
```

`candidate_snapshot_id` 由 `reflective_candidate_extractor_node` 在每次发布 candidates 时递增。它表示一批候选的不可变快照版本。

`ReflectiveRecognitionRequest.msg` 增加：

```text
uint64 request_id
uint64 candidate_snapshot_id
uint32[] candidate_ids
uint32[] support_revisions
```

字段约定：

```text
request_id:
  reflective_map_decision 每次发布 request 时递增

candidate_snapshot_id:
  request 针对的候选快照

candidate_ids / support_revisions:
  本次 request 要评分的候选及其 support 版本
  两个数组长度必须相等
```

`RecognitionScoreEntry.msg` 增加：

```text
uint32 support_revision
uint8 status
bool score_valid
uint32 windows_used
bool center_sensor_valid
string center_sensor_frame_id
time center_sensor_stamp
```

`RecognitionScoreEntry.status` 枚举：

```text
0 SCORED
1 RAW_UNAVAILABLE
2 INSUFFICIENT_WINDOWS
3 INSUFFICIENT_POINTS
4 NO_MATCHING_CLUSTER
5 BUDGET_DEFERRED
6 TIMEOUT
7 STALE_VERSION
8 WINDOW_EXTRACTION_BUDGET_DEFERRED
```

只有：

```text
status == SCORED && score_valid == true
```

时，`score` 才是有效 BPU 分数。其他状态表示无证据或未完成，不能当作低分拒识。

`RecognitionScores.msg` 增加：

```text
uint64 request_id
uint64 candidate_snapshot_id
```

decision 模块只接受以下版本完全一致的结果：

```text
map_epoch
request_id
candidate_snapshot_id
candidate_id
support_revision
```

### 3.5 BPU Admission Retry State

decision 为每个 candidate record 维护：

```text
last_scored_support_revision
last_checked_window_end_scan_id
last_status
last_quality_signature
```

候选可重新进入 scorer 的条件：

```text
1. candidate.support_revision > last_scored_support_revision

2. last_status 属于可重试状态，
   且 observation ring 出现了比 last_checked_window_end_scan_id 更新的 eligible source window

3. last_status == BUDGET_DEFERRED，
   且 scorer 预算窗口重新开放

4. last_status == WINDOW_EXTRACTION_BUDGET_DEFERRED，
   且 window extraction 预算窗口重新开放

5. last_status == INSUFFICIENT_POINTS，
   且新窗口的 rough quality 明显高于 last_quality_signature
```

状态对 `last_scored_support_revision` 的推进规则：

```text
SCORED:
  更新 last_scored_support_revision = candidate.support_revision

BUDGET_DEFERRED:
  不更新

WINDOW_EXTRACTION_BUDGET_DEFERRED:
  不更新

RAW_UNAVAILABLE:
  不更新

INSUFFICIENT_WINDOWS:
  不更新

INSUFFICIENT_POINTS:
  不更新

NO_MATCHING_CLUSTER:
  不更新；但同一 candidate_id + window_end_scan_id 不重复尝试

TIMEOUT:
  不更新

STALE_VERSION:
  不更新
```

## 4. Observation Domain

### 4.0 Observation Source Contract

`observation ring` 的点源必须与 no-mapping BPU 路线的源点一致，或是其严格超集。不得在写入 ring 前丢弃旧 BPU 路线本来会参与候选提取的点。

必须冻结以下逐点合同：

```text
原始点字段:
  x / y / z / reflectivity 或 intensity

坐标表达:
  单帧内去畸变到该帧 reference LiDAR frame

强度阈值:
  与 no-mapping route 相同

距离门限:
  与 no-mapping route 相同

NaN / invalid point 过滤:
  与 no-mapping route 相同

去畸变参考时刻:
  与 no-mapping route 的窗口参考定义兼容

LiDAR-IMU 外参:
  与 mapper 和 no-mapping route 共用同一份配置

外部 odom 插值 / 匹配:
  与 mapper 写入 map 时使用的 pose contract 一致

多帧窗口拼接顺序:
  与 no-mapping route 相同

单帧内部点顺序:
  保持驱动 / 原始点云顺序，不经过 unordered 容器重排
```

mapper 与 no-mapping route 应复用同一份高反筛选函数和同一组配置参数。若两条路线需要不同参数，`observation ring` 必须保存旧 BPU 路线所需的严格超集，再由 observation scorer 按旧参数过滤。

`ReflectiveObservationFrame` 记录：

```text
uint64 observation_source_contract_hash
```

hash 输入包括：

```text
reflectivity threshold
distance gate
invalid point rule
deskew reference rule
extrinsic config version
odom pose contract version
frame ordering rule
point ordering rule
```

A/C 回归时，A 与 C 的 `observation_source_contract_hash` 不一致则拒绝比较。

### 4.1 新消息 ReflectiveObservationFrame

新增消息：

```text
# msg/ReflectiveObservationFrame.msg
Header header
uint32 map_epoch
uint64 scan_id
uint64 observation_source_contract_hash
geometry_msgs/Pose lidar_pose_in_map
geometry_msgs/Point map_aabb_min
geometry_msgs/Point map_aabb_max
bool truncated
sensor_msgs/PointCloud2 cloud_lidar
```

字段约定：

```text
header.stamp:
  该 observation frame 的 reference time；必须与 no-mapping 旧 pipeline 的窗口参考时刻一致

header.frame_id:
  lidar frame 名称，仅表示 cloud_lidar 的参考坐标语义

map_epoch:
  与 map_snapshot / candidates 使用同一 epoch

scan_id:
  mapper 内递增 scan id

lidar_pose_in_map:
  T_map_lidar_ref，必须对应 header.stamp 的同一参考时刻

map_aabb_min / map_aabb_max:
  cloud_lidar 中所有点投到 map frame 后的 AABB

truncated:
  本帧高反点超过 max_observation_points_per_frame 后被截断

cloud_lidar:
  frame-local deskewed high-ref points
  fields: x, y, z, intensity
  type: float32
```

`cloud_lidar` 是 BPU 的权威源表示。`map_aabb_*` 和 `lidar_pose_in_map` 只用于检索和坐标变换。

`cloud_lidar` 中每个点必须已经去畸变到该帧 `header.stamp` 对应的同一个 LiDAR 参考坐标系。该参考时刻、外参和外部 odom 插值规则必须与 no-mapping 旧 pipeline 保持一致。

### 4.2 Mapper 生成 observation frame

在 `dynamic_reflective_mapping_node::CommitScan()` 中，当前已经计算：

```text
point_lidar_reference
point_local / point_map
external_rotation
external_translation
```

第一版增加：

```text
observation_points_lidar.push_back(point_lidar_reference, intensity)
observation_points_map_aabb.expand(point_map)
```

每帧最多保存：

```yaml
max_observation_points_per_frame: 4096
```

超过上限：

```text
truncated = true
该 frame 默认不参与 BPU source window
dropped_observation_point_budget += 1
```

第一版不使用 truncated frame 构造 BPU 输入。若后续必须使用 truncated frame，点保留策略必须确定、可复现，并通过 A/C 回归证明不会改变最终 selected cluster 与 BPU 输入。

`cloud_lidar` 只保存已通过高反阈值、距离门限、去畸变成功的点。

### 4.3 Observation Ring

新增节点 `reflective_observation_bpu_scorer_node` 维护短时全局 ring：

```text
std::deque<ObservationFrame> observation_ring
```

每个 `ObservationFrame` 保存：

```text
map_epoch
scan_id
stamp
T_map_lidar
T_lidar_map
map_aabb
truncated
points_lidar[]
```

Ring 上限：

```yaml
observation_ring_max_frames: 40
observation_ring_max_age_sec: 4.0
observation_ring_max_points_per_frame: 4096
```

淘汰规则：

```text
插入新 frame 后:
  删除超过 observation_ring_max_frames 的最老 frame
  删除 stamp 早于 now - observation_ring_max_age_sec 的 frame

收到 map_epoch 改变:
  清空 ring
  清空 pending recognition state
```

Ring 中只保存高反点，不保存原始全量点云。

## 5. Source Window

### 5.1 Raw frame 与 BPU window

`ReflectiveObservationFrame` 是单帧。

BPU source window 是若干连续 observation frame 的组合。

第一版窗口构造：

```text
window_end_frame = selected frame
window_frames = ring 中以 window_end_frame 结尾的连续 W 帧
W = bpu_window_frames
```

默认参数：

```yaml
bpu_window_frames: 10
max_frame_gap_sec: 0.15
```

正常识别模式固定使用旧 no-mapping pipeline 的窗口长度、窗口锚点和步长。不足 `bpu_window_frames`、scan_id 不连续、或相邻 frame 时间间隔超过 `max_frame_gap_sec` 的窗口无效，输出 `insufficient_windows`，不降级为 1 到 W-1 帧输入。

在线降级模式可在后续单独实现，但其结果不得与旧 pipeline 的等价回归混用，也不得直接沿用正常模式阈值做强判。

窗口参考系：

```text
L_ref = 旧 no-mapping pipeline 定义的窗口参考 LiDAR frame
```

若旧 pipeline 使用窗口末帧参考系，则：

```text
L_ref = window_end_frame lidar reference frame
```

若旧 pipeline 使用窗口首帧或中心帧参考系，则动态路线必须改为相同定义。

对窗口内每一帧 `F_j`：

```text
p_Lref = T_Lref_map * T_map_Lj * p_Lj
```

其中：

```text
T_map_Lj:
  F_j.lidar_pose_in_map

T_Lref_map:
  inverse(L_ref.lidar_pose_in_map)
```

窗口点云全部统一到 `L_ref` 后进入旧候选提取逻辑。

### 5.2 Window 预筛

对每个 candidate ROI，只考虑满足以下条件的 frame：

```text
frame.map_epoch == candidate.map_epoch
frame.map_aabb intersects candidate.roi_map
frame.truncated == false
```

从候选 frame 中生成 source windows。

第一阶段在整个有效 ring 范围内查找候选 window end frame，不只检查最近若干帧。

候选 end frame 必须满足：

```text
end_frame.map_epoch == candidate.map_epoch
end_frame.map_aabb intersects candidate.roi_map
end_frame.truncated == false
以 end_frame 结尾可以形成完整 bpu_window_frames 连续窗口
窗口内所有 frame 均 truncated=false
窗口内所有 frame 均 map_epoch 相同
窗口满足连续 scan_id 或 max_frame_gap_sec 约束
```

对每个候选 end frame 先计算廉价质量：

```text
window 内各 frame 的 ROI overlap rough point count 总和
window 内与 ROI AABB 相交的 frame 数
window end frame 与 candidate.center_map 的粗略距离
window end frame 的视角 / 距离质量
window truncated count，第一版必须为 0
```

第二阶段从候选 end frame 中选择少量 source windows 进入完整窗口重建和聚类。

默认最多完整提取：

```yaml
max_candidate_source_windows_to_check: 8
```

选择策略：

```text
先按廉价质量排序
依次选择 end frame
新窗口必须与已选窗口满足时间或位姿去冗余
达到 max_candidate_source_windows_to_check 后停止
```

去冗余条件第一版使用 scan 间隔：

```yaml
min_window_separation_frames: 10
```

也就是两个被选 source window 的 end scan_id 至少间隔 `bpu_window_frames`，尽量避免 10 帧窗口之间高度重叠。

后续可增加位姿去冗余：

```yaml
min_window_translation_m: 0.15
min_window_rotation_rad: 0.10
```

## 6. ROI-Guided Old Candidate Extraction

### 6.1 窗口内旧候选提取

对每个 source window：

```text
1. 将窗口点统一到 L_ref
2. 运行旧 EuclideanCluster
3. 运行旧 ApplyCandidateFilter
4. 运行 min_inference_cluster_points 过滤
```

步骤 2 的输入是 source window 内全部高反点。第一版不允许在聚类前按 `roi_map` 裁剪点。

第一版复用 `siamese_bpu_infer_node.cpp` 中的参数语义：

```yaml
cluster_tolerance_m: 0.25
min_cluster_points: 3
max_cluster_points: 2000
min_inference_cluster_points: 8
enable_size_filter: true
enable_plane_filter: true
size_filter_min_long_axis_m: 0.12
size_filter_max_long_axis_m: 0.35
size_filter_min_short_axis_m: 0.06
size_filter_max_short_axis_m: 0.30
plane_filter_max_thickness_m: 0.08
```

这些参数先从 `siamese_bpu_infer.yaml` 拷贝到动态建图 yaml 的 observation scorer 段，保证动态 route 与 no-mapping route 使用同一套候选域。

### 6.2 ROI 匹配

对窗口内每个 cluster：

```text
cluster_Lref
-> 使用 T_map_Lref 投到 map frame
-> 计算 cluster_map_aabb
-> 计算与 candidate.roi_map 的 AABB IoU
-> 计算 cluster center 到 candidate.center_map 的距离
```

选择规则：

```text
候选 cluster 必须满足:
  cluster_map_aabb intersects candidate.roi_map
  roi_overlap_point_count >= min_roi_overlap_points
  center_distance <= max_roi_center_distance_m

在满足条件的 cluster 中:
  先选 roi_overlap_point_count 最大者
  若并列，选 center_distance 最小者
```

默认参数：

```yaml
min_roi_overlap_points: 8
max_roi_center_distance_m: 0.35
```

被选中的 cluster 作为该 candidate 在该 source window 下的 BPU source candidate。

### 6.3 禁止直接 ROI 裁点

实现中不允许使用以下路径作为默认 BPU 输入：

```text
candidate.roi_map
-> 直接裁剪 raw window 中所有落入 ROI 的点
-> BPU
```

ROI 只用于选择旧候选提取结果，不能直接定义 BPU 点集。

ROI 在第一版只允许用于：

```text
筛 source frame
筛 source window
选择 window 内旧候选提取得到的 cluster
```

ROI 不允许用于：

```text
聚类前裁剪点
改变 EuclideanCluster 的输入点集
改变 ApplyCandidateFilter 的输入 cluster
```

若后续全窗口聚类成为 CPU 瓶颈，可以新增宽松预门控实验；门控 margin 必须覆盖 `cluster_tolerance_m + candidate size margin`，并通过 A/C 回归证明 selected cluster 与 BPU 输入没有实质差异后才能默认启用。

### 6.4 Request 内 Window Extraction Cache

同一个 request 内以 source window 为中心复用聚类结果，不允许每个 candidate 对同一个 window 重复重建和聚类。

临时缓存 key：

```text
WindowExtractionKey {
  uint32_t map_epoch;
  uint64_t window_end_scan_id;
  uint64_t pipeline_config_hash;
}
```

缓存内容：

```text
window_points_lref
all EuclideanCluster results
ApplyCandidateFilter kept clusters
每个 cluster 的 map AABB
每个 cluster 的 map center
每个 cluster 的 point count
filter reason / reject reason
```

request 处理流程：

```text
1. 从 request candidates 汇总所有候选可能使用的 source window
2. 按 window quality 和候选优先级排序 unique windows
3. 每个 unique window 最多重建和聚类一次
4. 所有 candidate 复用该 window 的 cluster 列表做 roi_map 匹配
5. 再按 candidate 选择 top-K windows 并进入 BPU 预算队列
```

窗口聚类本身也要有硬预算：

```yaml
max_window_extractions_per_request: 12
```

超过预算的 window 标记为：

```text
window_extraction_budget_deferred
```

该状态不等于 reject。

### 6.5 Candidate-Window Pair Budget

request 内 candidate 数和 candidate-window pair 预处理也必须有上界。

参数：

```yaml
max_candidates_per_request: 32
max_candidate_window_pairs_to_prepare_per_request: 48
max_preprocessor_attempts_per_candidate: 2
```

Phase 2A 采用两级质量逻辑：

```text
预筛质量:
  cluster 原始点数
  roi_overlap_point_count
  center_distance
  window rough quality
  window 时间 / 位姿去冗余

仅对预筛 top-M candidate-window pair:
  运行 BpuPreprocessor
  计算 valid_point_count
  再进入 BPU inference 预算队列
```

超过 pair 预算的 candidate 输出：

```text
status=BUDGET_DEFERRED
score_valid=false
```

超过 `max_candidates_per_request` 时：

```text
按 request 显式 candidate 顺序、candidate evidence_count、support_revision 新旧排序
只处理预算内 candidate
预算外 candidate 输出 BUDGET_DEFERRED
```

## 7. BPU Inference

### 7.1 Preprocessor

被选中的 cluster 以 `L_ref` 坐标表达输入：

```text
std::vector<siamese_bpu::BpuPoint>
```

随后严格调用现有：

```cpp
BpuPreprocessor::prepare(points, reflectivity_threshold)
BpuModel::infer(candidatePoint, candidateMask, candidateMeta, candidateCountScale)
```

不新增 map-domain normalization。

不对地图点做 inverse-current-pose 后送 BPU。

### 7.2 单窗口结果

每个 source window 输出：

```text
candidate_id
window_end_scan_id
window_stamp
support_revision
source_point_count
valid_point_count
roi_overlap_point_count
quality_rank
score
```

这些结果先保存在节点内部，不新增 ROS 消息。

### 7.3 窗口评分阶段

Phase 2A 固定单窗口：

```text
min_bpu_windows_per_candidate = 1
max_bpu_windows_per_candidate = 1
使用质量最高的一个 source window
输出 score = single_window_score
```

Phase 2A 用于 A/C 等价性回归和旧阈值验证，不混入多窗口聚合。

窗口质量排序字典序：

```text
1. valid_point_count 高
2. roi_overlap_point_count 高
3. window_end_scan_id 与已选窗口间隔 >= min_window_separation_frames
```

Phase 2A 默认参数：

```yaml
min_bpu_windows_per_candidate: 1
max_bpu_windows_per_candidate: 1
min_window_separation_frames: 10
min_bpu_valid_points: 8
```

Phase 2B 才启用多窗口：

```text
max_bpu_windows_per_candidate = 3
candidate_score = median(score(W_1), score(W_2), score(W_3))
```

Phase 2B 必须重新在 validation 数据上标定 aggregation 后阈值。若后续允许 `K` 可变，`RecognitionScoreEntry` 必须携带 `windows_used`，并按不同 K 分析阈值稳定性。

若可用窗口数 `< min_bpu_windows_per_candidate`：

```text
发布该 candidate 的 RecognitionScoreEntry，status=INSUFFICIENT_WINDOWS，score_valid=false
记录 reason=insufficient_windows
```

若某窗口 `valid_point_count < min_bpu_valid_points`：

```text
该窗口不参与 BPU
记录 reason=insufficient_points
```

### 7.4 BPU 任务语义与预算

第一版动态建图 route 使用当前部署模型文件：

```text
v8_c3_bag_qat.bin
```

scorer 启动时必须显式配置并记录 BPU 任务语义：

```text
bpu_task_mode:
  fixed_template_verifier
  multi_class_classifier
  runtime_template_matcher
```

第一版默认实现 `fixed_template_verifier`：

```text
单模型输出一个 logit-like score
RecognitionScoreEntry.score 表示该固定 verifier 的匹配分数
```

若现有模型实际用于区分 board_01 / board_02，则必须把任务模式配置为对应的 `multi_class_classifier` 或 `runtime_template_matcher`，并同步扩展 `RecognitionScores` 的输出语义。scorer 不允许在任务模式不明确时启动。

实现前必须通过模型信息和 no-mapping route 确认：

```text
输入张量列表
template 是否为运行时输入
输出是单 score、双分类还是多类别 logits
no-mapping route 如何把输出映射成 board_id / template_id
```

第一版不引入 map-domain BPU 模型，也不引入 map/raw 双模型融合。

scorer 自身必须限制真实 BPU inference 次数：

```yaml
max_bpu_inferences_per_request: 24
max_bpu_inferences_per_sec: 30
max_inflight_bpu_jobs: 1
```

预算单位为：

```text
一次 BpuModel::infer 调用
```

若候选窗口数超过预算：

```text
按以下优先级排序:
  1. request 中显式指定的 candidate 顺序
  2. candidate evidence_count 高
  3. source window quality 高
  4. support_revision 更新较新

预算内候选执行 BPU
预算外候选标记 budget_deferred
budget_deferred 不作为 reject
```

### 7.5 输出 RecognitionScores

`reflective_observation_bpu_scorer_node` 发布现有：

```text
/reflective/recognition_scores
```

消息字段保持：

```text
header = request.header
map_epoch = request.map_epoch
request_id = request.request_id
candidate_snapshot_id = request.candidate_snapshot_id
entries[].candidate_id = candidate_id
entries[].support_revision = candidate.support_revision
entries[].score = aggregated median score
entries[].status = SCORED / failure status
entries[].score_valid = status == SCORED
entries[].windows_used = used BPU source window count
entries[].center_map = candidate.center_map
entries[].center_sensor = T_lidar_map(t_request) * candidate.center_map, only if center_sensor_valid
entries[].center_sensor_valid = true only when request-time lidar pose is available
entries[].center_sensor_frame_id = lidar frame used for center_sensor
entries[].center_sensor_stamp = request.header.stamp
entries[].voxel_count = candidate.voxel_count
entries[].evidence_count = candidate.evidence_count
```

`center_sensor` 的时间必须是 `request.header.stamp`。若 scorer 没有该时刻对应的 lidar pose，则：

```text
center_sensor_valid = false
center_sensor = (0, 0, 0)
center_sensor_frame_id = ""
center_sensor_stamp = request.header.stamp
```

并在日志中记录 `center_sensor_unavailable`。不得用 ring 中最近一帧或 selected window 的 `L_ref` 代替 request 时刻。下游只能在 `center_sensor_valid=true` 时使用 `center_sensor`。

`center_map` / `center_sensor` 都表示 high-ref support center，不表示物理板中心。

窗口细节通过 ROS 日志和 debug dump 输出。

## 8. Recognition Request 处理

`reflective_observation_bpu_scorer_node` 订阅：

```text
/reflective/candidates
/reflective/recognition_request
/reflective/observation_frame
```

处理规则：

```text
收到 candidates:
  按 candidate_snapshot_id 保存 latest_candidates

收到 recognition_request:
  若已有 pending_request 或正在 scoring:
    用最新 request 覆盖旧 pending_request
    coalesced_request_count += 1
  保存 pending_request
  若存在 candidate_snapshot_id/map_epoch 匹配的 candidates:
    立刻执行 scoring

收到 observation_frame:
  写入 ring
  若存在 pending_request 且 candidates 版本匹配:
    可再次尝试 scoring
```

匹配条件：

```text
request.map_epoch == candidates.map_epoch
request.candidate_snapshot_id == candidates.candidate_snapshot_id
request.candidate_ids[i] 存在于 candidates
request.support_revisions[i] == matched_candidate.support_revision
```

Phase 2A 不等待未来窗口：

```text
request 到达:
  只在当前 ring 中查找完整 source window

找到:
  评分并结束 request

当前 ring 没有任何与 ROI 相关的 source frame:
  发布 status=RAW_UNAVAILABLE

存在相关 source frame，但无法构成完整、连续、同 epoch、未截断 W 帧窗口:
  发布 status=INSUFFICIENT_WINDOWS
```

默认参数：

```yaml
request_timeout_sec: 0.0
max_pending_requests: 1
```

第一版不做无界 request 队列，也不等待未来窗口。scorer 忙时只保留最新 request，并通过日志记录 `coalesced_request_count`。旧 request 被合并不等于 reject。

request 合并规则：

```text
decision 发布新 request 时:
  本地将上一个未完成 request 标记为 superseded
  不再等待旧 request_id 的 RecognitionScores

scorer 覆盖 pending request 时:
  记录 superseded_request_id
  不保证再发布旧 request 的结果

active request 已经开始执行时:
  可以完成并发布
  decision 若发现 result.request_id 不是当前等待 request_id，则丢弃该结果
  不更新 candidate identity
```

后续若启用等待未来窗口：

```text
request_timeout_sec >= (bpu_window_frames - 1) / lidar_frame_rate_hz
                       + extraction_time_budget
                       + scheduling_margin
```

当前 10 Hz / W=10 时，等待式 timeout 不得低于 1.2s。

### 8.1 RequestContext

每个 active request 建立内部上下文：

```text
struct RequestContext {
  uint64_t request_id;
  uint64_t candidate_snapshot_id;
  uint32_t map_epoch;
  ros::Time deadline;
  std::vector<uint32_t> candidate_ids;
  std::vector<uint32_t> support_revisions;
  std::set<std::pair<uint32_t, uint64_t>> inferred_candidate_windows;
  std::unordered_map<uint32_t, CandidateStatus> per_candidate_status;
  std::unordered_map<uint32_t, std::vector<uint64_t>> selected_window_end_scan_ids;
  WindowExtractionCache window_cache;
  bool published;
};
```

约束：

```text
同一个 request_id 下，同一个 candidate_id + window_end_scan_id 最多 infer 一次。
同一个 request_id 只发布一次汇总 RecognitionScores。
request 完成或 timeout 后发布，未完成 candidate 以明确 status 输出。
published=true 后不再对该 request 重试。
```

第一版采用一次性汇总输出模式：

```text
不做增量 RecognitionScores
不发布 final=false 的半成品
```

### 8.2 CandidateSnapshotLite Cache

scorer 不缓存完整 `ReflectiveCandidates` 消息。收到 candidates 后提取轻量快照：

```text
struct CandidateSnapshotLite {
  uint64_t candidate_snapshot_id;
  uint32_t map_epoch;
  ros::Time stamp;
  std::vector<CandidateLite> candidates;
};

struct CandidateLite {
  uint32_t candidate_id;
  uint32_t support_revision;
  geometry_msgs::Point center_map;
  geometry_msgs::Point support_min_map;
  geometry_msgs::Point support_max_map;
  geometry_msgs::Point roi_min_map;
  geometry_msgs::Point roi_max_map;
  uint32_t evidence_count;
  uint32_t voxel_count;
};
```

缓存上限：

```yaml
max_cached_candidate_snapshots: 4
candidate_snapshot_cache_ttl_sec: 2.0
```

淘汰规则：

```text
超过 max_cached_candidate_snapshots:
  删除最老 snapshot

超过 candidate_snapshot_cache_ttl_sec:
  删除过期 snapshot

map_epoch 改变:
  清空 snapshot cache
```

request 到达时若找不到对应 `candidate_snapshot_id`：

```text
为 request 中每个 candidate 输出 status=STALE_VERSION
```

不得只在日志里记录版本不匹配。

## 9. Candidate Observation Evidence

第一版不向每个 candidate 复制 raw 点。

节点内部为每个 candidate 保存轻量 evidence：

```text
struct CandidateObservationEvidence {
  uint64_t request_id;
  uint64_t candidate_snapshot_id;
  uint32_t candidate_id;
  uint32_t map_epoch;
  uint32_t support_revision;
  uint64_t window_end_scan_id;
  std::vector<uint64_t> window_scan_ids;
  ros::Time stamp;
  int source_point_count;
  int valid_point_count;
  int roi_overlap_point_count;
  float score;
};
```

每个 candidate 最多保存：

```yaml
max_observation_evidence_per_candidate: 5
```

淘汰规则：

```text
同一 candidate_id:
  优先保留 support_revision 最新的 evidence
  同 revision 内保留 score 已计算且 quality 更高的 evidence
  超过上限删除质量最低或最老 evidence

map_epoch 改变:
  清空全部 evidence
```

第一版不保存 BPU-ready tensor。

## 10. Debug 与日志

新增 debug 话题：

```text
/reflective/bpu_source_candidate_cloud
```

内容：

```text
所有实际送入 BPU 的 selected cluster 点
frame_id = selected window L_ref
fields = x/y/z/intensity/candidate_id
```

日志每次 scoring 输出一行 summary：

```text
observation_bpu epoch=...
request_id=...
candidate_snapshot_id=...
candidates=...
scored=...
rejected(no_frames=..., no_cluster=..., insufficient_points=..., timeout=...)
budget_deferred=...
coalesced_requests=...
ring_frames=...
ring_points=...
```

每个被评分 candidate 输出：

```text
candidate=...
support_revision=...
windows_checked=...
windows_used=...
valid_points=[...]
overlap_points=[...]
score_values=[...]
score_median=...
```

日志频率：

```yaml
summary_log_throttle_sec: 1.0
candidate_log_throttle_sec: 0.5
```

## 11. Launch 与配置

`dynamic_reflective_mapping.launch` 第一版变更：

```text
保留:
  dynamic_reflective_mapping
  reflective_candidate_extractor
  reflective_map_decision
  reflective_board_pose_estimator
  resource_monitor_dynamic_total
  rviz

替换:
  reflective_bpu_scorer
  -> reflective_observation_bpu_scorer
```

新增 launch arg：

```xml
<arg name="start_observation_bpu_scorer" default="true"/>
```

动态建图 yaml 新增段：

```yaml
reflective_observation_bpu_scorer:
  observation_frame_topic: "/reflective/observation_frame"
  candidates_topic: "/reflective/candidates"
  recognition_request_topic: "/reflective/recognition_request"
  scores_topic: "/reflective/recognition_scores"
  debug_source_cloud_topic: "/reflective/bpu_source_candidate_cloud"
  bpu_model_path: "/home/sunrise/catkin_ws/siamese_pointnet_explore/deployment/rdk_x5/v8_c3_bag_bpu/v8_c3_bag_qat.bin"
  bpu_task_mode: "fixed_template_verifier"
  reflectivity_threshold: 160.0
  observation_source_contract_hash: 0

  observation_ring_max_frames: 40
  observation_ring_max_age_sec: 4.0
  observation_ring_max_points_per_frame: 4096
  max_cached_candidate_snapshots: 4
  candidate_snapshot_cache_ttl_sec: 2.0

  bpu_window_frames: 10
  max_frame_gap_sec: 0.15
  max_candidates_per_request: 32
  max_candidate_source_windows_to_check: 8
  max_window_extractions_per_request: 12
  max_candidate_window_pairs_to_prepare_per_request: 48
  max_preprocessor_attempts_per_candidate: 2
  max_bpu_windows_per_candidate: 1
  min_bpu_windows_per_candidate: 1
  min_window_separation_frames: 10
  min_bpu_valid_points: 8
  max_bpu_inferences_per_request: 24
  max_bpu_inferences_per_sec: 30
  max_inflight_bpu_jobs: 1

  cluster_tolerance_m: 0.25
  min_cluster_points: 3
  max_cluster_points: 2000
  min_inference_cluster_points: 8
  enable_size_filter: true
  enable_plane_filter: true
  size_filter_min_long_axis_m: 0.12
  size_filter_max_long_axis_m: 0.35
  size_filter_min_short_axis_m: 0.06
  size_filter_max_short_axis_m: 0.30
  plane_filter_max_thickness_m: 0.08

  min_roi_overlap_points: 8
  max_roi_center_distance_m: 0.35
  request_timeout_sec: 0.0
  max_pending_requests: 1
  max_observation_evidence_per_candidate: 5
  summary_log_throttle_sec: 1.0
  candidate_log_throttle_sec: 0.5
```

`dynamic_reflective_mapping` 段新增：

```yaml
publish_observation_frame: true
observation_frame_topic: "/reflective/observation_frame"
max_observation_points_per_frame: 4096
```

`resource_monitor_dynamic_total.target_node_names` 改为：

```text
dynamic_reflective_mapping,
reflective_candidate_extractor,
reflective_observation_bpu_scorer,
reflective_map_decision,
reflective_board_pose_estimator,
reflective_intensity_filter
```

## 12. 资源上界

Observation ring 点存储上界：

```text
40 frames * 4096 points/frame * 16 bytes/point = 2.5 MB
```

加上 frame pose、AABB、vector overhead，第一版按 4 MB 作为 ring 裸存储预算。observation scorer 的总内存还包括 ROS 消息副本、窗口临时点云、WindowExtractionCache、cluster 结果、debug cloud 和 BPU runtime，不得只按 ring 裸数组估算。

窗口构造临时点云上界：

```text
bpu_window_frames * observation_ring_max_points_per_frame
= 10 * 4096
= 40960 points
```

第一版不在聚类前按 ROI 裁剪窗口点。source window 内全部高反点统一到 `L_ref` 后进入旧 `EuclideanCluster`。

窗口聚类上界：

```yaml
max_window_extractions_per_request: 12
```

每个 request 内最多重建/聚类 12 个 unique source window。

第一版 BPU 调用上界：

```text
min(max_bpu_inferences_per_request,
    candidate_window_pairs_after_quality_sort)
```

其中：

```yaml
max_bpu_inferences_per_request: 24
max_bpu_windows_per_candidate: 1
```

Phase 2B 启用 top-3 + median 后再改为：

```yaml
max_bpu_windows_per_candidate: 3
```

`reflective_map_decision` 控制 request 触发时机，`reflective_observation_bpu_scorer_node` 自身仍必须执行 `max_bpu_inferences_per_request`、`max_bpu_inferences_per_sec` 和 `max_inflight_bpu_jobs` 三个硬预算。

资源日志新增：

```text
ring_points
window_cache_count
window_cache_points
cluster_temp_peak_points
candidate_window_pairs_checked
candidate_window_pairs_prepared
preprocessor_attempts
unique_window_extractions
window_extraction_budget_deferred
candidate_pair_budget_deferred
bpu_inferences_used
bpu_budget_deferred
```

这些指标用于区分负载来自 BPU，还是来自 full-window clustering / window cache。

## 13. A/B/C 回归验证工具

新增离线调试模式：

```text
rosparam:
  debug_dump_observation_bpu_inputs: true
  debug_dump_dir: "/tmp/reflective_observation_bpu_debug"
```

每次 BPU 输入落盘：

```text
candidate_{candidate_id}_window_{scan_id}/
  points_lref.csv
  candidate_point.bin
  candidate_mask.bin
  candidate_meta.bin
  candidate_count_scale.txt
  score.txt
  meta.yaml
```

`meta.yaml` 包含：

```text
map_epoch
request_id
candidate_snapshot_id
candidate_id
support_revision
window_end_scan_id
window_frame_ids
window_reference_stamp
window_reference_frame_rule
source_point_count
valid_point_count
roi_overlap_point_count
selected_cluster_index
selected_cluster_map_aabb
selected_cluster_center_distance
score
```

额外落盘：

```text
all_clusters.csv:
  cluster_index
  point_count
  bbox_lref
  bbox_map
  filter_result
  filter_reason
  roi_overlap_point_count
  center_distance

preprocessor_debug.yaml:
  robust_center
  valid_ratio
  count_scale
  sampled_indices
  coord_clip_ratio
```

用于对比：

```text
A. no-mapping old route BPU input
B. current map-fused pseudo sensor input
C. observation-ring re-extracted input
```

判断依据：

```text
C 的 candidate_point / mask / meta / score 应接近 A
B 若与 A/C 明显偏离，则停用 map-fused pseudo sensor 输入
```

A 与 C 必须使用：

```text
同一组 scan_id
同一窗口长度
同一 L_ref
同一 cluster/filter 参数
同一 BpuPreprocessor 参数
同一 observation_source_contract_hash
```

输入顺序合同：

```text
窗口内按旧 pipeline 相同的 frame 顺序拼接
每帧内部保持驱动 / 原始点云顺序
不得经过 unordered_map / unordered_set 等不稳定容器重排
采样、FPS、PCA sign 必须确定性，或复用同一随机种子
```

验收门槛：

```yaml
abc_require_same_scan_ids: true
abc_require_same_lref_rule: true
abc_min_cluster_map_overlap_iou: 0.80
abc_max_valid_count_delta: 3
abc_max_robust_center_error_m: 0.01
abc_max_sampled_point_rms_error_m: 0.01
abc_max_sampled_point_p95_error_m: 0.025
abc_max_count_scale_abs_error: 0.02
abc_max_score_abs_error: 0.05
abc_require_same_template_decision: true
```

回归比较项：

```text
source window scan_id 列表
window reference stamp / frame rule
EuclideanCluster 的全部 cluster 数量、点数、bbox、过滤原因
ROI selected cluster index、map AABB、overlap count、center distance
BpuPreprocessor 的 robust_center、sampled_indices、valid_ratio、count_scale
candidate_point / candidate_mask / candidate_meta / score
```

A 与 C 比较前必须先做真实候选 pairing：

```text
1. 将 A 侧 selected cluster 和 C 侧 selected cluster 都投到 map frame
2. 使用已知板标注 ROI、人工标注 board support，或同一 map candidate roi_map 关联
3. 只比较对应同一真实板实例的 cluster
```

pairing 输出：

```text
paired_board_id / manual_label
A_selected_cluster_index
C_selected_cluster_index
A_cluster_map_aabb
C_cluster_map_aabb
A_C_roi_overlap
```

若 A 与 C 的 selected cluster 点集不一致，计算双向最近点距离或 Chamfer distance，并在日志中标记偏差发生阶段：

```text
window_reconstruction
cluster_extraction
roi_cluster_selection
preprocessing
bpu_score
```

## 14. 实现顺序

第一步：

```text
冻结 no-mapping 旧 pipeline 的窗口长度、窗口锚点、步长和参考坐标系
冻结 observation_source_contract，并生成 observation_source_contract_hash
通过模型信息和 no-mapping route 冻结当前 BPU 任务语义
```

第二步：

```text
新增 ReflectiveObservationFrame.msg
dynamic_reflective_mapping_node 发布 observation_frame
```

第三步：

```text
ReflectiveCandidate.msg 增加 ROI/support/revision 字段
ReflectiveCandidates.msg 增加 candidate_snapshot_id
ReflectiveRecognitionRequest.msg 增加 request_id / candidate_snapshot_id / candidate_ids / support_revisions
RecognitionScoreEntry.msg 增加 support_revision
RecognitionScores.msg 增加 request_id / candidate_snapshot_id
reflective_candidate_extractor_node 填充新字段
cloud_sensor 在动态路线中置空或不再使用
```

第四步：

```text
新增 reflective_observation_bpu_scorer_node
维护 observation ring
订阅 candidates / recognition_request
实现 ROI -> source window lookup
实现 request/candidate/support 版本校验
实现 CandidateSnapshotLite 有界缓存
实现 RequestContext，一次 request 只发布一次汇总 RecognitionScores
```

第五步：

```text
移植旧 EuclideanCluster / ApplyCandidateFilter / PointsToBpu
实现 request 内 WindowExtractionCache
实现 max_candidates / max_candidate_window_pairs_to_prepare_per_request / max_preprocessor_attempts_per_candidate 预算
在 source window 内重提取 candidate
通过 roi_map 选择 matching cluster
```

第六步：

```text
调用现有 BpuPreprocessor 和 BpuModel
加入 max_bpu_inferences_per_request / max_bpu_inferences_per_sec 预算
加入 max_window_extractions_per_request 预算
发布 RecognitionScores
输出 debug source cloud 和日志
```

第七步：

```text
更新 dynamic_reflective_mapping.launch
更新 dynamic_reflective_mapping.yaml
更新 resource monitor target nodes
编译并运行 bag 回归
Phase 2A 使用单窗口验证旧阈值
Phase 2B 再启用 top-3 + median 并重新标定阈值
```

## 15. 禁止路径

动态建图 route 禁止以下实现作为默认识别输入：

```text
map fused points
-> inverse current lidar pose
-> cloud_sensor
-> BPU
```

禁止每个 candidate 长期复制完整 raw frame/window 点云。

禁止 bounded observation ring 跨 map_epoch 使用。

禁止 truncated frame 参与第一版 BPU source window。

禁止 source window 在聚类前按 ROI 裁剪点。

禁止不足旧窗口长度或不连续 frame 的窗口参与正常识别模式。

禁止只依赖 ROS header stamp 关联 request、candidate 和 score。

禁止把当前稀疏窗口的低分作为未验证 candidate 的负判据。

禁止同一 request 中重复 infer 同一个 candidate-window pair。

禁止在 Phase 2A 混用 K=1 和 K=3 的分数阈值。

禁止在第一版引入 map-domain BPU 模型、双 BPU 路由器或 map/raw score 融合器。
