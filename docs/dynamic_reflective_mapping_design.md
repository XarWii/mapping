# Dynamic Reflective Mapping Design

## 1. Purpose

This document defines the first dynamic mapping stage for reflective-board
recognition. The goal is not to build a dense environment map or to improve
LIO registration. The goal is to accumulate only high-reflectivity LiDAR
observations from static reflective boards while the LiDAR is mounted on a
slow-moving vehicle.

The first-stage deliverable is a bounded, short-lived reflective observation
map in a continuous local coordinate frame. BPU scoring, board recognition,
and persistent landmarks are intentionally outside the first implementation.

## 2. Agreed Constraints

- LiDAR: Livox, 10 Hz.
- IMU: 200 Hz.
- Vehicle motion: slow; reflective boards are static in the environment.
- Current low-rate pose source: Super-LIO, approximately 10 Hz.
- Future pose sources may be GNSS/INS, wheel/vehicle odometry, or another
  estimator.
- The first implementation accepts only a complete, time-stamped 6-DoF LiDAR
  pose correction. Partial GNSS or wheel observations are deferred to an
  upstream motion-fusion adapter.
- The reflective mapping pipeline must not consume a LIO point cloud, LIO
  map, or LIO internal state.
- Super-LIO is currently only a time-stamped motion-correction source.
- The 50 MB memory and half-core CPU limits apply to the reflective mapping
  and recognition pipeline. They do not include Super-LIO itself.
- Tens of milliseconds of additional latency are acceptable. Correct temporal
  alignment is more important than immediate output.

## 3. Non-Goals

The first version will not:

- Build a dense or permanent environment map.
- Perform scan-to-map matching, KNN search, point-to-plane fitting, loop
  closure, or global graph optimization.
- Reuse Super-LIO's undistorted point cloud or its voxel map.
- Run BPU inference, candidate recognition, or landmark tracking.
- Depend on a specific localization implementation.

## 4. Architectural Principle

The reflective pipeline owns its own raw LiDAR and IMU inputs. It waits for an
external pose correction for the corresponding LiDAR scan before committing
the scan to the reflective map.

```text
                         +--------------------+
Livox raw scan ----------> Super-LIO           |
IMU ---------------------> (current source)    |--> time-stamped pose correction
                         +--------------------+

Livox raw scan ----------> threshold and pending-scan cache
IMU ---------------------> lightweight ESKF and IMU history
pose correction ----------> bidirectional scan/pose time matcher
                                      |
                                      v
                       deskew high-reflectivity points
                                      |
                                      v
                   transform to continuous local frame
                                      |
                                      v
                 TTL sparse reflective voxel hash map
```

This is a delayed-commit design. The reflective mapper begins inexpensive work
when a scan arrives, but it does not insert observations into the map until the
pose correction for that scan is available.

## 5. Source Independence

### 5.1 First-stage pose contract

The map layer must only depend on a generic, time-stamped pose correction. It
must not name or include a Super-LIO type. To keep the first implementation
well-defined, this contract requires a complete 6-DoF pose, not a partial
measurement.

The internal motion-state semantic is:

```text
T_reflective_odom_imu(t): IMU pose in a continuous local frame at time t
```

The first-stage correction payload is:

```text
T_reflective_odom_imu(t_ref), reference timestamp, frame epoch,
valid/reset flag, optional covariance
```

The fixed extrinsic is explicitly named `T_imu_lidar`. LiDAR pose is always
derived inside the motion component:

```text
T_reflective_odom_lidar(t) =
T_reflective_odom_imu(t) * T_imu_lidar
```

An adapter is responsible for translating each localization source into this
semantic:

- Super-LIO adapter: consumes the LIO odometry output.
- GNSS/INS adapter: consumes a fused navigation solution with a complete pose.
- Wheel-odometry adapter: consumes an upstream fusion solution that combines
  wheel constraints with IMU propagation into a complete pose.

GNSS-only position, wheel speed, LiDAR pose, and other source-specific
measurements must first enter an adapter or motion-fusion layer. That layer
converts every source to the complete IMU pose contract above. The reflective
mapper never interprets partial observations or performs LiDAR-to-IMU pose
conversion during a measurement update.

### 5.2 Local frame policy

The map frame must be continuous. It is named `reflective_odom` in this design,
even if the current source calls its frame `world`.

Global GNSS/map jumps, estimator restarts, or explicit relocalization must
produce a reset/re-anchor event. The first version clears the rolling
reflective map on this event. The adapter must also advance the epoch when it
observes an implausible pose discontinuity without an explicit reset, using
bounded position and rotation jump thresholds derived from the maximum expected
vehicle motion. This prevents old and newly corrected observations from being
mixed in one TTL window.

## 6. Input Data Contracts

### 6.1 LiDAR scan

Each input scan must preserve:

- Scan header timestamp.
- Per-point `x`, `y`, `z`.
- Per-point reflectivity.
- Per-point relative acquisition time (`offset_time`).
- The interpretation of `offset_time` and its units.

The threshold-filtered intermediate representation must not discard
`offset_time`. A plain `xyz + intensity` PointCloud2 cannot be deskewed after
the fact. The driver adapter must define the LiDAR time base `t_base`
explicitly; the mapper must not infer that it equals `header.stamp`.

```text
t_lidar(point) = t_base + offset_time
t_end_lidar = max(valid t_lidar(point))
```

`t_end_lidar` is computed from the complete raw scan before reflectivity
filtering, sensor-frame deduplication, or any point-budget reduction. It is
the scan interval end and provides a stable scan ordering/TTL observation time.
It does not require an external odometry source to publish a pose at this exact
time; every external correction retains its own true reference time `t_ref`.

All MotionCompensator queries use IMU-time timestamps only. With calibrated
LiDAR-to-IMU offset `delta_t_lidar_to_imu`:

```text
t_imu(point) = t_lidar(point) + delta_t_lidar_to_imu
t_end_imu = t_end_lidar + delta_t_lidar_to_imu
```

The mapper must log `t_base`, `header.stamp`, minimum/maximum point time, and
scan duration during dynamic validation.

For Livox CustomMsg, convert `offset_time` to seconds exactly once at the
input boundary. The pending scan stores the scan start time and the relative
time for every retained point, plus the complete raw scan reference end time.

### 6.1.1 Reflectivity-filter boundary

The current reusable filter consumes and publishes `livox_ros_driver2/CustomMsg`.
For every retained point it copies the complete `CustomPoint` record, so
`offset_time`, `tag`, `line`, coordinates, and reflectivity remain intact. It
also preserves `header`, `timebase`, and LiDAR metadata; only `points` and
`point_num` change. It deliberately does not convert to `PointCloud2`.

An exact threshold filter has an unavoidable `Theta(N)` cost for a scan of
`N` raw points: every reflectivity byte must be observed before the decision
for that point can be made. The implemented form is therefore optimal for this
operation: one sequential read, one `reflectivity >= threshold` branch, and
one compact write only for retained points. It uses no PCL, spatial search,
coordinate transform, or temporary point mask.

The standalone filter node publishes `/reflective/high_reflect_points` for
inspection and integration testing with a subscriber queue depth of one. In the
final resource-constrained mapper, the same `ReflectivityFilter` component is
called in-process before the bounded pending-scan cache; the separate topic
node is then not required and no extra ROS process or message serialization is
paid. This boundary performs reflectivity selection only. Range, invalid-tag,
and scan-budget policies remain explicit mapper-stage decisions.

For temporary RViz observation, the standalone node can additionally publish
`/reflective/high_reflect_points_visualization` as `PointCloud2` with
`x/y/z/intensity`. This message is display-only and is not a valid deskew or
mapping input because it intentionally omits `offset_time`. Disable
`publish_visualization` for resource measurements and for the final mapper
process.

### 6.2 IMU stream

Each IMU sample must contain:

- Timestamp in the IMU time domain used by all MotionCompensator queries.
- Angular velocity in `rad/s`.
- Linear acceleration in `m/s^2` inside the MotionCompensator.

All internal IMU acceleration, gravity, acceleration bias, and acceleration
noise parameters use SI units. The current Livox ROS driver forwards its raw
accelerometer values in `g` despite placing them in `sensor_msgs/Imu`. The
input adapter therefore applies exactly one conversion at the boundary:

```text
acceleration_mps2 = acceleration_g * gravity_mps2
```

For the current Livox setup, configure `imu_acceleration_input_unit: g` and
`gravity_mps2: 9.7946`. A future IMU adapter that already publishes `m/s^2`
must use `imu_acceleration_input_unit: mps2`; it must not apply a second scale.

The cache must retain samples covering a pending scan's converted IMU-time
interval plus at least one bracketing sample after its end. A small fixed time
history is sufficient; 2-3 seconds is ample for the expected LIO delay and
scan duration.

All arbitrary-time state queries use one shared continuous-time rule: linearly
interpolate adjacent IMU measurements, then apply the same midpoint integration
used by normal propagation to split an IMU interval at the requested time. This
rule is used identically for point-time deskew queries, full-scan-end queries,
external-reference-time queries, external-correction update, and delayed-update
replay. No operation may snap a reference time to the nearest IMU sample.

### 6.3 Motion correction

Each correction must have:

- Reference timestamp in the IMU time domain.
- Complete 6-DoF IMU pose in `reflective_odom`.
- Frame epoch and validity/reset status.
- Optional covariance.

The pose timestamp is the correction's true physical reference time `t_ref` in
the IMU time domain. The mapper never relabels it as scan start or scan end.
For the current Super-LIO adapter, the correction is paired with the earliest
unresolved LiDAR scan whose complete scan interval contains `t_ref`; the pose
is then used at its actual time. Other localization sources provide their own
adapter-specific scan association, but always preserve the true `t_ref`.

`epsilon_queue` may be used only to order and locate records at queue
boundaries. It never changes the geometric reference time of a pose or point.
If an adapter cannot associate a correction with an unresolved scan, that
correction is discarded as stale or unassociated rather than silently retimed.

Before an external correction is accepted, apply two distinct gates. First,
validate that the external source is physically continuous relative to its own
previous accepted output:

```text
position_delta_ext <= v_max * delta_t + position_margin
rotation_delta_ext <= w_max * delta_t + rotation_margin
```

An explicitly reset correction starts a new epoch. An unflagged correction that
fails the external-continuity gate is marked invalid and does not write the map.

Second, compare an externally continuous correction against the ESKF prediction.
Large ESKF innovation alone does not prove that the external pose is bad: IMU
drift may be the cause. In that case accept the external correction using
conservative measurement noise, or reinitialize the ESKF after repeated large
innovations. This prevents the feedback loop in which a drifting ESKF rejects
the more trustworthy external pose.

## 7. Per-Scan Processing Sequence

### 7.1 On LiDAR scan arrival

1. Validate the scan timestamp and per-point time offsets.
2. Apply the reflectivity threshold while retaining complete point records.
3. Retain only high-reflectivity points, including their `offset_time`.
4. Store the filtered scan in a bounded pending-scan queue.
5. Attempt a timestamp match against the pending pose queue.
6. Do not update the reflective map until a valid match exists.

If the pending-scan queue is full, discard the oldest unresolved scan and
increment `pending_overflow_drop`. A scan exceeding its high-reflectivity point
budget is first reduced by a deterministic, fixed-capacity sensor-frame voxel
deduplication. Each pending-scan voxel representative must be one real raw
point record, for example the strongest point or the point nearest the local
voxel center, and must retain its original `x`, `y`, `z`, reflectivity, and
`offset_time`. Raw points at different times must not be averaged before
deskew. The policy must not retain merely the first N points, because Livox
point order is correlated with acquisition time and field of view.

The original full scan is still independently consumed by Super-LIO for its
own localization. The reflective pipeline does not need to copy, inspect, or
wait for a LIO point cloud.

### 7.2 On IMU arrival

1. Append to a fixed-duration IMU ring buffer.
2. Propagate a lightweight error-state Kalman filter (ESKF) state:
   `R, p, v, bg, ba, g`.
3. Store a bounded predicted state/IMU history for scan-time interpolation.

The motion component is not a second LIO. It has no environment map and no
point-cloud registration. It provides short-interval relative motion for
deskew and uses complete external pose corrections to update position,
orientation, and, through state covariance coupling, velocity and observable
IMU error states. A pose observation directly measures position and attitude;
it does not directly measure velocity or IMU bias.

Because Super-LIO and this component consume the same IMU, the external pose
and propagated state are not statistically independent. The ESKF covariance is
therefore not a navigation-grade independent uncertainty estimate. Use
conservative fixed external-pose measurement noise and do not make safety or
high-confidence decisions from this covariance.

### 7.3 On external pose arrival

1. Store the correction in a bounded pending pose queue.
2. Attempt a timestamp match against the pending scan queue.

Both queues perform this match on insertion, so a pose that arrives before its
scan and a scan that arrives before its pose are handled identically. If the
pending pose queue is full, discard the oldest unmatched correction and record
`pending_pose_overflow_drop`; it cannot be applied out of timestamp order.

### 7.4 Initialization and reset

The motion component has the following state machine:

```text
UNINITIALIZED
  -> WAIT_FOR_VALID_POSE_AND_IMU
  -> INITIALIZED
  -> RUNNING
  -> RESETTING
```

At first initialization or after an epoch change, no scan is committed until a
valid external IMU pose and sufficient IMU history are available. The preferred
first-stage procedure is a 1-2 second stationary initialization to estimate
gravity and IMU biases, with zero initial velocity. If stationary start is not
available, initialize velocity from two valid external poses and mark the
initialization quality as degraded. A reset clears the map and pending queues,
then returns to `WAIT_FOR_VALID_POSE_AND_IMU`.

### 7.5 Ordered commit watermark

Map commits and ESKF corrections are processed strictly in increasing scan
reference timestamp order. The mapper maintains `next_commit_time` for the
oldest unresolved scan. A later scan with an already available pose remains in
the queue until every earlier scan has either committed or timed out.

This avoids out-of-sequence ESKF corrections. A correction for an old scan is
never applied after the state has committed a newer scan. When the oldest scan
reaches any terminal state, the watermark advances. Every scan reaches exactly
one terminal state:

```text
COMMITTED
TIMED_OUT
DROPPED
```

`DROPPED` includes pending-queue overflow, pose epoch mismatch, external
continuity-gate rejection, missing IMU bracket, invalid point time, and failure
of adapter scan association. A pose arriving for an already terminal scan is a
stale correction and is discarded. This rule prevents one unusable oldest scan
from blocking all later commits.

### 7.6 On a matched scan and pose

1. Verify that the IMU cache brackets the full scan interval.
2. Use the predicted IMU trajectory, whose initial state was corrected by
   earlier pose updates, to reconstruct the relative LiDAR pose at every point
   time and at the correction's true `t_ref`.
3. Deskew every high-reflectivity point into the LiDAR frame at `t_ref` using
   the predicted relative motion.
4. Use the just-arrived external pose at its true `t_ref` to place the
   deskewed points in `reflective_odom` and insert them into the voxel map.
5. Restore or split-integrate the predicted ESKF state exactly at the correction
   reference time, apply the external correction there, then replay buffered
   IMU samples from that reference time to the current IMU head time.
6. Replace the current ESKF state with the replayed state so later scans begin
   from the corrected trajectory.
7. Remove the committed scan and its matched pose from their queues, then
   advance the commit watermark.

This first stage intentionally uses route A: the external correction at `t_ref`
corrects the current scan's absolute placement in the map, while its within-scan
deskew uses the previously corrected IMU trajectory. A common rigid correction
at `t_ref` cannot change within-scan relative motion. True fixed-lag smoothing
or backward retrodiction is deferred until dynamic validation proves that the
100 ms predicted relative motion is insufficient.

The replay in steps 5-6 is delayed-update replay, not fixed-lag smoothing: it
is mandatory to repair the future ESKF state after a correction arrived later
than its reference time. It does not retrospectively change the just-committed
scan's route-A relative deskew.

If a correction is missing beyond a bounded timeout, discard that pending scan
and increment a diagnostic counter. Never insert it using an arbitrary latest
pose. A reset, epoch mismatch, missing IMU bracket, or invalid timestamp also
causes the scan to be dropped rather than approximately matched.

### 7.7 Concurrency model

LiDAR, IMU, and external-pose callbacks only append timestamped messages to
fixed-capacity ingress queues. One serial worker consumes those queues in
timestamp order and is the sole owner of IMU propagation, delayed-update replay,
ESKF correction, commit watermark, voxel-map mutation, and time-wheel advance.
This single-writer rule is required for deterministic replay and prevents
callbacks from applying a late correction concurrently with a newer map commit.

## 8. Deskew Convention

Let `T_pred_local_lidar(t)` be the LiDAR pose reconstructed from the predicted
IMU trajectory at any IMU-time `t`. Let `T_ext_local_imu(t_ref)` be the
external IMU pose correction at its true reference time, and define:

```text
T_ext_local_lidar(t_ref) = T_ext_local_imu(t_ref) * T_imu_lidar
```

A raw LiDAR point `p_l(t_point)` is represented in the LiDAR frame at the
external correction reference time as:

```text
p_l_ref = inverse(T_pred_local_lidar(t_ref))
          * T_pred_local_lidar(t_point) * p_l(t_point)
```

For insertion into the local reflective map:

```text
p_local = T_ext_local_lidar(t_ref) * p_l_ref
```

The predicted relative motion creates the scan shape; the external pose at its
actual `t_ref` controls absolute local placement. Direct insertion using a
predicted absolute point pose is not equivalent in route A and is therefore not
used. The explicit reference-frame representation is retained because it is
easier to inspect, publish for debugging, and later use as a stable candidate
input.

LiDAR-to-IMU extrinsics and any IMU-to-body transform must be explicit
configuration parameters with documented direction. The implementation must
not rely on global mutable transforms or implicit frame naming.

The LiDAR-to-IMU time offset and extrinsics are required preconditions for the
first dynamic test, not post-hoc tuning items. An unverified offset can look
like poor map fusion even when the hash map is correct.

## 9. Rolling Reflective Voxel Map

### 9.1 Map semantics

The map is not a permanent world map. It is a recent-observation accumulator:

```text
recent high-reflectivity observations in reflective_odom
```

Only threshold-passing observations are inserted. Low-reflectivity background
points are never stored.

### 9.2 Storage

Use a fixed-capacity, open-addressing sparse voxel index table plus a separate
fixed voxel-node pool. The hash table stores `key -> stable node_id`; the node
pool owns the actual ReflectVoxel and time-wheel links. Do not use
`std::unordered_map`, iKD-Tree, OctVox/HKNN, or a dynamically growing global
point cloud.

Each occupied voxel stores one fused representative and minimal evidence:

```cpp
struct ReflectVoxel {
  float x, y, z;          // fused representative in reflective_odom
  uint64_t last_seen_tick;     // monotonic sensor-time tick
  uint64_t last_evidence_scan_id;
  uint16_t fuse_count;    // capped incremental mean count
  uint16_t evidence;      // capped count of distinct scans
  uint8_t intensity_diag; // diagnostic only in stage one
  uint8_t flags;
  uint64_t expiry_tick;   // sensor-time expiration tick
  uint32_t wheel_prev, wheel_next;
  uint32_t generation;
};
```

The index table contains a complete key or a collision-verifiable key
fingerprint, a stable node ID, control metadata, and padding. The node pool has
a fixed free list. Backward-shift deletion may move index-table entries but
never moves a node-pool object, so time-wheel links remain valid. The final
layout must document index-table bytes, node-pool bytes, safe table load factor,
and wheel storage; payload bytes alone are not a memory estimate.

### 9.3 Insertion

For a point `p_local` and voxel size `r`:

```text
key = floor(p_local / r)
```

Before fusion, deduplicate each scan locally: each voxel accepts at most one
representative observation from a scan. The representative may be the local
mean, strongest point, or point nearest the voxel center, but the policy must
be deterministic.

For an existing key, first apply a merge gate:

```text
norm(p_local - voxel.mean) <= merge_distance
```

Parameter sweeps must constrain this gate to the voxel geometry:

```text
0 < merge_distance <= sqrt(3) * voxel_size
merge_distance = alpha * voxel_size, alpha in [0.5, 1.0], initially
```

`merge_distance` must be evaluated together with voxel size and measured
short-term registration error. The mapper records merge-reject rate by range
and motion condition.

Only a point that passes this gate may update the capped incremental mean. A
point that fails is rejected in stage one; it must not pull one representative
toward unrelated board/background geometry. Once `fuse_count` reaches its cap,
the representative position is frozen. A passing observation may still refresh
recency and per-scan evidence.

Evidence increments at most once per scan:

```text
if scan_id != last_evidence_scan_id:
    evidence = min(evidence + 1, evidence_cap)
    last_evidence_scan_id = scan_id
```

This represents repeated temporal observations, rather than LiDAR point density
within a single scan. The intensity field is diagnostic only in stage one; it
must not be treated as a recent peak if it is implemented as a lifetime maximum.
If the key is new, allocate a stable node from the fixed pool and insert its ID
into the fixed index table.

In stage one, `evidence` is an episode/lifetime count capped while the node
continues to be refreshed before TTL expiry. It is not a strict sliding-window
observation-density estimate and must not alone decide whether a future
candidate is stable. Candidate discovery will additionally use a short-time
observation bitmask or equivalent time-span/density measure.

The map uses TTL as its primary eviction rule. Its time base is the committed
scan reference timestamp, never callback arrival time or CPU wall time:

```text
last_seen_tick = scan_end_time_tick
expire_tick = scan_end_time_tick + T_keep_ticks
```

TTL expiry is driven by an intrusive, fixed-capacity time wheel. Every occupied
node is linked into exactly one expiry bucket and contains its current bucket
links, expiry tick, and generation. On refresh, unlink the node from its old
bucket and relink it into the new bucket. The wheel therefore has O(node_count)
storage rather than one appended expiry event per refresh.

For `bucket_count` buckets of width `wheel_tick`, require:

```text
bucket_count * wheel_tick > T_keep_ticks + max_watermark_gap_ticks + wheel_tick
```

Processing a bucket never expires a node solely because its bucket index became
current. Expire only when `node.expiry_tick <= watermark_tick`; a node sharing
the bucket through modulo arithmetic but with a later exact expiry stays linked
or is relinked. Capacity eviction likewise compares true expiry ticks rather
than bucket indices.

The wheel advances with the monotonic commit watermark, not merely successful
map inserts. A scan timeout also advances the watermark. If the next watermark
is more than `T_keep` ahead of the previous one, clear the map directly instead
of iterating a large number of expired buckets.

An open-addressing index entry in the middle of a probe chain must never be
cleared naively. The implementation uses backward-shift deletion for the index
table, then returns only the stable node-pool object to the free list. This
preserves lookup correctness without invalidating time-wheel links.

Hard capacity is a secondary safeguard. When full, evict an expired voxel
first; otherwise use a bounded deterministic policy such as the oldest time
wheel bucket or a clock hand. It must not scan the whole table to find a global
oldest/weakest entry. LRU is not the primary semantic because observation
recency, not query recency, is what matters.

### 9.4 Initial parameter sweep

These are experiment starting points, not frozen constants:

- LiDAR rate: 10 Hz.
- Time-to-live: 1.0 s initially; evaluate 1.0-3.0 s.
- Voxel size: evaluate 0.03, 0.05, 0.08, and 0.10 m.
- Table capacity: start at 32k-64k slots, then choose from dynamic-bag
  occupancy percentiles.
- Candidate count: not implemented in stage one; reserve at most eight in the
  future design.

The final voxel size must balance registration/deskew noise against the
smallest board structure that must remain visible.

## 10. Resource Design

### 10.1 Memory

The mapper must use bounded containers and fixed capacities for all hot-path
data:

- Pending scans: fixed count and fixed maximum retained high-reflection points
  per scan.
- IMU history: time-bounded ring buffer.
- Voxel map: fixed open-addressing index capacity and fixed stable node pool.
- Expiry structure: intrusive fixed-capacity time wheel, one linked entry per
  occupied node.
- Debug output: disabled by default and never allowed to retain historical
  full clouds.
- ROS subscriber queues: depth one unless a bounded synchronization queue is
  explicitly required.

The map budget must be calculated from full slots, not just voxel payload:

```text
map_bytes = index_slot_count * (key + node_id + control + padding)
            + node_capacity * (ReflectVoxel + free_list_metadata)
            + wheel_bucket_bytes
```

Safe occupied voxel count is lower than slot count because open-addressing
tables need headroom for bounded probe length. Candidate micro-maps, when
introduced, are negligible by comparison. The BPU runtime RSS must be measured
separately before assigning the final overall 50 MB budget.

The stage-one mapper allocation target is 10-15 MB peak, leaving room for ROS
runtime overhead, future candidate buffers, and BPU runtime. The initial fixed
budget is deliberately conservative and must be checked against actual struct
sizes at build time:

| Component | Initial fixed capacity | Budget |
| --- | ---: | ---: |
| Hash index | 65,536 entries at <= 24 B | <= 1.5 MB |
| Stable voxel node pool | 32,768 nodes at <= 64 B | <= 2.0 MB |
| Time-wheel buckets | 128 buckets | <= 0.01 MB |
| Pending high-reflection scans | 4 x 8,192 raw records at <= 24 B | <= 0.8 MB |
| LiDAR/pose ingress queues and message headroom | bounded depth | <= 2.0 MB |
| Sensor/map-frame temporary dedup tables | reused fixed buffers | <= 1.0 MB |
| IMU samples, predicted-state history, and replay scratch | 3 s history | <= 0.5 MB |
| Worker scratch and diagnostics | fixed buffers | <= 1.0 MB |
| Mapper-owned subtotal |  | <= 8.9 MB |

The remaining margin up to the 10-15 MB stage-one target covers allocator
overhead, alignment, and conservative growth. The measured process RSS remains
the acceptance metric; this table is not a substitute for profiling.

### 10.2 CPU

The hot path must be linear in the number of threshold-passing points:

- Threshold once.
- Deskew each retained point once.
- Perform one hash insertion/update per retained point.
- Do not run global Euclidean clustering over all accumulated observations.
- Do not invoke PCL or TBB for the reflective map hot path.

Future candidate discovery must inspect bounded neighborhoods around recently
touched voxels rather than scan the full map every frame.

## 11. Outputs and Diagnostics for Stage One

Production mode publishes only compact statistics and diagnostics. Point-cloud
debug output is disabled by default, generated only when explicitly requested,
and rate-limited to at most 1 Hz:

- Map statistics: occupied slots, insertions, updates, TTL expirations,
  capacity evictions, pending scan/pose count, dropped scans, merge rejects,
  external-continuity rejects, large ESKF innovations, delayed-update replays,
  epoch changes, initialization state, commit watermark, and timing mismatch
  count.
- Per-scan timing: LiDAR stamp, matched pose stamp, timestamp delta, waiting
  latency, deskew duration, and map insertion duration.
- Optional deskewed high-reflectivity cloud in the external pose-reference
  LiDAR coordinates.
- Optional rolling voxel-map cloud in `reflective_odom`.
- Optional raw-versus-deskewed comparison cloud.

No BPU scores, recognition decisions, target commands, or landmarks are part
of this stage.

## 12. Validation Plan

### 12.1 Required dynamic bag

Record a bag with:

- Raw Livox point cloud, including point offsets.
- Raw IMU.
- Super-LIO odometry output.
- A static reflective board while the vehicle moves.

Before evaluating map quality, measure and record LiDAR-to-IMU time alignment,
LiDAR-to-IMU extrinsics, and the exact reference-time semantics of the
Super-LIO pose output.

### 12.2 Functional checks

1. The current pose adapter's timestamp semantics are verified before the
   dynamic test begins.
2. Full-scan reference end time is computed before any reflectivity filtering
   or point-budget reduction.
3. Every committed scan has a matched pose, epoch, and IMU coverage.
4. Map commits and ESKF corrections occur in strictly increasing scan-reference
   timestamp order.
5. Every LiDAR point time is converted to IMU time using the configured offset.
6. A delayed correction restores, updates, and replays the ESKF to the current
   IMU head before any later scan is committed.
7. A stale, missing, or externally discontinuous correction never inserts a
   scan; large ESKF innovation alone does not automatically reject it.
8. Every scan reaches exactly one terminal state and advances the watermark.
9. Explicit resets and soft pose discontinuities clear the rolling map.
10. TTL expiry preserves hash lookup correctness after deletions and exact
    expiry-tick checks.
11. Evidence increments no more than once per voxel per scan.
12. Initialization and queue-overflow states never commit an unqualified scan.
13. Memory remains bounded under long-duration replay.

### 12.3 Quality checks

Compare raw accumulation, route-A pose-reference deskew, and route-A local-map
accumulation. For a static board measure:

- Cluster center jitter.
- Thickness along the board normal.
- Long and short spatial extents.
- Number of occupied voxels.
- Stability over roughly one second of observations.

The deskewed local accumulation should reduce smear and make the board cluster
more stable than raw sensor-frame accumulation. Upgrade to fixed-lag smoothing
or backward retrodiction only if route A fails these quality checks.

### 12.4 Resource checks

Measure the complete reflective pipeline, excluding Super-LIO:

- Process RSS and peak RSS.
- CPU usage over stationary, normal-motion, and high-reflection-background
  bags.
- Per-stage latency and pending-queue depth.
- BPU runtime memory baseline before BPU functionality is enabled.

## 13. Integration With Current Codebase

The current `siamese_bpu_infer_node` is a stationary-sensor pipeline: it
filters high-reflectivity points, retains several frames directly in
`livox_frame`, flattens them, and applies global quadratic Euclidean
clustering. It is not the stage-one implementation target.

The new mapper is implemented as an independent core with three logical
components:

```text
MotionCompensator
ReflectiveVoxelMap
ReflectiveMappingNode (ROS adapters, synchronization, diagnostics)
```

They may be deployed in one ROS process to avoid inter-process full-cloud
serialization and duplicated buffers. Super-LIO integration is restricted to
an odometry adapter. This keeps future localization-source replacement local
to that adapter.

The current first-stage executable is `dynamic_reflective_mapping_node`. It
subscribes to raw Livox `CustomMsg`, `sensor_msgs/Imu`, and generic
`nav_msgs/Odometry`; the current Super-LIO adapter is selected entirely by
parameters (`pose_topic: /lio/odom`, `pose_input_frame: lidar`). The mapper
does not subscribe to a Super-LIO cloud or map. Its optional
`/reflective/rolling_map` PointCloud2 is a rate-limited debug output only.

### 13.1 Current Super-LIO adapter

Super-LIO is an external pose provider and is not modified by this project. The
current low-rate `/lio/odom` output is a LiDAR pose at its true header timestamp
`t_ref`; it is not treated as an IMU pose. The adapter preserves that timestamp,
converts the pose with the configured fixed extrinsic, and then applies the
generic IMU-pose contract:

```text
T_reflective_odom_imu(t_ref) =
T_reflective_odom_lidar(t_ref) * inverse(T_imu_lidar)
```

The correction is associated with an unresolved raw scan only when `t_ref`
falls in that scan's IMU-time interval. The higher-rate `/lio/robo/odom` is not
used as the low-rate external correction source in this stage.

The reflective mapper then uses the route-A transform in Section 8 to express
every retained point relative to `t_ref`; it does not relabel the Super-LIO pose
as scan-end pose or require Super-LIO to expose its internal map, point cloud,
or IMU state. The Livox `t_base` assumption and the observed relationship
between Super-LIO pose stamps and raw scan intervals remain required bag checks.

## 14. Deferred Work

After stage one passes validation:

1. Add bounded local voxel-connected candidate discovery.
2. Add per-candidate fixed-size micro-maps.
3. Re-express each candidate in the current LiDAR frame before BPU input.
4. Validate the existing BPU model and decision threshold on dynamic,
   deskewed, multi-frame candidate data.
5. Add confirmed-board landmarks with explicit reset/re-anchor behavior.

## 15. Items To Measure Before Freezing Parameters

These must be measured from dynamic bags before freezing parameters or enabling
the next stage:

- Livox driver `t_base` semantics and scan point-time range.
- LiDAR-to-IMU clock offset sign and value.
- External pose arrival delay and delayed-update replay duration.
- High-reflectivity point count per scan and one-to-three-second voxel
  occupancy percentile.
- Maximum board range, minimum board dimensions, and expected simultaneous
  board count.
- Dynamic BPU score distribution after candidate micro-map integration.
