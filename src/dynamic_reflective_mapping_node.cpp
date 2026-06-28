#include <Eigen/Dense>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <livox_reflective_marker/reflectivity_filter.h>
#include <livox_reflective_marker/ReflectiveMapSnapshot.h>
#include <livox_reflective_marker/ReflectiveObservationFrame.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();

bool IsFinite(double value) { return std::isfinite(value); }

bool IsFinite(const Eigen::Vector3d& value) {
  return value.allFinite();
}

Eigen::Matrix3d Skew(const Eigen::Vector3d& value) {
  Eigen::Matrix3d result;
  result << 0.0, -value.z(), value.y(),
      value.z(), 0.0, -value.x(),
      -value.y(), value.x(), 0.0;
  return result;
}

Eigen::Quaterniond ExpQuaternion(const Eigen::Vector3d& rotation) {
  const double angle = rotation.norm();
  if (angle < 1e-12) {
    return Eigen::Quaterniond(1.0, 0.5 * rotation.x(), 0.5 * rotation.y(),
                              0.5 * rotation.z()).normalized();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation / angle));
}

Eigen::Vector3d QuaternionLog(const Eigen::Quaterniond& quaternion) {
  Eigen::Quaterniond q = quaternion.normalized();
  if (q.w() < 0.0) q.coeffs() *= -1.0;
  const Eigen::Vector3d imag(q.x(), q.y(), q.z());
  const double imag_norm = imag.norm();
  if (imag_norm < 1e-12) return 2.0 * imag;
  return 2.0 * std::atan2(imag_norm, q.w()) * imag / imag_norm;
}

double RotationDistance(const Eigen::Quaterniond& first,
                        const Eigen::Quaterniond& second) {
  return QuaternionLog(first.conjugate() * second).norm();
}

size_t Tag30Bin(uint8_t tag) {
  return static_cast<size_t>((tag & 0x30) >> 4);
}

struct ImuSample {
  double time = 0.0;
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();
};

struct NominalState {
  double time = 0.0;
  Eigen::Quaterniond q_odom_imu = Eigen::Quaterniond::Identity();
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 15, 15> covariance =
      Eigen::Matrix<double, 15, 15>::Identity();
};

struct TrajectoryKnot {
  ImuSample imu;
  NominalState state;
};

struct PoseCorrection {
  double time = 0.0;
  Eigen::Quaterniond q_odom_imu = Eigen::Quaterniond::Identity();
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
};

struct OdomPoseSample {
  PoseCorrection pose;
  uint64_t segment_id = 0;
};

struct OdomCoverageResult {
  bool valid = false;
  uint64_t segment_id = 0;
  double max_gap = 0.0;
  bool has_future_odom = false;
  bool crossed_jump = false;
  const char* reason = "NO_ODOM";
};

struct OdomPoseQuery {
  bool valid = false;
  uint64_t segment_id = 0;
  double left_stamp = 0.0;
  double right_stamp = 0.0;
  double alpha = 0.0;
  const char* mode = "invalid";
  PoseCorrection pose;
};

class OdomBuffer {
 public:
  struct Options {
    size_t max_samples = 64;
    double max_velocity_mps = 5.0;
    double max_angular_velocity_rps = 2.0;
    double position_margin_m = 0.5;
    double rotation_margin_rad = 0.3;
  };

  explicit OdomBuffer(Options options) : options_(options) {}

  void Clear() {
    samples_.clear();
    segment_id_ = 0;
  }

  bool Push(const PoseCorrection& pose, std::string* segment_reason) {
    bool new_segment = false;
    const char* reason = "";
    if (!samples_.empty()) {
      const OdomPoseSample& previous = samples_.back();
      const double dt = pose.time - previous.pose.time;
      const double translation_delta = (pose.position - previous.pose.position).norm();
      const double rotation_delta =
          RotationDistance(previous.pose.q_odom_imu, pose.q_odom_imu);
      if (dt <= 0.0) {
        new_segment = true;
        reason = "TIMESTAMP_REGRESSION";
      } else if (translation_delta >
                 options_.max_velocity_mps * dt + options_.position_margin_m) {
        new_segment = true;
        reason = "TRANSLATION_JUMP";
      } else if (rotation_delta >
                 options_.max_angular_velocity_rps * dt + options_.rotation_margin_rad) {
        new_segment = true;
        reason = "ROTATION_JUMP";
      }
    }
    if (new_segment) {
      ++segment_id_;
      if (segment_reason) *segment_reason = reason;
    }
    samples_.push_back(OdomPoseSample{pose, segment_id_});
    while (samples_.size() > options_.max_samples) samples_.pop_front();
    return new_segment;
  }

  OdomCoverageResult CheckCoverage(double begin, double end,
                                   double max_gap_sec) const {
    OdomCoverageResult result;
    if (!IsFinite(begin) || !IsFinite(end) || end < begin) {
      result.reason = "TIME_INVALID";
      return result;
    }
    if (samples_.size() < 2) {
      result.has_future_odom = samples_.empty() || samples_.back().pose.time < end;
      result.reason = result.has_future_odom ? "WAIT_FUTURE_ODOM" : "NO_BRACKET";
      return result;
    }
    if (end > samples_.back().pose.time) {
      result.has_future_odom = true;
      result.reason = "WAIT_FUTURE_ODOM";
      return result;
    }
    if (begin < samples_.front().pose.time) {
      result.reason = "NO_LEFT_BRACKET";
      return result;
    }

    const auto begin_upper = LowerBound(begin);
    const auto end_upper = LowerBound(end);
    if (begin_upper == samples_.end()) {
      result.reason = "NO_LEFT_BRACKET";
      return result;
    }
    size_t left_idx = static_cast<size_t>(
        std::distance(samples_.begin(), begin_upper));
    if (std::abs(begin_upper->pose.time - begin) >= 1e-9) {
      if (begin_upper == samples_.begin()) {
        result.reason = "NO_LEFT_BRACKET";
        return result;
      }
      --left_idx;
    }
    if (end_upper == samples_.end()) {
      result.reason = "NO_RIGHT_BRACKET";
      return result;
    }
    const size_t right_idx = static_cast<size_t>(
        std::distance(samples_.begin(), end_upper));
    if (left_idx > right_idx) {
      result.reason = "NO_LEFT_BRACKET";
      return result;
    }

    result.segment_id = samples_[left_idx].segment_id;
    for (size_t i = left_idx; i <= right_idx; ++i) {
      if (samples_[i].segment_id != result.segment_id) {
        result.crossed_jump = true;
        result.reason = "CROSSED_SEGMENT";
        return result;
      }
      if (i == right_idx) break;
      const double gap = samples_[i + 1].pose.time - samples_[i].pose.time;
      result.max_gap = std::max(result.max_gap, gap);
      if (gap > max_gap_sec) {
        result.reason = "GAP_TOO_LARGE";
        return result;
      }
    }
    result.valid = true;
    result.reason = "FULL";
    return result;
  }

  OdomPoseQuery PoseAt(double time) const {
    OdomPoseQuery query;
    if (!IsFinite(time) || samples_.size() < 2 ||
        time < samples_.front().pose.time ||
        time > samples_.back().pose.time) {
      return query;
    }
    const auto upper = LowerBound(time);
    if (upper != samples_.end() && std::abs(upper->pose.time - time) < 1e-9) {
      query.valid = true;
      query.segment_id = upper->segment_id;
      query.left_stamp = upper->pose.time;
      query.right_stamp = upper->pose.time;
      query.alpha = 0.0;
      query.mode = "exact";
      query.pose = upper->pose;
      return query;
    }
    if (upper == samples_.begin() || upper == samples_.end()) return query;
    const auto lower = std::prev(upper);
    if (lower->segment_id != upper->segment_id) return query;
    const double dt = upper->pose.time - lower->pose.time;
    if (dt <= 0.0) return query;
    const double alpha = (time - lower->pose.time) / dt;
    query.valid = true;
    query.segment_id = lower->segment_id;
    query.left_stamp = lower->pose.time;
    query.right_stamp = upper->pose.time;
    query.alpha = alpha;
    query.mode = "interpolated";
    query.pose.time = time;
    query.pose.position =
        lower->pose.position +
        alpha * (upper->pose.position - lower->pose.position);
    query.pose.q_odom_imu =
        lower->pose.q_odom_imu.slerp(alpha, upper->pose.q_odom_imu).normalized();
    return query;
  }

  size_t size() const { return samples_.size(); }
  double latest_time() const {
    return samples_.empty() ? -std::numeric_limits<double>::infinity()
                            : samples_.back().pose.time;
  }

 private:
  std::deque<OdomPoseSample>::const_iterator LowerBound(double time) const {
    return std::lower_bound(
        samples_.begin(), samples_.end(), time,
        [](const OdomPoseSample& sample, double target) {
          return sample.pose.time < target;
        });
  }

  Options options_;
  uint64_t segment_id_ = 0;
  std::deque<OdomPoseSample> samples_;
};

struct MapFrameAnchor {
  std::string parent_frame;
  double time = 0.0;
  Eigen::Quaterniond q_parent_reflective = Eigen::Quaterniond::Identity();
  Eigen::Vector3d p_parent_reflective = Eigen::Vector3d::Zero();
};

struct PendingPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint32_t offset_time = 0;
  uint8_t reflectivity = 0;
};

struct DeskewedHighPoint {
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  float intensity = 0.0f;
};

struct MapObservation {
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  Eigen::Vector3f lidar_reference = Eigen::Vector3f::Zero();
  float intensity = 0.0f;
  uint32_t count = 0;
};

struct PendingScan {
  bool active = false;
  uint64_t scan_id = 0;
  double lidar_base_time = 0.0;
  double lidar_header_time = 0.0;
  double lidar_timebase_time = 0.0;
  double imu_start_time = 0.0;
  double imu_end_time = 0.0;
  double accepted_imu_start_time = std::numeric_limits<double>::infinity();
  double accepted_imu_end_time = -std::numeric_limits<double>::infinity();
  uint64_t end_tick = 0;
  uint32_t raw_point_count = 0;
  uint32_t accepted_point_count = 0;
  uint32_t rejected_intensity_count = 0;
  uint32_t rejected_nonfinite_count = 0;
  uint32_t rejected_range_count = 0;
  uint32_t dropped_budget_count = 0;
  std::array<uint32_t, 4> raw_tag30_bins{{0, 0, 0, 0}};
  std::array<uint32_t, 4> accepted_tag30_bins{{0, 0, 0, 0}};
  bool trace_wait_logged = false;
  std::vector<PendingPoint> points;
};

struct VoxelKey {
  int32_t x = 0;
  int32_t y = 0;
  int32_t z = 0;
};

bool operator==(const VoxelKey& first, const VoxelKey& second) {
  return first.x == second.x && first.y == second.y && first.z == second.z;
}

uint64_t Mix64(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

uint64_t HashKey(const VoxelKey& key) {
  uint64_t hash = Mix64(static_cast<uint32_t>(key.x));
  hash ^= Mix64(static_cast<uint32_t>(key.y) + 0x9e3779b9U);
  hash ^= Mix64(static_cast<uint32_t>(key.z) + 0x85ebca6bU);
  return Mix64(hash);
}

struct VoxelNode {
  bool occupied = false;
  VoxelKey key;
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  uint64_t last_seen_tick = 0;
  uint64_t expiry_tick = 0;
  uint64_t last_evidence_scan_id = 0;
  uint16_t fuse_count = 0;
  uint16_t evidence = 0;
  uint8_t intensity_diag = 0;
  uint32_t wheel_prev = kInvalidIndex;
  uint32_t wheel_next = kInvalidIndex;
  uint32_t wheel_bucket = kInvalidIndex;
  uint32_t next_free = kInvalidIndex;
};

struct MapStats {
  uint64_t insertions = 0;
  uint64_t updates = 0;
  uint64_t merge_rejects = 0;
  uint64_t ttl_expirations = 0;
  uint64_t capacity_evictions = 0;
};

class ReflectiveVoxelMap {
 public:
  struct Options {
    size_t hash_capacity = 65536;
    size_t node_capacity = 32768;
    size_t wheel_bucket_count = 128;
    float voxel_size = 0.05f;
    float merge_distance = 0.05f;
    bool enable_ttl = true;
    uint64_t ttl_ticks = 20;
    uint16_t fuse_count_cap = 32;
    uint16_t evidence_cap = 255;
  };

  explicit ReflectiveVoxelMap(Options options) : options_(options) {
    slots_.assign(options_.hash_capacity, kInvalidIndex);
    nodes_.resize(options_.node_capacity);
    wheel_heads_.assign(options_.wheel_bucket_count, kInvalidIndex);
    for (uint32_t i = 0; i < nodes_.size(); ++i) {
      nodes_[i].next_free = i + 1 < nodes_.size() ? i + 1 : kInvalidIndex;
    }
    free_head_ = nodes_.empty() ? kInvalidIndex : 0;
  }

  void Advance(uint64_t watermark_tick) {
    if (!options_.enable_ttl) return;
    if (!have_watermark_) {
      watermark_tick_ = watermark_tick;
      have_watermark_ = true;
      return;
    }
    if (watermark_tick <= watermark_tick_) return;
    if (watermark_tick - watermark_tick_ > options_.ttl_ticks) {
      Clear();
      watermark_tick_ = watermark_tick;
      have_watermark_ = true;
      return;
    }
    while (watermark_tick_ < watermark_tick) {
      ++watermark_tick_;
      const size_t bucket = watermark_tick_ % wheel_heads_.size();
      uint32_t node_id = wheel_heads_[bucket];
      while (node_id != kInvalidIndex) {
        const uint32_t next = nodes_[node_id].wheel_next;
        if (nodes_[node_id].expiry_tick <= watermark_tick_) {
          EraseNode(node_id);
          ++stats_.ttl_expirations;
        }
        node_id = next;
      }
    }
  }

  void Insert(const Eigen::Vector3f& point, uint64_t scan_id,
              uint64_t observation_tick, uint8_t reflectivity) {
    const VoxelKey key = ToKey(point);
    const size_t slot = FindSlot(key);
    if (slots_[slot] != kInvalidIndex) {
      VoxelNode& node = nodes_[slots_[slot]];
      if ((node.position - point).squaredNorm() >
          options_.merge_distance * options_.merge_distance) {
        ++stats_.merge_rejects;
        return;
      }
      if (node.last_evidence_scan_id != scan_id) {
        if (node.fuse_count < options_.fuse_count_cap) {
          const float denominator = static_cast<float>(node.fuse_count + 1);
          node.position += (point - node.position) / denominator;
          ++node.fuse_count;
        }
        node.evidence = std::min<uint16_t>(
            options_.evidence_cap, static_cast<uint16_t>(node.evidence + 1));
        node.last_evidence_scan_id = scan_id;
      }
      node.intensity_diag = std::max(node.intensity_diag, reflectivity);
      Refresh(slots_[slot], observation_tick);
      ++stats_.updates;
      return;
    }

    uint32_t node_id = AllocateNode();
    if (node_id == kInvalidIndex) return;
    VoxelNode& node = nodes_[node_id];
    node.occupied = true;
    node.key = key;
    node.position = point;
    node.last_evidence_scan_id = scan_id;
    node.fuse_count = 1;
    node.evidence = 1;
    node.intensity_diag = reflectivity;
    slots_[slot] = node_id;
    Refresh(node_id, observation_tick);
    ++stats_.insertions;
  }

  void Clear() {
    std::fill(slots_.begin(), slots_.end(), kInvalidIndex);
    std::fill(wheel_heads_.begin(), wheel_heads_.end(), kInvalidIndex);
    for (uint32_t i = 0; i < nodes_.size(); ++i) {
      nodes_[i] = VoxelNode{};
      nodes_[i].next_free = i + 1 < nodes_.size() ? i + 1 : kInvalidIndex;
    }
    free_head_ = nodes_.empty() ? kInvalidIndex : 0;
    occupied_count_ = 0;
  }

  size_t occupied_count() const { return occupied_count_; }
  const std::vector<VoxelNode>& nodes() const { return nodes_; }
  const MapStats& stats() const { return stats_; }

 private:
  VoxelKey ToKey(const Eigen::Vector3f& point) const {
    return VoxelKey{static_cast<int32_t>(std::floor(point.x() / options_.voxel_size)),
                    static_cast<int32_t>(std::floor(point.y() / options_.voxel_size)),
                    static_cast<int32_t>(std::floor(point.z() / options_.voxel_size))};
  }

  size_t FindSlot(const VoxelKey& key) const {
    size_t slot = HashKey(key) % slots_.size();
    for (size_t probe = 0; probe < slots_.size(); ++probe) {
      const uint32_t node_id = slots_[slot];
      if (node_id == kInvalidIndex || nodes_[node_id].key == key) return slot;
      slot = (slot + 1) % slots_.size();
    }
    return 0;
  }

  size_t ProbeDistance(size_t slot, size_t home) const {
    return (slot + slots_.size() - home) % slots_.size();
  }

  void BackshiftDelete(size_t hole) {
    size_t next = (hole + 1) % slots_.size();
    while (slots_[next] != kInvalidIndex) {
      const size_t home = HashKey(nodes_[slots_[next]].key) % slots_.size();
      if (ProbeDistance(next, home) >= ProbeDistance(next, hole)) {
        slots_[hole] = slots_[next];
        hole = next;
      }
      next = (next + 1) % slots_.size();
    }
    slots_[hole] = kInvalidIndex;
  }

  uint32_t AllocateNode() {
    if (free_head_ == kInvalidIndex && !EvictForCapacity()) return kInvalidIndex;
    const uint32_t node_id = free_head_;
    free_head_ = nodes_[node_id].next_free;
    nodes_[node_id] = VoxelNode{};
    ++occupied_count_;
    return node_id;
  }

  bool EvictForCapacity() {
    if (!options_.enable_ttl) {
      uint32_t oldest = kInvalidIndex;
      uint64_t oldest_tick = std::numeric_limits<uint64_t>::max();
      for (uint32_t node_id = 0; node_id < nodes_.size(); ++node_id) {
        const VoxelNode& node = nodes_[node_id];
        if (node.occupied && node.last_seen_tick < oldest_tick) {
          oldest = node_id;
          oldest_tick = node.last_seen_tick;
        }
      }
      if (oldest == kInvalidIndex) return false;
      EraseNode(oldest);
      ++stats_.capacity_evictions;
      return true;
    }
    for (size_t count = 0; count < wheel_heads_.size(); ++count) {
      const size_t bucket = eviction_bucket_++ % wheel_heads_.size();
      if (wheel_heads_[bucket] != kInvalidIndex) {
        EraseNode(wheel_heads_[bucket]);
        ++stats_.capacity_evictions;
        return true;
      }
    }
    return false;
  }

  void Refresh(uint32_t node_id, uint64_t observation_tick) {
    VoxelNode& node = nodes_[node_id];
    if (node.wheel_bucket != kInvalidIndex) UnlinkWheel(node_id);
    node.last_seen_tick = observation_tick;
    if (!options_.enable_ttl) {
      node.expiry_tick = 0;
      return;
    }
    node.expiry_tick = observation_tick + options_.ttl_ticks;
    node.wheel_bucket = node.expiry_tick % wheel_heads_.size();
    node.wheel_prev = kInvalidIndex;
    node.wheel_next = wheel_heads_[node.wheel_bucket];
    if (node.wheel_next != kInvalidIndex) nodes_[node.wheel_next].wheel_prev = node_id;
    wheel_heads_[node.wheel_bucket] = node_id;
  }

  void UnlinkWheel(uint32_t node_id) {
    VoxelNode& node = nodes_[node_id];
    if (node.wheel_prev != kInvalidIndex) {
      nodes_[node.wheel_prev].wheel_next = node.wheel_next;
    } else {
      wheel_heads_[node.wheel_bucket] = node.wheel_next;
    }
    if (node.wheel_next != kInvalidIndex) nodes_[node.wheel_next].wheel_prev = node.wheel_prev;
    node.wheel_prev = kInvalidIndex;
    node.wheel_next = kInvalidIndex;
    node.wheel_bucket = kInvalidIndex;
  }

  void EraseNode(uint32_t node_id) {
    VoxelNode& node = nodes_[node_id];
    const size_t slot = FindSlot(node.key);
    if (slots_[slot] == node_id) BackshiftDelete(slot);
    if (node.wheel_bucket != kInvalidIndex) UnlinkWheel(node_id);
    node = VoxelNode{};
    node.next_free = free_head_;
    free_head_ = node_id;
    --occupied_count_;
  }

  Options options_;
  std::vector<uint32_t> slots_;
  std::vector<VoxelNode> nodes_;
  std::vector<uint32_t> wheel_heads_;
  uint32_t free_head_ = kInvalidIndex;
  size_t occupied_count_ = 0;
  size_t eviction_bucket_ = 0;
  uint64_t watermark_tick_ = 0;
  bool have_watermark_ = false;
  MapStats stats_;
};

struct MapElementUpdateStats {
  uint32_t observations = 0;
  uint32_t created = 0;
  uint32_t fused = 0;
  uint32_t matured = 0;
  uint32_t dropped_capacity = 0;
  uint32_t mature_count = 0;
  uint32_t candidate_count = 0;
  double mature_rms_p50 = 0.0;
  double mature_rms_p95 = 0.0;
  double mature_rms_max = 0.0;
};

class MapElementFilter {
 public:
  struct Options {
    bool enabled = false;
    float association_radius = 0.03f;
    uint32_t mature_min_support_scans = 3;
    float mature_max_rms = 0.025f;
    float weight_cap = 8.0f;
    size_t max_elements = 32768;
    bool publish_mature_only = true;
  };

  explicit MapElementFilter(Options options) : options_(options) {}

  void Clear() {
    elements_.clear();
    latest_stats_ = MapElementUpdateStats{};
  }

  MapElementUpdateStats UpdateScan(const std::vector<MapObservation>& observations,
                                   uint64_t scan_id) {
    MapElementUpdateStats stats;
    stats.observations = static_cast<uint32_t>(observations.size());
    const size_t initial_element_count = elements_.size();
    std::vector<GroupedObservation> grouped;
    grouped.reserve(observations.size());
    std::vector<MapObservation> unmatched;
    unmatched.reserve(observations.size());

    const float association_radius_sq =
        options_.association_radius * options_.association_radius;
    for (const MapObservation& observation : observations) {
      int best_index = -1;
      float best_distance_sq = association_radius_sq;
      for (size_t i = 0; i < initial_element_count; ++i) {
        const float distance_sq =
            (elements_[i].mu - observation.position).squaredNorm();
        if (distance_sq <= best_distance_sq) {
          best_distance_sq = distance_sq;
          best_index = static_cast<int>(i);
        }
      }
      if (best_index < 0) {
        unmatched.push_back(observation);
        continue;
      }
      bool merged_group = false;
      for (GroupedObservation& group : grouped) {
        if (group.element_index == static_cast<size_t>(best_index)) {
          AccumulateObservation(observation, &group.observation);
          merged_group = true;
          break;
        }
      }
      if (!merged_group) {
        grouped.push_back(GroupedObservation{
            static_cast<size_t>(best_index), observation});
      }
    }

    for (const GroupedObservation& group : grouped) {
      if (UpdateElement(group.element_index, group.observation, scan_id)) {
        ++stats.fused;
        if (elements_[group.element_index].became_mature_this_update) {
          ++stats.matured;
        }
      }
    }
    for (const MapObservation& observation : unmatched) {
      if (elements_.size() >= options_.max_elements) {
        ++stats.dropped_capacity;
        continue;
      }
      CreateElement(observation, scan_id);
      ++stats.created;
    }

    FillSummaryStats(&stats);
    latest_stats_ = stats;
    return stats;
  }

  size_t published_count() const {
    if (!options_.publish_mature_only) return elements_.size();
    size_t count = 0;
    for (const MapElement& element : elements_) {
      if (element.mature) ++count;
    }
    return count;
  }

  size_t element_count() const { return elements_.size(); }
  const MapElementUpdateStats& latest_stats() const { return latest_stats_; }

  template <typename Callback>
  void ForEachPublished(Callback callback) const {
    for (const MapElement& element : elements_) {
      if (options_.publish_mature_only && !element.mature) continue;
      callback(element.mu, element.intensity,
               static_cast<float>(element.support_scan_count));
    }
  }

 private:
  struct MapElement {
    Eigen::Vector3f mu = Eigen::Vector3f::Zero();
    Eigen::Vector3f empirical_mean = Eigen::Vector3f::Zero();
    Eigen::Vector3f empirical_m2 = Eigen::Vector3f::Zero();
    float weight_sum = 0.0f;
    uint32_t support_scan_count = 0;
    uint32_t total_observation_count = 0;
    uint64_t first_seen_scan = 0;
    uint64_t last_seen_scan = 0;
    uint64_t last_support_scan = 0;
    uint64_t last_fusion_scan = 0;
    bool mature = false;
    bool became_mature_this_update = false;
    float intensity = 0.0f;
  };

  struct GroupedObservation {
    size_t element_index = 0;
    MapObservation observation;
  };

  static void AccumulateObservation(const MapObservation& source,
                                    MapObservation* destination) {
    const uint32_t old_count = destination->count;
    const uint32_t new_count = old_count + source.count;
    if (new_count == 0) return;
    const float source_weight = static_cast<float>(source.count);
    const float old_weight = static_cast<float>(old_count);
    destination->position =
        (destination->position * old_weight + source.position * source_weight) /
        static_cast<float>(new_count);
    destination->lidar_reference =
        (destination->lidar_reference * old_weight +
         source.lidar_reference * source_weight) /
        static_cast<float>(new_count);
    destination->intensity = std::max(destination->intensity, source.intensity);
    destination->count = new_count;
  }

  void CreateElement(const MapObservation& observation, uint64_t scan_id) {
    MapElement element;
    element.mu = observation.position;
    element.empirical_mean = observation.position;
    element.weight_sum = 1.0f;
    element.support_scan_count = 1;
    element.total_observation_count = 1;
    element.first_seen_scan = scan_id;
    element.last_seen_scan = scan_id;
    element.last_support_scan = scan_id;
    element.last_fusion_scan = scan_id;
    element.intensity = observation.intensity;
    element.mature =
        options_.mature_min_support_scans <= 1 && Rms(element) <= options_.mature_max_rms;
    element.became_mature_this_update = element.mature;
    elements_.push_back(element);
  }

  bool UpdateElement(size_t element_index, const MapObservation& observation,
                     uint64_t scan_id) {
    if (element_index >= elements_.size()) return false;
    MapElement& element = elements_[element_index];
    if (element.last_fusion_scan == scan_id) return false;
    element.became_mature_this_update = false;
    element.last_seen_scan = scan_id;
    element.last_fusion_scan = scan_id;
    if (element.last_support_scan != scan_id) {
      ++element.support_scan_count;
      element.last_support_scan = scan_id;
    }
    ++element.total_observation_count;
    const float denominator =
        std::min(options_.weight_cap, element.weight_sum + 1.0f);
    const float gain = denominator > 0.0f ? 1.0f / denominator : 1.0f;
    element.mu += gain * (observation.position - element.mu);
    element.weight_sum = denominator;
    element.intensity = std::max(element.intensity, observation.intensity);

    const float count = static_cast<float>(element.total_observation_count);
    const Eigen::Vector3f delta = observation.position - element.empirical_mean;
    element.empirical_mean += delta / count;
    const Eigen::Vector3f delta2 = observation.position - element.empirical_mean;
    element.empirical_m2 += delta.cwiseProduct(delta2);

    if (!element.mature &&
        element.support_scan_count >= options_.mature_min_support_scans &&
        Rms(element) <= options_.mature_max_rms) {
      element.mature = true;
      element.became_mature_this_update = true;
    }
    return true;
  }

  static double Rms(const MapElement& element) {
    if (element.total_observation_count < 2) return 0.0;
    const double variance_sum =
        static_cast<double>(element.empirical_m2.x() +
                            element.empirical_m2.y() +
                            element.empirical_m2.z()) /
        static_cast<double>(element.total_observation_count - 1);
    return std::sqrt(std::max(0.0, variance_sum));
  }

  void FillSummaryStats(MapElementUpdateStats* stats) const {
    std::vector<double> mature_rms;
    mature_rms.reserve(elements_.size());
    for (const MapElement& element : elements_) {
      if (element.mature) {
        ++stats->mature_count;
        mature_rms.push_back(Rms(element));
      } else {
        ++stats->candidate_count;
      }
    }
    if (mature_rms.empty()) return;
    std::sort(mature_rms.begin(), mature_rms.end());
    const auto percentile = [&mature_rms](double fraction) {
      const size_t index = std::min(
          mature_rms.size() - 1,
          static_cast<size_t>(std::floor(fraction *
                                         static_cast<double>(mature_rms.size() - 1))));
      return mature_rms[index];
    };
    stats->mature_rms_p50 = percentile(0.50);
    stats->mature_rms_p95 = percentile(0.95);
    stats->mature_rms_max = mature_rms.back();
  }

  Options options_;
  std::vector<MapElement> elements_;
  MapElementUpdateStats latest_stats_;
};

class MotionCompensator {
 public:
  struct Options {
    size_t max_imu_samples = 800;
    double gravity_mps2 = 9.7946;
    double gyro_noise_std = 0.02;
    double accel_noise_std = 0.2;
    double gyro_bias_noise_std = 0.0005;
    double accel_bias_noise_std = 0.005;
    double pose_position_std = 0.15;
    double pose_rotation_std = 0.08;
    bool hard_pose_anchor = false;
    double initial_bias_window_sec = 0.5;
  };

  explicit MotionCompensator(Options options) : options_(options) {}

  void PushImu(const ImuSample& sample) {
    if (!imu_history_.empty() && sample.time <= imu_history_.back().time) return;
    imu_history_.push_back(sample);
    if (initialized_ && sample.time > trajectory_.back().state.time) {
      AppendKnot(sample);
    }
    while (imu_history_.size() > options_.max_imu_samples) imu_history_.pop_front();
    while (trajectory_.size() > options_.max_imu_samples) trajectory_.pop_front();
  }

  bool Initialize(const PoseCorrection& correction) {
    ImuSample reference_imu;
    if (!InterpolateImu(correction.time, &reference_imu)) return false;
    NominalState state;
    state.time = correction.time;
    state.q_odom_imu = correction.q_odom_imu;
    state.position = correction.position;
    state.covariance.setZero();
    state.covariance.block<3, 3>(0, 0).diagonal().setConstant(0.25);
    state.covariance.block<3, 3>(3, 3).diagonal().setConstant(1.0);
    state.covariance.block<3, 3>(6, 6).diagonal().setConstant(0.1);
    state.covariance.block<3, 3>(9, 9).diagonal().setConstant(0.02);
    state.covariance.block<3, 3>(12, 12).diagonal().setConstant(0.2);

    EstimateInitialBiases(correction.time, state.q_odom_imu, &state);
    trajectory_.clear();
    trajectory_.push_back(TrajectoryKnot{reference_imu, state});
    for (const ImuSample& sample : imu_history_) {
      if (sample.time > correction.time) AppendKnot(sample);
    }
    initialized_ = true;
    return true;
  }

  void Reset() {
    initialized_ = false;
    imu_history_.clear();
    trajectory_.clear();
  }

  bool initialized() const { return initialized_; }

  bool StateAt(double time, NominalState* state) const {
    if (!initialized_ || trajectory_.empty() || time < trajectory_.front().state.time ||
        time > trajectory_.back().state.time) {
      return false;
    }
    const auto upper = std::lower_bound(
        trajectory_.begin(), trajectory_.end(), time,
        [](const TrajectoryKnot& knot, double target) { return knot.state.time < target; });
    if (upper == trajectory_.begin()) {
      *state = upper->state;
      return std::abs(upper->state.time - time) < 1e-6;
    }
    if (upper != trajectory_.end() && std::abs(upper->state.time - time) < 1e-6) {
      *state = upper->state;
      return true;
    }
    const auto lower = std::prev(upper);
    ImuSample target_imu;
    if (upper == trajectory_.end()) {
      if (!InterpolateImu(time, &target_imu)) return false;
    } else {
      const double ratio = (time - lower->imu.time) / (upper->imu.time - lower->imu.time);
      target_imu.time = time;
      target_imu.gyro = lower->imu.gyro + ratio * (upper->imu.gyro - lower->imu.gyro);
      target_imu.accel = lower->imu.accel + ratio * (upper->imu.accel - lower->imu.accel);
    }
    *state = Propagate(lower->state, lower->imu, target_imu);
    return true;
  }

  bool ApplyPoseCorrection(const PoseCorrection& correction) {
    NominalState corrected;
    ImuSample reference_imu;
    if (!StateAt(correction.time, &corrected) ||
        !InterpolateImu(correction.time, &reference_imu)) {
      return false;
    }
    CorrectPose(correction, &corrected);
    trajectory_.clear();
    trajectory_.push_back(TrajectoryKnot{reference_imu, corrected});
    for (const ImuSample& sample : imu_history_) {
      if (sample.time > correction.time) AppendKnot(sample);
    }
    return true;
  }

 private:
  bool InterpolateImu(double time, ImuSample* result) const {
    if (imu_history_.size() < 2 || time < imu_history_.front().time ||
        time > imu_history_.back().time) {
      return false;
    }
    const auto upper = std::lower_bound(
        imu_history_.begin(), imu_history_.end(), time,
        [](const ImuSample& sample, double target) { return sample.time < target; });
    if (upper != imu_history_.end() && std::abs(upper->time - time) < 1e-9) {
      *result = *upper;
      return true;
    }
    if (upper == imu_history_.begin() || upper == imu_history_.end()) return false;
    const ImuSample& lower = *std::prev(upper);
    const double ratio = (time - lower.time) / (upper->time - lower.time);
    result->time = time;
    result->gyro = lower.gyro + ratio * (upper->gyro - lower.gyro);
    result->accel = lower.accel + ratio * (upper->accel - lower.accel);
    return true;
  }

  NominalState Propagate(const NominalState& previous, const ImuSample& first,
                         const ImuSample& second) const {
    NominalState result = previous;
    const double dt = second.time - first.time;
    if (dt <= 0.0) return result;
    const Eigen::Vector3d gyro = 0.5 * (first.gyro + second.gyro) - previous.gyro_bias;
    const Eigen::Vector3d accel = 0.5 * (first.accel + second.accel) - previous.accel_bias;
    const Eigen::Quaterniond q_mid = previous.q_odom_imu * ExpQuaternion(0.5 * gyro * dt);
    const Eigen::Vector3d global_accel =
        q_mid * accel + Eigen::Vector3d(0.0, 0.0, -options_.gravity_mps2);
    result.position += previous.velocity * dt + 0.5 * global_accel * dt * dt;
    result.velocity += global_accel * dt;
    result.q_odom_imu = (previous.q_odom_imu * ExpQuaternion(gyro * dt)).normalized();
    result.time = second.time;

    Eigen::Matrix<double, 15, 15> transition =
        Eigen::Matrix<double, 15, 15>::Identity();
    transition.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    transition.block<3, 3>(3, 6) = -previous.q_odom_imu.toRotationMatrix() * Skew(accel) * dt;
    transition.block<3, 3>(3, 12) = -previous.q_odom_imu.toRotationMatrix() * dt;
    transition.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity() * dt;
    Eigen::Matrix<double, 15, 15> noise = Eigen::Matrix<double, 15, 15>::Zero();
    noise.block<3, 3>(3, 3).diagonal().setConstant(options_.accel_noise_std * options_.accel_noise_std * dt);
    noise.block<3, 3>(6, 6).diagonal().setConstant(options_.gyro_noise_std * options_.gyro_noise_std * dt);
    noise.block<3, 3>(9, 9).diagonal().setConstant(options_.gyro_bias_noise_std * options_.gyro_bias_noise_std * dt);
    noise.block<3, 3>(12, 12).diagonal().setConstant(options_.accel_bias_noise_std * options_.accel_bias_noise_std * dt);
    result.covariance = transition * previous.covariance * transition.transpose() + noise;
    result.covariance = 0.5 * (result.covariance + result.covariance.transpose());
    return result;
  }

  void CorrectPose(const PoseCorrection& correction, NominalState* state) const {
    Eigen::Matrix<double, 6, 1> innovation;
    innovation.head<3>() = correction.position - state->position;
    innovation.tail<3>() = QuaternionLog(state->q_odom_imu.conjugate() * correction.q_odom_imu);
    Eigen::Matrix<double, 6, 15> measurement = Eigen::Matrix<double, 6, 15>::Zero();
    measurement.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    measurement.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 6, 6> noise = Eigen::Matrix<double, 6, 6>::Zero();
    noise.block<3, 3>(0, 0).diagonal().setConstant(
        options_.pose_position_std * options_.pose_position_std);
    noise.block<3, 3>(3, 3).diagonal().setConstant(
        options_.pose_rotation_std * options_.pose_rotation_std);
    const Eigen::Matrix<double, 6, 6> innovation_covariance =
        measurement * state->covariance * measurement.transpose() + noise;
    const Eigen::Matrix<double, 15, 6> gain =
        state->covariance * measurement.transpose() * innovation_covariance.ldlt().solve(
            Eigen::Matrix<double, 6, 6>::Identity());
    const Eigen::Matrix<double, 15, 1> delta = gain * innovation;
    state->position += delta.segment<3>(0);
    state->velocity += delta.segment<3>(3);
    state->q_odom_imu = (state->q_odom_imu * ExpQuaternion(delta.segment<3>(6))).normalized();
    state->gyro_bias += delta.segment<3>(9);
    state->accel_bias += delta.segment<3>(12);
    const Eigen::Matrix<double, 15, 15> identity =
        Eigen::Matrix<double, 15, 15>::Identity();
    state->covariance = (identity - gain * measurement) * state->covariance *
                        (identity - gain * measurement).transpose() + gain * noise * gain.transpose();
    state->covariance = 0.5 * (state->covariance + state->covariance.transpose());
    if (options_.hard_pose_anchor) {
      state->position = correction.position;
      state->q_odom_imu = correction.q_odom_imu.normalized();
      state->covariance.block<3, 3>(0, 0).diagonal().setConstant(
          options_.pose_position_std * options_.pose_position_std);
      state->covariance.block<3, 3>(6, 6).diagonal().setConstant(
          options_.pose_rotation_std * options_.pose_rotation_std);
    }
  }

  void AppendKnot(const ImuSample& sample) {
    const TrajectoryKnot& previous = trajectory_.back();
    trajectory_.push_back(TrajectoryKnot{sample,
        Propagate(previous.state, previous.imu, sample)});
  }

  void EstimateInitialBiases(double reference_time, const Eigen::Quaterniond& orientation,
                             NominalState* state) const {
    Eigen::Vector3d gyro_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_sum = Eigen::Vector3d::Zero();
    size_t count = 0;
    for (const ImuSample& sample : imu_history_) {
      if (sample.time >= reference_time - options_.initial_bias_window_sec &&
          sample.time <= reference_time) {
        gyro_sum += sample.gyro;
        accel_sum += sample.accel;
        ++count;
      }
    }
    if (count < 10) return;
    state->gyro_bias = gyro_sum / static_cast<double>(count);
    const Eigen::Vector3d expected_specific_force =
        orientation.conjugate() * Eigen::Vector3d(0.0, 0.0, options_.gravity_mps2);
    state->accel_bias = accel_sum / static_cast<double>(count) - expected_specific_force;
  }

  Options options_;
  bool initialized_ = false;
  std::deque<ImuSample> imu_history_;
  std::deque<TrajectoryKnot> trajectory_;
};

struct Config {
  std::string lidar_topic = "/livox/lidar";
  std::string imu_topic = "/livox/imu";
  std::string pose_topic = "/lio/odom";
  std::string pose_input_frame = "lidar";
  std::string external_odom_frame;
  std::string imu_acceleration_input_unit = "g";
  std::string lidar_time_base = "header_stamp";
  std::string map_frame = "reflective_odom";
  // livox_ros_driver2 forwards its native accelerometer measurement in g,
  // even though sensor_msgs/Imu conventionally uses m/s^2.
  double imu_acceleration_scale = 9.80665;
  std::string lidar_frame = "livox_frame";
  int reflectivity_threshold = 160;
  float min_distance_m = 0.1f;
  float max_distance_m = 30.0f;
  double offset_time_scale_sec = 1e-9;
  double lidar_to_imu_time_offset_sec = 0.0;
  double pose_match_epsilon_sec = 0.002;
  double pose_match_max_time_diff_sec = 0.06;
  double pose_match_max_future_sec = 0.01;
  double pose_timeout_sec = 2.0;
  std::string trajectory_mode = "legacy_imu_single_anchor_debug";
  double odom_time_offset_sec = 0.0;
  double max_odom_bracket_gap_sec = 0.15;
  int max_odom_samples = 64;
  double max_external_velocity_mps = 5.0;
  double max_external_angular_velocity_rps = 2.0;
  double external_position_margin_m = 0.5;
  double external_rotation_margin_rad = 0.3;
  int max_pending_scans = 24;
  int max_pending_poses = 32;
  int max_high_points_per_scan = 8192;
  double wheel_tick_sec = 0.05;
  double ttl_sec = 1.0;
  int map_hash_capacity = 65536;
  int map_node_capacity = 32768;
  int wheel_bucket_count = 128;
  double voxel_size_m = 0.05;
  double merge_distance_m = 0.05;
  bool enable_map_element_filter = true;
  double scan_observation_voxel_size_m = 0.015;
  double map_element_association_radius_m = 0.03;
  int map_element_mature_min_support_scans = 3;
  double map_element_mature_max_rms_m = 0.025;
  double map_element_weight_cap = 8.0;
  int map_element_max_elements = 32768;
  bool publish_mature_map_only = true;
  bool deskew_use_imu_translation = true;
  bool deskew_use_external_pose_interpolation = false;
  bool publish_debug_map = true;
  double debug_map_publish_rate_hz = 1.0;
  std::string debug_map_topic = "/reflective/rolling_map";
  bool publish_map_snapshot = true;
  std::string map_snapshot_topic = "/reflective/map_snapshot";
	  bool publish_deskewed_high_ref_cloud = true;
	  std::string deskewed_high_ref_cloud_topic = "/reflective/deskewed_high_ref_cloud";
  bool publish_all_high_ref_history = true;
  std::string all_high_ref_history_topic = "/reflective/all_high_ref_history";
  int max_all_high_ref_history_points = 0;
  bool save_all_high_ref_history_pcd = true;
  std::string all_high_ref_history_pcd_path =
      "/workspace/moving/reflective_all_high_history_ge160.pcd";
  std::string all_high_ref_history_parent_pcd_path =
      "/workspace/moving/reflective_all_high_history_ge160_world.pcd";
	  bool publish_observation_frame = true;
	  std::string observation_frame_topic = "/reflective/observation_frame";
	  int max_observation_points_per_frame = 4096;
	  uint64_t observation_source_contract_hash = 0;
	  bool publish_path_since_start = true;
  std::string path_since_start_topic = "/reflective/path_since_start";
  int max_path_poses = 20000;
  double worker_rate_hz = 250.0;
  std::vector<double> imu_from_lidar_translation{-0.011, -0.02329, 0.04412};
  std::vector<double> imu_from_lidar_rotation{1.0, 0.0, 0.0, 0.0, 1.0,
                                                0.0, 0.0, 0.0, 1.0};
  MotionCompensator::Options motion;
};

class DynamicReflectiveMappingNode {
 public:
  DynamicReflectiveMappingNode()
      : private_nh_("~"), config_(LoadConfig()),
        reflectivity_filter_(static_cast<uint8_t>(config_.reflectivity_threshold)),
        motion_(config_.motion), map_(MakeMapOptions(config_)),
        odom_buffer_(MakeOdomBufferOptions(config_)),
        map_element_filter_(MakeMapElementOptions(config_)) {
    ValidateConfig();
    imu_from_lidar_rotation_ = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
        config_.imu_from_lidar_rotation.data());
    imu_from_lidar_translation_ = Eigen::Map<const Eigen::Vector3d>(
        config_.imu_from_lidar_translation.data());

    pending_scans_.resize(config_.max_pending_scans);
    for (PendingScan& scan : pending_scans_) {
      scan.points.reserve(config_.max_high_points_per_scan);
    }

    lidar_subscriber_ = nh_.subscribe(config_.lidar_topic, 1,
        &DynamicReflectiveMappingNode::HandleLidar, this, ros::TransportHints().tcpNoDelay());
    imu_subscriber_ = nh_.subscribe(config_.imu_topic, 1,
        &DynamicReflectiveMappingNode::HandleImu, this, ros::TransportHints().tcpNoDelay());
    pose_subscriber_ = nh_.subscribe(config_.pose_topic, 1,
        &DynamicReflectiveMappingNode::HandlePose, this, ros::TransportHints().tcpNoDelay());
    if (config_.publish_debug_map) {
      map_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(config_.debug_map_topic, 1, false);
      InitializeMapMessage();
    }
    if (config_.publish_map_snapshot) {
      map_snapshot_publisher_ =
          nh_.advertise<livox_reflective_marker::ReflectiveMapSnapshot>(
              config_.map_snapshot_topic, 1, false);
      if (!config_.publish_debug_map) InitializeMapMessage();
    }
	    if (config_.publish_deskewed_high_ref_cloud) {
	      deskewed_high_ref_cloud_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(
	          config_.deskewed_high_ref_cloud_topic, 1, false);
	    }
    if (config_.publish_all_high_ref_history) {
      all_high_ref_history_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(
          config_.all_high_ref_history_topic, 1, true);
    }
	    if (config_.publish_observation_frame) {
	      observation_frame_publisher_ =
	          nh_.advertise<livox_reflective_marker::ReflectiveObservationFrame>(
	              config_.observation_frame_topic, 2, false);
	    }
	    if (config_.publish_path_since_start) {
      path_since_start_.header.frame_id = config_.map_frame;
      path_since_start_publisher_ =
          nh_.advertise<nav_msgs::Path>(config_.path_since_start_topic, 1, true);
    }
    worker_timer_ = nh_.createTimer(ros::Duration(1.0 / config_.worker_rate_hz),
                                    &DynamicReflectiveMappingNode::Worker, this);
    if (config_.publish_debug_map || config_.publish_map_snapshot) {
      map_publish_timer_ = nh_.createTimer(
          ros::Duration(1.0 / config_.debug_map_publish_rate_hz),
          &DynamicReflectiveMappingNode::PublishMapTimer, this);
    }

    ROS_INFO("Dynamic reflective mapper: lidar=%s imu=%s pose=%s map=%s accel_unit=%s",
             config_.lidar_topic.c_str(), config_.imu_topic.c_str(),
             config_.pose_topic.c_str(), config_.debug_map_topic.c_str(),
             config_.imu_acceleration_input_unit.c_str());
    ROS_INFO("reflective mapper config: pose_input_frame=%s lidar_time_base=%s "
             "offset_scale=%.3g lidar_to_imu_offset=%.6f reflectivity_threshold=%d "
             "trajectory_mode=%s odom_time_offset=%.6f max_odom_gap=%.3f odom_samples=%d "
             "map_element_filter=%s scan_obs_voxel=%.3f assoc_radius=%.3f "
             "mature_support=%d mature_rms=%.3f mature_only=%s "
             "all_high_history=%s all_high_topic=%s max_all_high=%d "
             "save_all_high_pcd=%s pcd_path=%s parent_pcd_path=%s "
             "range=[%.3f, %.3f] pose_std=[%.4fm %.4frad] hard_pose_anchor=%s "
             "deskew_use_imu_translation=%s external_pose_interp=%s "
             "T_imu_lidar_t=[%.5f %.5f %.5f]",
             config_.pose_input_frame.c_str(), config_.lidar_time_base.c_str(),
             config_.offset_time_scale_sec, config_.lidar_to_imu_time_offset_sec,
             config_.reflectivity_threshold, config_.trajectory_mode.c_str(),
             config_.odom_time_offset_sec, config_.max_odom_bracket_gap_sec,
             config_.max_odom_samples,
             config_.enable_map_element_filter ? "true" : "false",
             config_.scan_observation_voxel_size_m,
             config_.map_element_association_radius_m,
             config_.map_element_mature_min_support_scans,
             config_.map_element_mature_max_rms_m,
             config_.publish_mature_map_only ? "true" : "false",
             config_.publish_all_high_ref_history ? "true" : "false",
             config_.all_high_ref_history_topic.c_str(),
             config_.max_all_high_ref_history_points,
             config_.save_all_high_ref_history_pcd ? "true" : "false",
             config_.all_high_ref_history_pcd_path.c_str(),
             config_.all_high_ref_history_parent_pcd_path.c_str(),
             config_.min_distance_m,
             config_.max_distance_m, config_.motion.pose_position_std,
             config_.motion.pose_rotation_std,
             config_.motion.hard_pose_anchor ? "true" : "false",
             config_.deskew_use_imu_translation ? "true" : "false",
             config_.deskew_use_external_pose_interpolation ? "true" : "false",
             imu_from_lidar_translation_.x(), imu_from_lidar_translation_.y(),
             imu_from_lidar_translation_.z());
  }

  ~DynamicReflectiveMappingNode() {
    SaveAllHighRefHistoryPcd();
  }

 private:
  static ReflectiveVoxelMap::Options MakeMapOptions(const Config& config) {
    ReflectiveVoxelMap::Options options;
    options.hash_capacity = config.map_hash_capacity;
    options.node_capacity = config.map_node_capacity;
    options.wheel_bucket_count = config.wheel_bucket_count;
    options.voxel_size = static_cast<float>(config.voxel_size_m);
    options.merge_distance = static_cast<float>(config.merge_distance_m);
    options.enable_ttl = config.ttl_sec != -1.0;
    if (options.enable_ttl) {
      options.ttl_ticks = std::max<uint64_t>(1, static_cast<uint64_t>(
          std::ceil(config.ttl_sec / config.wheel_tick_sec)));
    }
    return options;
  }

  static OdomBuffer::Options MakeOdomBufferOptions(const Config& config) {
    OdomBuffer::Options options;
    options.max_samples = static_cast<size_t>(config.max_odom_samples);
    options.max_velocity_mps = config.max_external_velocity_mps;
    options.max_angular_velocity_rps = config.max_external_angular_velocity_rps;
    options.position_margin_m = config.external_position_margin_m;
    options.rotation_margin_rad = config.external_rotation_margin_rad;
    return options;
  }

  static MapElementFilter::Options MakeMapElementOptions(const Config& config) {
    MapElementFilter::Options options;
    options.enabled = config.enable_map_element_filter;
    options.association_radius =
        static_cast<float>(config.map_element_association_radius_m);
    options.mature_min_support_scans = static_cast<uint32_t>(
        std::max(1, config.map_element_mature_min_support_scans));
    options.mature_max_rms =
        static_cast<float>(config.map_element_mature_max_rms_m);
    options.weight_cap = static_cast<float>(config.map_element_weight_cap);
    options.max_elements = static_cast<size_t>(
        std::max(1, config.map_element_max_elements));
    options.publish_mature_only = config.publish_mature_map_only;
    return options;
  }

  Config LoadConfig() {
    Config config;
    private_nh_.param("lidar_topic", config.lidar_topic, config.lidar_topic);
    private_nh_.param("imu_topic", config.imu_topic, config.imu_topic);
    private_nh_.param("pose_topic", config.pose_topic, config.pose_topic);
    private_nh_.param("pose_input_frame", config.pose_input_frame,
                      config.pose_input_frame);
    private_nh_.param("external_odom_frame", config.external_odom_frame,
                      config.external_odom_frame);
    private_nh_.param("imu_acceleration_input_unit",
                      config.imu_acceleration_input_unit,
                      config.imu_acceleration_input_unit);
    private_nh_.param("lidar_time_base", config.lidar_time_base, config.lidar_time_base);
    private_nh_.param("map_frame", config.map_frame, config.map_frame);
    private_nh_.param("imu_acceleration_scale", config.imu_acceleration_scale,
                      config.imu_acceleration_scale);
    private_nh_.param("lidar_frame", config.lidar_frame, config.lidar_frame);
    private_nh_.param("reflectivity_threshold", config.reflectivity_threshold,
                      config.reflectivity_threshold);
    private_nh_.param("min_distance_m", config.min_distance_m, config.min_distance_m);
    private_nh_.param("max_distance_m", config.max_distance_m, config.max_distance_m);
    private_nh_.param("offset_time_scale_sec", config.offset_time_scale_sec,
                      config.offset_time_scale_sec);
    private_nh_.param("lidar_to_imu_time_offset_sec", config.lidar_to_imu_time_offset_sec,
                      config.lidar_to_imu_time_offset_sec);
    private_nh_.param("pose_match_epsilon_sec", config.pose_match_epsilon_sec,
                      config.pose_match_epsilon_sec);
    private_nh_.param("pose_match_max_time_diff_sec",
                      config.pose_match_max_time_diff_sec,
                      config.pose_match_max_time_diff_sec);
    private_nh_.param("pose_match_max_future_sec",
                      config.pose_match_max_future_sec,
                      config.pose_match_max_future_sec);
    private_nh_.param("pose_timeout_sec", config.pose_timeout_sec, config.pose_timeout_sec);
    private_nh_.param("trajectory_mode", config.trajectory_mode,
                      config.trajectory_mode);
    private_nh_.param("odom_time_offset_sec", config.odom_time_offset_sec,
                      config.odom_time_offset_sec);
    private_nh_.param("max_odom_bracket_gap_sec", config.max_odom_bracket_gap_sec,
                      config.max_odom_bracket_gap_sec);
    private_nh_.param("max_odom_samples", config.max_odom_samples,
                      config.max_odom_samples);
    private_nh_.param("max_external_velocity_mps", config.max_external_velocity_mps,
                      config.max_external_velocity_mps);
    private_nh_.param("max_external_angular_velocity_rps",
                      config.max_external_angular_velocity_rps,
                      config.max_external_angular_velocity_rps);
    private_nh_.param("external_position_margin_m", config.external_position_margin_m,
                      config.external_position_margin_m);
    private_nh_.param("external_rotation_margin_rad", config.external_rotation_margin_rad,
                      config.external_rotation_margin_rad);
    private_nh_.param("max_pending_scans", config.max_pending_scans,
                      config.max_pending_scans);
    private_nh_.param("max_pending_poses", config.max_pending_poses,
                      config.max_pending_poses);
    private_nh_.param("max_high_points_per_scan", config.max_high_points_per_scan,
                      config.max_high_points_per_scan);
    private_nh_.param("wheel_tick_sec", config.wheel_tick_sec, config.wheel_tick_sec);
    private_nh_.param("ttl_sec", config.ttl_sec, config.ttl_sec);
    private_nh_.param("map_hash_capacity", config.map_hash_capacity, config.map_hash_capacity);
    private_nh_.param("map_node_capacity", config.map_node_capacity, config.map_node_capacity);
    private_nh_.param("wheel_bucket_count", config.wheel_bucket_count,
                      config.wheel_bucket_count);
    private_nh_.param("voxel_size_m", config.voxel_size_m, config.voxel_size_m);
    private_nh_.param("merge_distance_m", config.merge_distance_m,
                      config.merge_distance_m);
    private_nh_.param("enable_map_element_filter",
                      config.enable_map_element_filter,
                      config.enable_map_element_filter);
    private_nh_.param("scan_observation_voxel_size_m",
                      config.scan_observation_voxel_size_m,
                      config.scan_observation_voxel_size_m);
    private_nh_.param("map_element_association_radius_m",
                      config.map_element_association_radius_m,
                      config.map_element_association_radius_m);
    private_nh_.param("map_element_mature_min_support_scans",
                      config.map_element_mature_min_support_scans,
                      config.map_element_mature_min_support_scans);
    private_nh_.param("map_element_mature_max_rms_m",
                      config.map_element_mature_max_rms_m,
                      config.map_element_mature_max_rms_m);
    private_nh_.param("map_element_weight_cap",
                      config.map_element_weight_cap,
                      config.map_element_weight_cap);
    private_nh_.param("map_element_max_elements",
                      config.map_element_max_elements,
                      config.map_element_max_elements);
    private_nh_.param("publish_mature_map_only",
                      config.publish_mature_map_only,
                      config.publish_mature_map_only);
    private_nh_.param("deskew_use_imu_translation",
                      config.deskew_use_imu_translation,
                      config.deskew_use_imu_translation);
    private_nh_.param("deskew_use_external_pose_interpolation",
                      config.deskew_use_external_pose_interpolation,
                      config.deskew_use_external_pose_interpolation);
    private_nh_.param("publish_debug_map", config.publish_debug_map,
                      config.publish_debug_map);
    private_nh_.param("debug_map_publish_rate_hz", config.debug_map_publish_rate_hz,
                      config.debug_map_publish_rate_hz);
    private_nh_.param("debug_map_topic", config.debug_map_topic, config.debug_map_topic);
    private_nh_.param("publish_map_snapshot", config.publish_map_snapshot,
                      config.publish_map_snapshot);
    private_nh_.param("map_snapshot_topic", config.map_snapshot_topic,
                      config.map_snapshot_topic);
    private_nh_.param("publish_deskewed_high_ref_cloud",
                      config.publish_deskewed_high_ref_cloud,
                      config.publish_deskewed_high_ref_cloud);
	    private_nh_.param("deskewed_high_ref_cloud_topic",
	                      config.deskewed_high_ref_cloud_topic,
	                      config.deskewed_high_ref_cloud_topic);
    private_nh_.param("publish_all_high_ref_history",
                      config.publish_all_high_ref_history,
                      config.publish_all_high_ref_history);
    private_nh_.param("all_high_ref_history_topic",
                      config.all_high_ref_history_topic,
                      config.all_high_ref_history_topic);
    private_nh_.param("max_all_high_ref_history_points",
                      config.max_all_high_ref_history_points,
                      config.max_all_high_ref_history_points);
    private_nh_.param("save_all_high_ref_history_pcd",
                      config.save_all_high_ref_history_pcd,
                      config.save_all_high_ref_history_pcd);
    private_nh_.param("all_high_ref_history_pcd_path",
                      config.all_high_ref_history_pcd_path,
                      config.all_high_ref_history_pcd_path);
    private_nh_.param("all_high_ref_history_parent_pcd_path",
                      config.all_high_ref_history_parent_pcd_path,
                      config.all_high_ref_history_parent_pcd_path);
	    private_nh_.param("publish_observation_frame",
	                      config.publish_observation_frame,
	                      config.publish_observation_frame);
	    private_nh_.param("observation_frame_topic",
	                      config.observation_frame_topic,
	                      config.observation_frame_topic);
	    private_nh_.param("max_observation_points_per_frame",
	                      config.max_observation_points_per_frame,
	                      config.max_observation_points_per_frame);
	    int observation_source_contract_hash = 0;
	    private_nh_.param("observation_source_contract_hash",
	                      observation_source_contract_hash,
	                      observation_source_contract_hash);
	    config.observation_source_contract_hash =
	        static_cast<uint64_t>(std::max(0, observation_source_contract_hash));
	    private_nh_.param("publish_path_since_start", config.publish_path_since_start,
                      config.publish_path_since_start);
    private_nh_.param("path_since_start_topic", config.path_since_start_topic,
                      config.path_since_start_topic);
    private_nh_.param("max_path_poses", config.max_path_poses,
                      config.max_path_poses);
    private_nh_.param("worker_rate_hz", config.worker_rate_hz, config.worker_rate_hz);
    private_nh_.getParam("imu_from_lidar_translation", config.imu_from_lidar_translation);
    private_nh_.getParam("imu_from_lidar_rotation", config.imu_from_lidar_rotation);
    private_nh_.param("gravity_mps2", config.motion.gravity_mps2,
                      config.motion.gravity_mps2);
    int max_imu_samples = static_cast<int>(config.motion.max_imu_samples);
    private_nh_.param("max_imu_samples", max_imu_samples, max_imu_samples);
    config.motion.max_imu_samples = std::max(0, max_imu_samples);
    private_nh_.param("gyro_noise_std", config.motion.gyro_noise_std,
                      config.motion.gyro_noise_std);
    private_nh_.param("accel_noise_std", config.motion.accel_noise_std,
                      config.motion.accel_noise_std);
    private_nh_.param("gyro_bias_noise_std", config.motion.gyro_bias_noise_std,
                      config.motion.gyro_bias_noise_std);
    private_nh_.param("accel_bias_noise_std", config.motion.accel_bias_noise_std,
                      config.motion.accel_bias_noise_std);
    private_nh_.param("pose_position_std", config.motion.pose_position_std,
                      config.motion.pose_position_std);
    private_nh_.param("pose_rotation_std", config.motion.pose_rotation_std,
                      config.motion.pose_rotation_std);
    private_nh_.param("hard_pose_anchor", config.motion.hard_pose_anchor,
                      config.motion.hard_pose_anchor);
    private_nh_.param("initial_bias_window_sec", config.motion.initial_bias_window_sec,
                      config.motion.initial_bias_window_sec);
    if (config.trajectory_mode == "external_previous_current_debug") {
      config.deskew_use_external_pose_interpolation = true;
      config.deskew_use_imu_translation = false;
    } else if (config.trajectory_mode == "imu_rotation_debug") {
      config.deskew_use_external_pose_interpolation = false;
      config.deskew_use_imu_translation = false;
    } else if (config.trajectory_mode == "imu_full_debug") {
      config.deskew_use_external_pose_interpolation = false;
      config.deskew_use_imu_translation = true;
    }
    return config;
  }

  void ValidateConfig() {
    if (config_.reflectivity_threshold < 0 || config_.reflectivity_threshold > 255 ||
        config_.max_pending_scans <= 0 || config_.max_pending_poses <= 0 ||
        config_.max_high_points_per_scan <= 0 || config_.wheel_tick_sec <= 0.0 ||
        (config_.ttl_sec != -1.0 && config_.ttl_sec <= 0.0) ||
    config_.map_hash_capacity <= 1 || config_.map_node_capacity <= 0 ||
    config_.map_node_capacity >= config_.map_hash_capacity ||
    config_.voxel_size_m <= 0.0 || config_.merge_distance_m <= 0.0 ||
        config_.scan_observation_voxel_size_m <= 0.0 ||
        config_.map_element_association_radius_m <= 0.0 ||
        config_.map_element_mature_min_support_scans <= 0 ||
        config_.map_element_mature_max_rms_m <= 0.0 ||
        config_.map_element_weight_cap < 1.0 ||
        config_.map_element_max_elements <= 0 ||
        config_.merge_distance_m > std::sqrt(3.0) * config_.voxel_size_m ||
        config_.offset_time_scale_sec <= 0.0 || config_.imu_acceleration_scale <= 0.0 ||
        config_.pose_match_epsilon_sec < 0.0 ||
        config_.pose_match_max_time_diff_sec <= 0.0 ||
        config_.pose_match_max_future_sec < 0.0 ||
        config_.max_odom_bracket_gap_sec <= 0.0 ||
        config_.max_odom_samples < 2 ||
        config_.pose_timeout_sec <= 0.0 || config_.worker_rate_hz <= 0.0 ||
        config_.debug_map_publish_rate_hz <= 0.0 ||
        config_.max_observation_points_per_frame < 0 ||
        config_.max_all_high_ref_history_points < 0 ||
        config_.max_path_poses < 0 ||
        config_.motion.max_imu_samples < 2 ||
        config_.imu_from_lidar_translation.size() != 3 ||
        config_.imu_from_lidar_rotation.size() != 9 ||
        (config_.pose_input_frame != "imu" && config_.pose_input_frame != "lidar") ||
        (config_.trajectory_mode != "odom_poseat" &&
         config_.trajectory_mode != "external_previous_current_debug" &&
         config_.trajectory_mode != "legacy_imu_single_anchor_debug" &&
         config_.trajectory_mode != "imu_rotation_debug" &&
         config_.trajectory_mode != "imu_full_debug") ||
        (config_.imu_acceleration_input_unit != "g" &&
         config_.imu_acceleration_input_unit != "mps2") ||
        (config_.lidar_time_base != "header_stamp" &&
         config_.lidar_time_base != "timebase_ns")) {
      throw std::runtime_error("invalid dynamic_reflective_mapping parameters");
    }
  }

  double LidarBaseTime(const livox_ros_driver2::CustomMsg& scan) const {
    if (config_.lidar_time_base == "timebase_ns") {
      return static_cast<double>(scan.timebase) * 1e-9;
    }
    return scan.header.stamp.toSec();
  }

  uint64_t ToMapTick(double lidar_time) const {
    return static_cast<uint64_t>(std::floor(lidar_time / config_.wheel_tick_sec));
  }

  void HandleLidar(const livox_ros_driver2::CustomMsg::ConstPtr& message) {
    if (message->points.empty()) return;
    const double base_time = LidarBaseTime(*message);
    if (!IsFinite(base_time)) return;
    const double header_time = message->header.stamp.toSec();
    const double timebase_time = static_cast<double>(message->timebase) * 1e-9;
    uint32_t min_offset = std::numeric_limits<uint32_t>::max();
    uint32_t max_offset = 0;
    for (const livox_ros_driver2::CustomPoint& point : message->points) {
      min_offset = std::min(min_offset, point.offset_time);
      max_offset = std::max(max_offset, point.offset_time);
    }
    const double lidar_start = base_time + min_offset * config_.offset_time_scale_sec;
    const double lidar_end = base_time + max_offset * config_.offset_time_scale_sec;
    if (!IsFinite(lidar_start) || !IsFinite(lidar_end) || lidar_end < lidar_start) return;

    const uint32_t scan_slot = AllocateScanSlot();
    PendingScan& scan = pending_scans_[scan_slot];
    scan.active = true;
    scan.scan_id = ++next_scan_id_;
    scan.lidar_base_time = base_time;
    scan.lidar_header_time = header_time;
    scan.lidar_timebase_time = timebase_time;
    scan.imu_start_time = lidar_start + config_.lidar_to_imu_time_offset_sec;
    scan.imu_end_time = lidar_end + config_.lidar_to_imu_time_offset_sec;
    scan.end_tick = ToMapTick(lidar_end);
    scan.raw_point_count = static_cast<uint32_t>(message->points.size());
    scan.accepted_point_count = 0;
    scan.accepted_imu_start_time = std::numeric_limits<double>::infinity();
    scan.accepted_imu_end_time = -std::numeric_limits<double>::infinity();
    scan.rejected_intensity_count = 0;
    scan.rejected_nonfinite_count = 0;
    scan.rejected_range_count = 0;
    scan.dropped_budget_count = 0;
    scan.raw_tag30_bins.fill(0);
    scan.accepted_tag30_bins.fill(0);
    scan.trace_wait_logged = false;
    scan.points.clear();

    for (const livox_ros_driver2::CustomPoint& point : message->points) {
      ++scan.raw_tag30_bins[Tag30Bin(point.tag)];
      if (!reflectivity_filter_.Accepts(point)) {
        ++scan.rejected_intensity_count;
        continue;
      }
      const Eigen::Vector3f location(point.x, point.y, point.z);
      const float distance = location.norm();
      if (!location.allFinite()) {
        ++scan.rejected_nonfinite_count;
        continue;
      }
      if (distance < config_.min_distance_m || distance > config_.max_distance_m) {
        ++scan.rejected_range_count;
        continue;
      }
      if (scan.points.size() == scan.points.capacity()) {
        ++dropped_point_budget_;
        ++scan.dropped_budget_count;
        continue;
      }
      scan.points.push_back(PendingPoint{point.x, point.y, point.z, point.offset_time,
                                         point.reflectivity});
      ++scan.accepted_point_count;
      ++scan.accepted_tag30_bins[Tag30Bin(point.tag)];
      const double accepted_imu_time = base_time +
          point.offset_time * config_.offset_time_scale_sec +
          config_.lidar_to_imu_time_offset_sec;
      scan.accepted_imu_start_time =
          std::min(scan.accepted_imu_start_time, accepted_imu_time);
      scan.accepted_imu_end_time =
          std::max(scan.accepted_imu_end_time, accepted_imu_time);
    }
    ROS_INFO_THROTTLE(
        2.0,
        "scan input id=%llu raw=%u high=%u reject(intensity=%u nonfinite=%u range=%u budget=%u) "
        "tag30_raw=[%u,%u,%u,%u] tag30_high=[%u,%u,%u,%u] "
        "time_base=%s header=%.6f timebase=%.6f header_minus_timebase=%.3fms "
        "offset=[%u,%u] duration=%.3fms imu_window=[%.6f,%.6f] latest_imu_lag=%.3fms",
        static_cast<unsigned long long>(scan.scan_id), scan.raw_point_count,
        scan.accepted_point_count, scan.rejected_intensity_count,
        scan.rejected_nonfinite_count, scan.rejected_range_count,
        scan.dropped_budget_count, scan.raw_tag30_bins[0], scan.raw_tag30_bins[1],
        scan.raw_tag30_bins[2], scan.raw_tag30_bins[3], scan.accepted_tag30_bins[0],
        scan.accepted_tag30_bins[1], scan.accepted_tag30_bins[2],
        scan.accepted_tag30_bins[3], config_.lidar_time_base.c_str(), header_time,
        timebase_time, (header_time - timebase_time) * 1000.0, min_offset,
        max_offset, (lidar_end - lidar_start) * 1000.0, scan.imu_start_time,
        scan.imu_end_time, (latest_imu_time_ - scan.imu_end_time) * 1000.0);
    scan_order_.push_back(scan_slot);
  }

  void HandleImu(const sensor_msgs::Imu::ConstPtr& message) {
    const double time = message->header.stamp.toSec();
    const Eigen::Vector3d gyro(message->angular_velocity.x, message->angular_velocity.y,
                               message->angular_velocity.z);
    const Eigen::Vector3d raw_accel(message->linear_acceleration.x,
                                    message->linear_acceleration.y,
                                    message->linear_acceleration.z);
    const double accel_scale = config_.imu_acceleration_input_unit == "g"
        ? config_.imu_acceleration_scale : 1.0;
    const Eigen::Vector3d accel = raw_accel * accel_scale;
    if (!IsFinite(time) || !IsFinite(gyro) || !IsFinite(accel)) return;
    // A rosbag replay can restart its recorded clock while this node remains
    // alive.  The old IMU history is then from the future and PushImu() would
    // reject every sample in the new replay, leaving the mapper permanently
    // unable to initialize.
    constexpr double kTimeResetToleranceSec = 0.1;
    if (IsFinite(latest_imu_time_) && time < latest_imu_time_ - kTimeResetToleranceSec) {
      ROS_WARN("reflective mapper reset after IMU time moved backwards by %.3f s",
               latest_imu_time_ - time);
      ResetEpoch();
    }
    latest_imu_time_ = std::max(latest_imu_time_, time);
    motion_.PushImu(ImuSample{time, gyro, accel});
  }

  bool ConvertExternalPoseToMapFrame(const PoseCorrection& external,
                                     const std::string& parent_frame,
                                     PoseCorrection* local) {
    if (!map_anchor_.has_value()) {
      map_anchor_ = MapFrameAnchor{parent_frame, external.time,
                                   external.q_odom_imu, external.position};
      if (parent_frame.empty()) {
        ROS_WARN("reflective mapper anchored without an external odom frame; "
                 "map TF will not be published");
      } else if (parent_frame == config_.map_frame) {
        ROS_WARN("reflective mapper external frame equals map frame (%s); "
                 "map TF will not be published",
                 config_.map_frame.c_str());
      } else {
        ROS_INFO("reflective mapper anchored %s -> %s at t=%.6f",
                 parent_frame.c_str(), config_.map_frame.c_str(), external.time);
      }
    } else if (!parent_frame.empty() &&
               !map_anchor_->parent_frame.empty() &&
               parent_frame != map_anchor_->parent_frame) {
      ROS_WARN_THROTTLE(2.0,
                        "dropping pose because odom frame changed from %s to %s",
                        map_anchor_->parent_frame.c_str(), parent_frame.c_str());
      return false;
    }

    local->time = external.time;
    const Eigen::Quaterniond q_reflective_parent =
        map_anchor_->q_parent_reflective.conjugate();
    local->q_odom_imu =
        (q_reflective_parent * external.q_odom_imu).normalized();
    local->position =
        q_reflective_parent *
        (external.position - map_anchor_->p_parent_reflective);
    PublishMapFrameTf();
    return true;
  }

  void HandlePose(const nav_msgs::Odometry::ConstPtr& message) {
    const double time = message->header.stamp.toSec();
    Eigen::Quaterniond orientation(message->pose.pose.orientation.w,
                                   message->pose.pose.orientation.x,
                                   message->pose.pose.orientation.y,
                                   message->pose.pose.orientation.z);
    const Eigen::Vector3d position(message->pose.pose.position.x,
                                   message->pose.pose.position.y,
                                   message->pose.pose.position.z);
    if (!IsFinite(time) || !IsFinite(position) || orientation.norm() < 1e-8) return;
    orientation.normalize();
    if (pending_poses_.size() == static_cast<size_t>(config_.max_pending_poses)) {
      pending_poses_.pop_front();
      ++dropped_pose_overflow_;
    }
    PoseCorrection correction;
    correction.time = time;
    if (config_.pose_input_frame == "lidar") {
      const Eigen::Matrix3d rotation_odom_lidar = orientation.toRotationMatrix();
      const Eigen::Matrix3d rotation_odom_imu =
          rotation_odom_lidar * imu_from_lidar_rotation_.transpose();
      correction.q_odom_imu = Eigen::Quaterniond(rotation_odom_imu).normalized();
      correction.position = position - rotation_odom_imu * imu_from_lidar_translation_;
    } else {
      correction.q_odom_imu = orientation;
      correction.position = position;
    }
    std::string parent_frame = config_.external_odom_frame.empty()
                                   ? message->header.frame_id
                                   : config_.external_odom_frame;
	    PoseCorrection local_correction;
	    if (!ConvertExternalPoseToMapFrame(correction, parent_frame,
	                                       &local_correction)) {
	      return;
	    }
    const Eigen::Quaterniond q_map_lidar =
        local_correction.q_odom_imu * Eigen::Quaterniond(imu_from_lidar_rotation_);
    const Eigen::Vector3d p_map_lidar =
        local_correction.q_odom_imu * imu_from_lidar_translation_ +
        local_correction.position;
    ROS_INFO_THROTTLE(
        2.0,
        "pose input stamp=%.6f parent=%s child=%s pose_input_frame=%s raw_p=[%.3f %.3f %.3f] "
        "interpreted_imu_p=[%.3f %.3f %.3f] local_imu_p=[%.3f %.3f %.3f] "
        "local_lidar_p=[%.3f %.3f %.3f] q_norm=%.6f pending_poses=%zu "
        "odom_samples=%zu",
        time, parent_frame.c_str(), message->child_frame_id.c_str(),
        config_.pose_input_frame.c_str(), position.x(), position.y(), position.z(),
        correction.position.x(), correction.position.y(), correction.position.z(),
        local_correction.position.x(), local_correction.position.y(),
        local_correction.position.z(), p_map_lidar.x(), p_map_lidar.y(),
        p_map_lidar.z(), q_map_lidar.norm(), pending_poses_.size(),
        odom_buffer_.size());
	    PublishPathSinceStart(local_correction);
    std::string segment_reason;
    if (odom_buffer_.Push(local_correction, &segment_reason)) {
      ROS_WARN(
          "ODOM_SEGMENT_EVENT reason=%s stamp=%.6f buffer_size=%zu "
          "local_imu_p=[%.3f %.3f %.3f]",
          segment_reason.c_str(), local_correction.time, odom_buffer_.size(),
          local_correction.position.x(), local_correction.position.y(),
          local_correction.position.z());
    }
    if (config_.trajectory_mode != "odom_poseat") {
      pending_poses_.push_back(local_correction);
    }
	  }

  void Worker(const ros::TimerEvent&) {
    while (!scan_order_.empty()) {
      const uint32_t scan_slot = scan_order_.front();
      PendingScan& scan = pending_scans_[scan_slot];
      if (!scan.active) {
        scan_order_.pop_front();
        continue;
      }
      if (config_.trajectory_mode == "odom_poseat") {
        if (ProcessScanWithOdomPoseAt(scan)) {
          continue;
        }
        break;
      }
      const std::optional<size_t> pose_index = FindMatchingPose(scan);
      if (!pose_index.has_value()) {
        DiscardStalePoses(scan);
        if (EarliestPoseIsAfterScan(scan)) {
          ROS_WARN_THROTTLE(
              2.0,
              "dropping scan because available poses have passed it: scan_end=%.6f "
              "first_pose=%.6f tolerance=%.3fs",
              scan.imu_end_time, pending_poses_.front().time,
              config_.pose_match_max_time_diff_sec);
          DropOldestScan();
          ++dropped_pose_mismatch_;
          continue;
        }
        if (latest_imu_time_ > scan.imu_end_time + config_.pose_timeout_sec) {
          ROS_WARN_THROTTLE(
              2.0,
              "dropping scan without matching pose: scan=[%.6f, %.6f] latest_imu=%.6f "
              "pending_poses=%zu timeout=%.2fs",
              scan.imu_start_time, scan.imu_end_time, latest_imu_time_,
              pending_poses_.size(), config_.pose_timeout_sec);
          DropOldestScan();
          ++dropped_pose_timeout_;
          continue;
        }
        const double first_pose_time = pending_poses_.empty()
            ? std::numeric_limits<double>::quiet_NaN()
            : pending_poses_.front().time;
        const double last_pose_time = pending_poses_.empty()
            ? std::numeric_limits<double>::quiet_NaN()
            : pending_poses_.back().time;
        ROS_DEBUG_THROTTLE(
            2.0,
            "waiting for matching pose: scan=[%.6f, %.6f] latest_imu=%.6f "
            "pending_poses=%zu pose_range=[%.6f, %.6f]",
            scan.imu_start_time, scan.imu_end_time, latest_imu_time_,
            pending_poses_.size(), first_pose_time, last_pose_time);
        break;
      }
      const PoseCorrection correction = pending_poses_[*pose_index];
      if (!IsExternallyContinuous(correction)) {
        pending_poses_.erase(pending_poses_.begin() + *pose_index);
        ResetEpoch();
        continue;
      }
      if (!motion_.initialized()) {
        if (!motion_.Initialize(correction)) {
          if (latest_imu_time_ > scan.imu_end_time + config_.pose_timeout_sec) {
            pending_poses_.erase(pending_poses_.begin() + *pose_index);
            DropOldestScan();
            ++dropped_missing_imu_;
            continue;
          }
          break;
        }
        last_external_pose_ = correction;
        pending_poses_.erase(pending_poses_.begin() + *pose_index);
        DropOldestScan();
        ++initialization_scans_;
        continue;
      }
      NominalState reference_state;
	      if (!motion_.StateAt(scan.imu_end_time, &scratch_end_) ||
	          !motion_.StateAt(correction.time, &reference_state)) {
	        if (latest_imu_time_ > scan.imu_end_time + config_.pose_timeout_sec) {
	          pending_poses_.erase(pending_poses_.begin() + *pose_index);
	          DropOldestScan();
          ++dropped_missing_imu_;
          continue;
	        }
	        break;
	      }
      const double pose_to_scan_end = correction.time - scan.imu_end_time;
      const double pose_residual_translation =
          (correction.position - reference_state.position).norm();
      const double pose_residual_rotation =
          RotationDistance(reference_state.q_odom_imu, correction.q_odom_imu);
      const double predicted_end_translation_error =
          (correction.position - scratch_end_.position).norm();
      const double predicted_end_rotation_error =
          RotationDistance(scratch_end_.q_odom_imu, correction.q_odom_imu);
      ROS_INFO_THROTTLE(
          2.0,
          "scan ready id=%llu duration=%.3fms pose_dt_to_end=%.3fms imu_margin=%.3fms "
          "pose_residual_at_pose_time=[%.4fm %.4frad] predicted_end_vs_pose=[%.4fm %.4frad] "
          "pending(scans=%zu poses=%zu) committed=%llu drops(timeout=%llu mismatch=%llu missing_imu=%llu)",
          static_cast<unsigned long long>(scan.scan_id),
          (scan.imu_end_time - scan.imu_start_time) * 1000.0,
          pose_to_scan_end * 1000.0,
          (latest_imu_time_ - scan.imu_end_time) * 1000.0,
          pose_residual_translation, pose_residual_rotation,
          predicted_end_translation_error, predicted_end_rotation_error,
          scan_order_.size(), pending_poses_.size(),
          static_cast<unsigned long long>(committed_scans_),
          static_cast<unsigned long long>(dropped_pose_timeout_),
          static_cast<unsigned long long>(dropped_pose_mismatch_),
          static_cast<unsigned long long>(dropped_missing_imu_));
      if (std::abs(pose_to_scan_end) > 0.03 ||
          pose_residual_translation > 0.05 || pose_residual_rotation > 0.03) {
        ROS_WARN_THROTTLE(
            2.0,
            "trajectory diagnostic: scan=%llu pose_dt=%.3fms residual=[%.4fm %.4frad] "
            "pred_end_vs_pose=[%.4fm %.4frad]",
            static_cast<unsigned long long>(scan.scan_id),
            pose_to_scan_end * 1000.0, pose_residual_translation,
            pose_residual_rotation, predicted_end_translation_error,
            predicted_end_rotation_error);
      }
	      CommitScan(scan, correction, reference_state, last_external_pose_);
	      if (!motion_.ApplyPoseCorrection(correction)) {
	        ROS_ERROR_THROTTLE(1.0, "delayed pose replay failed after map insertion");
	      }
      last_external_pose_ = correction;
      pending_poses_.erase(pending_poses_.begin() + *pose_index);
      DropOldestScan();
      ++committed_scans_;
      MarkMapForPublish(correction);
    }
    ROS_INFO_THROTTLE(2.0,
        "reflective map voxels=%zu elements(total=%zu mature=%u candidate=%u) "
        "all_high_history=%zu "
        "committed=%llu initialized=%llu pending_scans=%zu "
        "pending_poses=%zu odom_samples=%zu drops(timeout=%llu mismatch=%llu missing_imu=%llu "
        "stale_pose=%llu scan_overflow=%llu pose_overflow=%llu odom_timeout=%llu "
        "odom_gap=%llu odom_jump=%llu time_invalid=%llu empty=%llu poseat_invalid=%llu)",
        config_.enable_map_element_filter ? map_element_filter_.published_count()
                                          : map_.occupied_count(),
        map_element_filter_.element_count(),
        map_element_filter_.latest_stats().mature_count,
        map_element_filter_.latest_stats().candidate_count,
        all_high_ref_history_.size(),
        static_cast<unsigned long long>(committed_scans_),
        static_cast<unsigned long long>(initialization_scans_),
        scan_order_.size(), pending_poses_.size(), odom_buffer_.size(),
        static_cast<unsigned long long>(dropped_pose_timeout_),
        static_cast<unsigned long long>(dropped_pose_mismatch_),
        static_cast<unsigned long long>(dropped_missing_imu_),
        static_cast<unsigned long long>(dropped_stale_pose_),
        static_cast<unsigned long long>(dropped_scan_overflow_),
        static_cast<unsigned long long>(dropped_pose_overflow_),
        static_cast<unsigned long long>(dropped_odom_wait_timeout_),
        static_cast<unsigned long long>(dropped_odom_gap_),
        static_cast<unsigned long long>(dropped_odom_jump_),
        static_cast<unsigned long long>(dropped_time_invalid_),
        static_cast<unsigned long long>(dropped_empty_scan_),
        static_cast<unsigned long long>(dropped_poseat_invalid_));
  }

  bool ProcessScanWithOdomPoseAt(PendingScan& scan) {
    if (scan.points.empty()) {
      ROS_INFO_THROTTLE(
          2.0,
          "SCAN_TRACE id=%llu state=SCAN_EMPTY raw=%u high=%u action=DROP",
          static_cast<unsigned long long>(scan.scan_id), scan.raw_point_count,
          scan.accepted_point_count);
      DropOldestScan();
      ++dropped_empty_scan_;
      return true;
    }

    const double scan_begin = scan.accepted_imu_start_time +
                              config_.odom_time_offset_sec;
    const double scan_end = scan.accepted_imu_end_time +
                            config_.odom_time_offset_sec;
    const OdomCoverageResult coverage =
        odom_buffer_.CheckCoverage(scan_begin, scan_end,
                                   config_.max_odom_bracket_gap_sec);
    if (!coverage.valid) {
      const double latest_odom = odom_buffer_.latest_time();
      if (coverage.has_future_odom &&
          latest_odom <= scan_end + config_.pose_timeout_sec) {
        if (!scan.trace_wait_logged) {
          ROS_DEBUG(
              "SCAN_TRACE id=%llu state=WAIT_ODOM reason=%s high=%zu "
              "scan=[%.6f,%.6f] query=[%.6f,%.6f] latest_odom=%.6f "
              "odom_samples=%zu pending_scans=%zu",
              static_cast<unsigned long long>(scan.scan_id), coverage.reason,
              scan.points.size(), scan.imu_start_time, scan.imu_end_time,
              scan_begin, scan_end, latest_odom, odom_buffer_.size(),
              scan_order_.size());
          scan.trace_wait_logged = true;
        }
        return false;
      }

      if (coverage.has_future_odom) {
        ++dropped_odom_wait_timeout_;
      } else if (coverage.crossed_jump ||
                 std::string(coverage.reason) == "CROSSED_SEGMENT") {
        ++dropped_odom_jump_;
      } else if (std::string(coverage.reason) == "TIME_INVALID") {
        ++dropped_time_invalid_;
      } else {
        ++dropped_odom_gap_;
      }
      ROS_WARN(
          "SCAN_TRACE id=%llu state=SKIPPED reason=%s high=%zu "
          "scan=[%.6f,%.6f] query=[%.6f,%.6f] latest_odom=%.6f "
          "odom_samples=%zu max_gap=%.4f future=%d crossed_jump=%d action=DROP",
          static_cast<unsigned long long>(scan.scan_id), coverage.reason,
          scan.points.size(), scan.imu_start_time, scan.imu_end_time,
          scan_begin, scan_end, latest_odom, odom_buffer_.size(),
          coverage.max_gap, coverage.has_future_odom ? 1 : 0,
          coverage.crossed_jump ? 1 : 0);
      DropOldestScan();
      return true;
    }

    const OdomPoseQuery end_pose = odom_buffer_.PoseAt(scan_end);
    if (!end_pose.valid) {
      ROS_WARN(
          "SCAN_TRACE id=%llu state=SKIPPED reason=END_POSEAT_INVALID "
          "query_end=%.6f action=DROP",
          static_cast<unsigned long long>(scan.scan_id), scan_end);
      DropOldestScan();
      ++dropped_poseat_invalid_;
      return true;
    }

    ROS_INFO_THROTTLE(
        2.0,
        "SCAN_TRACE id=%llu state=SCAN_READY trajectory_mode=odom_poseat "
        "high=%zu duration=%.3fms query=[%.6f,%.6f] coverage_gap=%.3fms "
        "segment=%llu odom_samples=%zu pending_scans=%zu",
        static_cast<unsigned long long>(scan.scan_id), scan.points.size(),
        (scan.accepted_imu_end_time - scan.accepted_imu_start_time) * 1000.0,
        scan_begin, scan_end, coverage.max_gap * 1000.0,
        static_cast<unsigned long long>(coverage.segment_id),
        odom_buffer_.size(), scan_order_.size());
    CommitScanOdomPoseAt(scan, coverage, end_pose.pose);
    DropOldestScan();
    ++committed_scans_;
    MarkMapForPublish(end_pose.pose);
    return true;
  }

  std::optional<size_t> FindMatchingPose(const PendingScan& scan) const {
    std::optional<size_t> match;
    double closest_to_scan_end = std::numeric_limits<double>::infinity();
    const double target_time = scan.imu_end_time;
    for (size_t i = 0; i < pending_poses_.size(); ++i) {
      const double time = pending_poses_[i].time;
      if (time > target_time + config_.pose_match_max_future_sec) {
        break;
      }
      if (time >= target_time - config_.pose_match_max_time_diff_sec) {
        // LIO odometry is normally stamped at scan end.  A Livox frame spans
        // about 100 ms and can contain two valid 10 Hz odometry samples, so
        // picking the first queued sample produces a full-scan pose error.
        const double distance_to_end = std::abs(time - target_time);
        if (distance_to_end < closest_to_scan_end) {
          closest_to_scan_end = distance_to_end;
          match = i;
        }
      }
    }
    return match;
  }

  void DiscardStalePoses(const PendingScan& scan) {
    const double oldest_useful_pose =
        scan.imu_end_time - config_.pose_match_max_time_diff_sec;
    while (!pending_poses_.empty() &&
           pending_poses_.front().time < oldest_useful_pose) {
      pending_poses_.pop_front();
      ++dropped_stale_pose_;
    }
  }

  bool EarliestPoseIsAfterScan(const PendingScan& scan) const {
    return !pending_poses_.empty() &&
           pending_poses_.front().time >
               scan.imu_end_time + config_.pose_match_max_future_sec;
  }

  bool IsExternallyContinuous(const PoseCorrection& correction) const {
    if (!last_external_pose_.has_value()) return true;
    const double dt = correction.time - last_external_pose_->time;
    if (dt <= 0.0) return false;
    const double position_delta = (correction.position - last_external_pose_->position).norm();
    const double rotation_delta = RotationDistance(last_external_pose_->q_odom_imu,
                                                   correction.q_odom_imu);
    return position_delta <= config_.max_external_velocity_mps * dt +
                                 config_.external_position_margin_m &&
           rotation_delta <= config_.max_external_angular_velocity_rps * dt +
                                 config_.external_rotation_margin_rad;
  }

  VoxelKey ObservationKey(const Eigen::Vector3f& point) const {
    const float size = static_cast<float>(config_.scan_observation_voxel_size_m);
    return VoxelKey{static_cast<int32_t>(std::floor(point.x() / size)),
                    static_cast<int32_t>(std::floor(point.y() / size)),
                    static_cast<int32_t>(std::floor(point.z() / size))};
  }

  void AddScanObservation(const MapObservation& observation,
                          std::vector<MapObservation>* observations) const {
    const VoxelKey key = ObservationKey(observation.position);
    for (MapObservation& existing : *observations) {
      if (ObservationKey(existing.position) == key) {
        const uint32_t old_count = existing.count;
        const uint32_t new_count = old_count + observation.count;
        const float old_weight = static_cast<float>(old_count);
        const float new_weight = static_cast<float>(observation.count);
        existing.position =
            (existing.position * old_weight + observation.position * new_weight) /
            static_cast<float>(new_count);
        existing.lidar_reference =
            (existing.lidar_reference * old_weight +
             observation.lidar_reference * new_weight) /
            static_cast<float>(new_count);
        existing.intensity = std::max(existing.intensity, observation.intensity);
        existing.count = new_count;
        return;
      }
    }
    observations->push_back(observation);
  }

  void CommitScanOdomPoseAt(const PendingScan& scan,
                            const OdomCoverageResult& coverage,
                            const PoseCorrection& reference_pose) {
    if (!config_.enable_map_element_filter) map_.Advance(scan.end_tick);
    const MapStats stats_before = map_.stats();
    const size_t occupied_before = map_.occupied_count();
    uint32_t committed_point_count = 0;
    uint32_t poseat_invalid_count = 0;
    uint32_t nonfinite_deskew_count = 0;
    uint32_t exact_query_count = 0;
    uint32_t interpolated_query_count = 0;
    double max_query_gap = 0.0;
    double min_query_alpha = std::numeric_limits<double>::infinity();
    double max_query_alpha = -std::numeric_limits<double>::infinity();

    std::vector<DeskewedHighPoint> deskewed_points;
    if (config_.publish_deskewed_high_ref_cloud) {
      deskewed_points.reserve(scan.points.size());
    }
    std::vector<MapObservation> map_observations;
    map_observations.reserve(scan.points.size());
    std::vector<DeskewedHighPoint> observation_points_lidar;
    bool observation_truncated = false;
    Eigen::Vector3f observation_map_min =
        Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector3f observation_map_max =
        Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
    if (config_.publish_observation_frame &&
        config_.max_observation_points_per_frame > 0) {
      observation_points_lidar.reserve(std::min<size_t>(
          scan.points.size(),
          static_cast<size_t>(config_.max_observation_points_per_frame)));
    }

    const Eigen::Matrix3d reference_rotation =
        reference_pose.q_odom_imu.toRotationMatrix() * imu_from_lidar_rotation_;
    const Eigen::Vector3d reference_translation =
        reference_pose.q_odom_imu * imu_from_lidar_translation_ +
        reference_pose.position;

    for (const PendingPoint& point : scan.points) {
      const double point_imu_time = scan.lidar_base_time +
          point.offset_time * config_.offset_time_scale_sec +
          config_.lidar_to_imu_time_offset_sec +
          config_.odom_time_offset_sec;
      const OdomPoseQuery query = odom_buffer_.PoseAt(point_imu_time);
      if (!query.valid) {
        ++poseat_invalid_count;
        ++dropped_poseat_invalid_;
        continue;
      }
      if (std::string(query.mode) == "exact") {
        ++exact_query_count;
      } else {
        ++interpolated_query_count;
      }
      max_query_gap = std::max(max_query_gap, query.right_stamp - query.left_stamp);
      min_query_alpha = std::min(min_query_alpha, query.alpha);
      max_query_alpha = std::max(max_query_alpha, query.alpha);

      const Eigen::Vector3d point_lidar(point.x, point.y, point.z);
      const Eigen::Vector3d point_imu =
          imu_from_lidar_rotation_ * point_lidar + imu_from_lidar_translation_;
      const Eigen::Vector3d point_local =
          query.pose.q_odom_imu * point_imu + query.pose.position;
      if (!IsFinite(point_local)) {
        ++nonfinite_deskew_count;
        continue;
      }
      const Eigen::Vector3d point_lidar_reference =
          reference_rotation.transpose() * (point_local - reference_translation);
      const Eigen::Vector3f point_map = point_local.cast<float>();
      ++committed_point_count;
      const DeskewedHighPoint high_history_point{
          point_map, static_cast<float>(point.reflectivity)};
      if (config_.publish_deskewed_high_ref_cloud) {
        deskewed_points.push_back(high_history_point);
      }
      AppendAllHighRefHistory(high_history_point);
      AddScanObservation(
          MapObservation{point_map, point_lidar_reference.cast<float>(),
                         static_cast<float>(point.reflectivity), 1},
          &map_observations);
    }

    MapElementUpdateStats element_stats;
    if (config_.enable_map_element_filter) {
      element_stats = map_element_filter_.UpdateScan(map_observations, scan.scan_id);
    } else {
      for (const MapObservation& observation : map_observations) {
        map_.Insert(observation.position, scan.scan_id, scan.end_tick,
                    static_cast<uint8_t>(std::min(255.0f, observation.intensity)));
      }
    }

    if (config_.publish_observation_frame) {
      for (const MapObservation& observation : map_observations) {
        if (observation_points_lidar.size() <
            static_cast<size_t>(config_.max_observation_points_per_frame)) {
          observation_points_lidar.push_back(
              DeskewedHighPoint{observation.lidar_reference,
                                observation.intensity});
          observation_map_min = observation_map_min.cwiseMin(observation.position);
          observation_map_max = observation_map_max.cwiseMax(observation.position);
        } else {
          observation_truncated = true;
          ++dropped_observation_point_budget_;
        }
      }
    }

    if (config_.publish_deskewed_high_ref_cloud) {
      PublishDeskewedHighRefCloud(reference_pose.time, deskewed_points);
    }
    if (config_.publish_all_high_ref_history) {
      PublishAllHighRefHistoryCloud(reference_pose.time);
    }
    if (config_.publish_observation_frame) {
      PublishObservationFrame(reference_pose, observation_points_lidar,
                              observation_map_min, observation_map_max,
                              observation_truncated);
    }
    const MapStats stats_after = map_.stats();
    const Eigen::Vector3f log_map_min = observation_points_lidar.empty()
        ? Eigen::Vector3f::Zero() : observation_map_min;
    const Eigen::Vector3f log_map_max = observation_points_lidar.empty()
        ? Eigen::Vector3f::Zero() : observation_map_max;
    const double log_min_alpha =
        std::isfinite(min_query_alpha) ? min_query_alpha : 0.0;
    const double log_max_alpha =
        std::isfinite(max_query_alpha) ? max_query_alpha : 0.0;
    ROS_INFO_THROTTLE(
        2.0,
        "deskew/map scan=%llu deskew_mode=odom_poseat input_high=%zu committed_points=%u "
        "poseat_invalid=%u nonfinite_deskew=%u query(exact=%u interp=%u max_gap=%.3fms "
        "alpha=[%.3f,%.3f]) coverage(segment=%llu max_gap=%.3fms) "
        "scan_obs=%zu element(created=%u fused=%u matured=%u mature=%u candidate=%u "
        "rms=[%.4f %.4f %.4f] cap_drop=%u) "
        "map_delta(insert=%llu update=%llu merge_reject=%llu occ_delta=%lld) occupied=%zu "
        "obs_points=%zu obs_truncated=%d aabb_min=[%.3f %.3f %.3f] "
        "aabb_max=[%.3f %.3f %.3f]",
        static_cast<unsigned long long>(scan.scan_id), scan.points.size(),
        committed_point_count, poseat_invalid_count, nonfinite_deskew_count,
        exact_query_count, interpolated_query_count, max_query_gap * 1000.0,
        log_min_alpha, log_max_alpha,
        static_cast<unsigned long long>(coverage.segment_id),
        coverage.max_gap * 1000.0,
        map_observations.size(), element_stats.created, element_stats.fused,
        element_stats.matured, element_stats.mature_count,
        element_stats.candidate_count, element_stats.mature_rms_p50,
        element_stats.mature_rms_p95, element_stats.mature_rms_max,
        element_stats.dropped_capacity,
        static_cast<unsigned long long>(stats_after.insertions - stats_before.insertions),
        static_cast<unsigned long long>(stats_after.updates - stats_before.updates),
        static_cast<unsigned long long>(stats_after.merge_rejects - stats_before.merge_rejects),
        static_cast<long long>(map_.occupied_count()) -
            static_cast<long long>(occupied_before),
        config_.enable_map_element_filter ? map_element_filter_.published_count()
                                          : map_.occupied_count(),
        observation_points_lidar.size(),
        observation_truncated ? 1 : 0,
        log_map_min.x(), log_map_min.y(), log_map_min.z(),
        log_map_max.x(), log_map_max.y(), log_map_max.z());
  }

	  void CommitScan(const PendingScan& scan, const PoseCorrection& correction,
	                  const NominalState& reference_state,
	                  const std::optional<PoseCorrection>& start_correction) {
	    map_.Advance(scan.end_tick);
    const MapStats stats_before = map_.stats();
    const size_t occupied_before = map_.occupied_count();
    uint32_t committed_point_count = 0;
    uint32_t missing_state_count = 0;
    uint32_t nonfinite_deskew_count = 0;
		    std::vector<DeskewedHighPoint> deskewed_points;
		    if (config_.publish_deskewed_high_ref_cloud) {
		      deskewed_points.reserve(scan.points.size());
		    }
	    std::vector<DeskewedHighPoint> observation_points_lidar;
	    bool observation_truncated = false;
	    Eigen::Vector3f observation_map_min =
	        Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
	    Eigen::Vector3f observation_map_max =
	        Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
	    if (config_.publish_observation_frame &&
	        config_.max_observation_points_per_frame > 0) {
	      observation_points_lidar.reserve(std::min<size_t>(
	          scan.points.size(),
	          static_cast<size_t>(config_.max_observation_points_per_frame)));
	    }
    const Eigen::Matrix3d predicted_reference_rotation =
        reference_state.q_odom_imu.toRotationMatrix() * imu_from_lidar_rotation_;
    const Eigen::Vector3d predicted_reference_translation =
        reference_state.q_odom_imu * imu_from_lidar_translation_ +
        (config_.deskew_use_imu_translation ? reference_state.position
                                            : Eigen::Vector3d::Zero());
    const Eigen::Matrix3d external_rotation =
        correction.q_odom_imu.toRotationMatrix() * imu_from_lidar_rotation_;
    const Eigen::Vector3d external_translation =
        correction.q_odom_imu * imu_from_lidar_translation_ + correction.position;
    const bool use_external_interp =
        config_.deskew_use_external_pose_interpolation &&
        start_correction.has_value() &&
        correction.time > start_correction->time;

    for (const PendingPoint& point : scan.points) {
      const double point_imu_time = scan.lidar_base_time +
          point.offset_time * config_.offset_time_scale_sec +
          config_.lidar_to_imu_time_offset_sec;
      const Eigen::Vector3d point_lidar(point.x, point.y, point.z);
      Eigen::Vector3d point_local;
      Eigen::Vector3d point_lidar_reference;
      if (use_external_interp) {
        const double ratio = std::max(
            0.0, std::min(1.0, (point_imu_time - start_correction->time) /
                                   (correction.time - start_correction->time)));
        const Eigen::Quaterniond q_interp =
            start_correction->q_odom_imu.slerp(ratio, correction.q_odom_imu);
        const Eigen::Vector3d p_interp =
            start_correction->position +
            ratio * (correction.position - start_correction->position);
        point_local = q_interp * (imu_from_lidar_rotation_ * point_lidar +
                                  imu_from_lidar_translation_) + p_interp;
        point_lidar_reference =
            external_rotation.transpose() * (point_local - external_translation);
      } else {
        NominalState point_state;
        if (!motion_.StateAt(point_imu_time, &point_state)) {
          ++dropped_point_missing_state_;
          ++missing_state_count;
          continue;
        }
        const Eigen::Vector3d point_world_predicted =
            point_state.q_odom_imu * (imu_from_lidar_rotation_ * point_lidar +
                                      imu_from_lidar_translation_) +
            (config_.deskew_use_imu_translation ? point_state.position
                                                : Eigen::Vector3d::Zero());
        point_lidar_reference =
            predicted_reference_rotation.transpose() *
            (point_world_predicted - predicted_reference_translation);
        point_local = external_rotation * point_lidar_reference + external_translation;
      }
	      if (!IsFinite(point_local)) {
        ++nonfinite_deskew_count;
        continue;
      }
	      const Eigen::Vector3f point_map = point_local.cast<float>();
	      map_.Insert(point_map, scan.scan_id, scan.end_tick, point.reflectivity);
      ++committed_point_count;
      const DeskewedHighPoint high_history_point{
          point_map, static_cast<float>(point.reflectivity)};
	      if (config_.publish_deskewed_high_ref_cloud) {
        deskewed_points.push_back(high_history_point);
	      }
      AppendAllHighRefHistory(high_history_point);
	      if (config_.publish_observation_frame) {
	        if (observation_points_lidar.size() <
	            static_cast<size_t>(config_.max_observation_points_per_frame)) {
	          observation_points_lidar.push_back(
	              DeskewedHighPoint{point_lidar_reference.cast<float>(),
	                                static_cast<float>(point.reflectivity)});
	          observation_map_min = observation_map_min.cwiseMin(point_map);
	          observation_map_max = observation_map_max.cwiseMax(point_map);
	        } else {
	          observation_truncated = true;
	          ++dropped_observation_point_budget_;
	        }
	      }
	    }
	    if (config_.publish_deskewed_high_ref_cloud) {
	      PublishDeskewedHighRefCloud(correction.time, deskewed_points);
	    }
    if (config_.publish_all_high_ref_history) {
      PublishAllHighRefHistoryCloud(correction.time);
    }
		    if (config_.publish_observation_frame) {
		      PublishObservationFrame(correction, observation_points_lidar,
		                              observation_map_min, observation_map_max,
		                              observation_truncated);
		    }
    const MapStats stats_after = map_.stats();
    const Eigen::Vector3f log_map_min = observation_points_lidar.empty()
        ? Eigen::Vector3f::Zero() : observation_map_min;
    const Eigen::Vector3f log_map_max = observation_points_lidar.empty()
        ? Eigen::Vector3f::Zero() : observation_map_max;
    ROS_INFO_THROTTLE(
        2.0,
        "deskew/map scan=%llu deskew_mode=%s trans_deskew=%s input_high=%zu committed_points=%u "
        "missing_state=%u nonfinite_deskew=%u "
        "map_delta(insert=%llu update=%llu merge_reject=%llu occ_delta=%lld) occupied=%zu "
        "obs_points=%zu obs_truncated=%d aabb_min=[%.3f %.3f %.3f] aabb_max=[%.3f %.3f %.3f]",
        static_cast<unsigned long long>(scan.scan_id),
        use_external_interp ? "external_interp" : "imu_trajectory",
        config_.deskew_use_imu_translation ? "imu" : "off",
        scan.points.size(),
        committed_point_count, missing_state_count, nonfinite_deskew_count,
        static_cast<unsigned long long>(stats_after.insertions - stats_before.insertions),
        static_cast<unsigned long long>(stats_after.updates - stats_before.updates),
        static_cast<unsigned long long>(stats_after.merge_rejects - stats_before.merge_rejects),
        static_cast<long long>(map_.occupied_count()) -
            static_cast<long long>(occupied_before),
        map_.occupied_count(), observation_points_lidar.size(),
        observation_truncated ? 1 : 0,
        log_map_min.x(), log_map_min.y(), log_map_min.z(),
        log_map_max.x(), log_map_max.y(), log_map_max.z());
		  }

  uint32_t AllocateScanSlot() {
    if (scan_order_.size() == pending_scans_.size()) {
      DropOldestScan();
      ++dropped_scan_overflow_;
    }
    for (uint32_t i = 0; i < pending_scans_.size(); ++i) {
      if (!pending_scans_[i].active) return i;
    }
    return 0;
  }

  void DropOldestScan() {
    if (scan_order_.empty()) return;
    PendingScan& scan = pending_scans_[scan_order_.front()];
    map_.Advance(scan.end_tick);
    scan.active = false;
    scan.points.clear();
    scan_order_.pop_front();
  }

  void AppendAllHighRefHistory(const DeskewedHighPoint& point) {
    if (!config_.publish_all_high_ref_history) return;
    all_high_ref_history_.push_back(point);
    const int max_points = config_.max_all_high_ref_history_points;
    if (max_points > 0 &&
        all_high_ref_history_.size() > static_cast<size_t>(max_points)) {
      const size_t overflow =
          all_high_ref_history_.size() - static_cast<size_t>(max_points);
      all_high_ref_history_.erase(all_high_ref_history_.begin(),
                                  all_high_ref_history_.begin() + overflow);
    }
  }

  void ResetEpoch() {
    map_.Clear();
    map_element_filter_.Clear();
    all_high_ref_history_.clear();
    motion_.Reset();
    odom_buffer_.Clear();
    latest_imu_time_ = -std::numeric_limits<double>::infinity();
    last_external_pose_.reset();
    map_anchor_.reset();
    map_frame_tf_published_ = false;
    path_since_start_.poses.clear();
    pending_poses_.clear();
    while (!scan_order_.empty()) {
      pending_scans_[scan_order_.front()].active = false;
      pending_scans_[scan_order_.front()].points.clear();
      scan_order_.pop_front();
    }
    ++epoch_resets_;
    ROS_WARN("reflective mapper reset after discontinuous external pose");
  }

  void InitializeMapMessage() {
    map_message_.height = 1;
    map_message_.is_bigendian = false;
    map_message_.is_dense = true;
    // This is the mapper's compact snapshot interface.  Intensity and evidence
    // are consumed by the separate candidate extractor; RViz only needs xyz.
    map_message_.point_step = 20;
    sensor_msgs::PointField field;
    field.count = 1;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.name = "x";
    field.offset = 0;
    map_message_.fields.push_back(field);
    field.name = "y";
    field.offset = 4;
    map_message_.fields.push_back(field);
    field.name = "z";
    field.offset = 8;
    map_message_.fields.push_back(field);
    field.name = "intensity";
    field.offset = 12;
    map_message_.fields.push_back(field);
    field.name = "evidence";
    field.offset = 16;
    map_message_.fields.push_back(field);
  }

  sensor_msgs::PointCloud2 MakeXyzIntensityCloud(
      const std_msgs::Header& header,
      const std::vector<DeskewedHighPoint>& points) const {
    sensor_msgs::PointCloud2 cloud;
    cloud.header = header;
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(points.size());
    cloud.is_bigendian = false;
    cloud.is_dense = true;
    cloud.point_step = 16;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step);

    sensor_msgs::PointField field;
    field.count = 1;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.name = "x";
    field.offset = 0;
    cloud.fields.push_back(field);
    field.name = "y";
    field.offset = 4;
    cloud.fields.push_back(field);
    field.name = "z";
    field.offset = 8;
    cloud.fields.push_back(field);
    field.name = "intensity";
    field.offset = 12;
    cloud.fields.push_back(field);

    for (size_t i = 0; i < points.size(); ++i) {
      uint8_t* destination = cloud.data.data() + i * cloud.point_step;
      std::memcpy(destination, &points[i].position.x(), sizeof(float));
      std::memcpy(destination + 4, &points[i].position.y(), sizeof(float));
      std::memcpy(destination + 8, &points[i].position.z(), sizeof(float));
      std::memcpy(destination + 12, &points[i].intensity, sizeof(float));
    }
    return cloud;
  }

	  void PublishDeskewedHighRefCloud(
	      double stamp_sec, const std::vector<DeskewedHighPoint>& points) {
    if (!deskewed_high_ref_cloud_publisher_) return;
    std_msgs::Header header;
    header.stamp.fromSec(stamp_sec);
    header.frame_id = config_.map_frame;
    deskewed_high_ref_cloud_publisher_.publish(
	        MakeXyzIntensityCloud(header, points));
	  }

  void PublishAllHighRefHistoryCloud(double stamp_sec) {
    if (!all_high_ref_history_publisher_) return;
    std_msgs::Header header;
    header.stamp.fromSec(stamp_sec);
    header.frame_id = config_.map_frame;
    all_high_ref_history_publisher_.publish(
        MakeXyzIntensityCloud(header, all_high_ref_history_));
  }

  bool WriteHighRefHistoryPcd(
      const std::string& path,
      const std::vector<DeskewedHighPoint>& points) const {
    if (path.empty()) return true;
    std::ofstream output(path);
    if (!output.is_open()) {
      ROS_ERROR("failed to open all-high-ref PCD output path: %s", path.c_str());
      return false;
    }
    output << "# .PCD v0.7 - Point Cloud Data file format\n";
    output << "VERSION 0.7\n";
    output << "FIELDS x y z intensity\n";
    output << "SIZE 4 4 4 4\n";
    output << "TYPE F F F F\n";
    output << "COUNT 1 1 1 1\n";
    output << "WIDTH " << points.size() << "\n";
    output << "HEIGHT 1\n";
    output << "VIEWPOINT 0 0 0 1 0 0 0\n";
    output << "POINTS " << points.size() << "\n";
    output << "DATA ascii\n";
    for (const DeskewedHighPoint& point : points) {
      output << point.position.x() << " " << point.position.y() << " "
             << point.position.z() << " " << point.intensity << "\n";
    }
    return true;
  }

  void SaveAllHighRefHistoryPcd() const {
    if (!config_.save_all_high_ref_history_pcd) return;
    if (WriteHighRefHistoryPcd(config_.all_high_ref_history_pcd_path,
                               all_high_ref_history_)) {
      ROS_INFO("saved all-high-ref history PCD: path=%s points=%zu frame=%s",
               config_.all_high_ref_history_pcd_path.c_str(),
               all_high_ref_history_.size(), config_.map_frame.c_str());
    }
    if (config_.all_high_ref_history_parent_pcd_path.empty()) return;
    if (!map_anchor_.has_value()) {
      ROS_WARN("cannot save parent-frame all-high-ref PCD because map anchor is unset");
      return;
    }
    std::vector<DeskewedHighPoint> parent_points;
    parent_points.reserve(all_high_ref_history_.size());
    for (const DeskewedHighPoint& point : all_high_ref_history_) {
      parent_points.push_back(DeskewedHighPoint{
          (map_anchor_->q_parent_reflective * point.position.cast<double>() +
           map_anchor_->p_parent_reflective).cast<float>(),
          point.intensity});
    }
    if (WriteHighRefHistoryPcd(config_.all_high_ref_history_parent_pcd_path,
                               parent_points)) {
      ROS_INFO("saved all-high-ref parent-frame PCD: path=%s points=%zu frame=%s",
               config_.all_high_ref_history_parent_pcd_path.c_str(),
               parent_points.size(), map_anchor_->parent_frame.c_str());
    }
  }

	  void PublishObservationFrame(
	      const PoseCorrection& correction,
	      const std::vector<DeskewedHighPoint>& points_lidar,
	      const Eigen::Vector3f& map_min,
	      const Eigen::Vector3f& map_max,
	      bool truncated) {
	    if (!observation_frame_publisher_) return;
	    livox_reflective_marker::ReflectiveObservationFrame frame;
	    frame.header.stamp.fromSec(correction.time);
	    frame.header.frame_id = config_.lidar_frame;
	    frame.map_epoch = static_cast<uint32_t>(epoch_resets_);
	    frame.scan_id = next_observation_scan_id_++;
	    frame.observation_source_contract_hash =
	        config_.observation_source_contract_hash;

	    const Eigen::Quaterniond q_map_lidar =
	        correction.q_odom_imu * Eigen::Quaterniond(imu_from_lidar_rotation_);
	    const Eigen::Vector3d p_map_lidar =
	        correction.q_odom_imu * imu_from_lidar_translation_ +
	        correction.position;
	    frame.lidar_pose_in_map.position.x = p_map_lidar.x();
	    frame.lidar_pose_in_map.position.y = p_map_lidar.y();
	    frame.lidar_pose_in_map.position.z = p_map_lidar.z();
	    frame.lidar_pose_in_map.orientation.x = q_map_lidar.x();
	    frame.lidar_pose_in_map.orientation.y = q_map_lidar.y();
	    frame.lidar_pose_in_map.orientation.z = q_map_lidar.z();
	    frame.lidar_pose_in_map.orientation.w = q_map_lidar.w();

	    if (points_lidar.empty()) {
	      frame.map_aabb_min.x = 0.0;
	      frame.map_aabb_min.y = 0.0;
	      frame.map_aabb_min.z = 0.0;
	      frame.map_aabb_max = frame.map_aabb_min;
	    } else {
	      frame.map_aabb_min.x = map_min.x();
	      frame.map_aabb_min.y = map_min.y();
	      frame.map_aabb_min.z = map_min.z();
	      frame.map_aabb_max.x = map_max.x();
	      frame.map_aabb_max.y = map_max.y();
	      frame.map_aabb_max.z = map_max.z();
	    }
	    frame.truncated = truncated;
	    frame.cloud_lidar =
	        MakeXyzIntensityCloud(frame.header, points_lidar);
	    observation_frame_publisher_.publish(frame);
	  }

  void MarkMapForPublish(const PoseCorrection& correction) {
    latest_map_correction_ = correction;
    map_dirty_ = true;
  }

  void PublishMapTimer(const ros::TimerEvent&) {
    if ((!config_.publish_debug_map && !config_.publish_map_snapshot) ||
        !latest_map_correction_.has_value()) {
      return;
    }
    PublishMapFrameTf();
    PublishMap(*latest_map_correction_, map_dirty_);
    map_dirty_ = false;
  }

  void PublishMapFrameTf() {
    if (!map_anchor_.has_value() || map_anchor_->parent_frame.empty() ||
        map_anchor_->parent_frame == config_.map_frame) {
      return;
    }
    if (map_frame_tf_published_) return;
    geometry_msgs::TransformStamped transform;
    transform.header.stamp = ros::Time(0);
    transform.header.frame_id = map_anchor_->parent_frame;
    transform.child_frame_id = config_.map_frame;
    transform.transform.translation.x = map_anchor_->p_parent_reflective.x();
    transform.transform.translation.y = map_anchor_->p_parent_reflective.y();
    transform.transform.translation.z = map_anchor_->p_parent_reflective.z();
    transform.transform.rotation.x = map_anchor_->q_parent_reflective.x();
    transform.transform.rotation.y = map_anchor_->q_parent_reflective.y();
    transform.transform.rotation.z = map_anchor_->q_parent_reflective.z();
    transform.transform.rotation.w = map_anchor_->q_parent_reflective.w();
    static_tf_broadcaster_.sendTransform(transform);
    map_frame_tf_published_ = true;
  }

  void PublishPathSinceStart(const PoseCorrection& correction) {
    if (!path_since_start_publisher_) return;
    if (config_.max_path_poses == 0) return;

    geometry_msgs::PoseStamped pose;
    pose.header.stamp.fromSec(correction.time);
    pose.header.frame_id = config_.map_frame;
    const Eigen::Quaterniond q_map_lidar =
        correction.q_odom_imu * Eigen::Quaterniond(imu_from_lidar_rotation_);
    const Eigen::Vector3d p_map_lidar =
        correction.q_odom_imu * imu_from_lidar_translation_ + correction.position;
    pose.pose.position.x = p_map_lidar.x();
    pose.pose.position.y = p_map_lidar.y();
    pose.pose.position.z = p_map_lidar.z();
    pose.pose.orientation.x = q_map_lidar.x();
    pose.pose.orientation.y = q_map_lidar.y();
    pose.pose.orientation.z = q_map_lidar.z();
    pose.pose.orientation.w = q_map_lidar.w();

    path_since_start_.header = pose.header;
    path_since_start_.poses.push_back(pose);
    if (config_.max_path_poses > 0 &&
        path_since_start_.poses.size() >
            static_cast<size_t>(config_.max_path_poses)) {
      path_since_start_.poses.erase(path_since_start_.poses.begin());
    }
    path_since_start_publisher_.publish(path_since_start_);
  }

  void PublishMap(const PoseCorrection& correction, bool publish_snapshot) {
    const double stamp = correction.time;
    map_message_.header.stamp.fromSec(stamp);
    map_message_.header.frame_id = config_.map_frame;
    map_message_.width = static_cast<uint32_t>(
        config_.enable_map_element_filter ? map_element_filter_.published_count()
                                          : map_.occupied_count());
    map_message_.row_step = map_message_.width * map_message_.point_step;
    map_message_.data.resize(map_message_.row_step);
    size_t output_index = 0;
    if (config_.enable_map_element_filter) {
      map_element_filter_.ForEachPublished(
          [this, &output_index](const Eigen::Vector3f& position,
                                float intensity, float evidence) {
            uint8_t* destination =
                map_message_.data.data() + output_index * map_message_.point_step;
            std::memcpy(destination, &position.x(), sizeof(float));
            std::memcpy(destination + 4, &position.y(), sizeof(float));
            std::memcpy(destination + 8, &position.z(), sizeof(float));
            std::memcpy(destination + 12, &intensity, sizeof(float));
            std::memcpy(destination + 16, &evidence, sizeof(float));
            ++output_index;
          });
    } else {
      for (const VoxelNode& node : map_.nodes()) {
        if (!node.occupied) continue;
        uint8_t* destination =
            map_message_.data.data() + output_index * map_message_.point_step;
        const float intensity = static_cast<float>(node.intensity_diag);
        const float evidence = static_cast<float>(node.evidence);
        std::memcpy(destination, &node.position.x(), sizeof(float));
        std::memcpy(destination + 4, &node.position.y(), sizeof(float));
        std::memcpy(destination + 8, &node.position.z(), sizeof(float));
        std::memcpy(destination + 12, &intensity, sizeof(float));
        std::memcpy(destination + 16, &evidence, sizeof(float));
        ++output_index;
      }
    }
    if (config_.publish_debug_map) map_publisher_.publish(map_message_);
    if (config_.publish_map_snapshot && publish_snapshot) {
      livox_reflective_marker::ReflectiveMapSnapshot snapshot;
      snapshot.header = map_message_.header;
      snapshot.map_epoch = static_cast<uint32_t>(epoch_resets_);
      snapshot.voxel_size_m = static_cast<float>(
          config_.enable_map_element_filter
              ? config_.map_element_association_radius_m
              : config_.voxel_size_m);
      snapshot.lidar_frame = config_.lidar_frame;
      const Eigen::Quaterniond q_map_lidar =
          correction.q_odom_imu * Eigen::Quaterniond(imu_from_lidar_rotation_);
      const Eigen::Vector3d p_map_lidar =
          correction.q_odom_imu * imu_from_lidar_translation_ + correction.position;
      snapshot.lidar_pose_in_map.position.x = p_map_lidar.x();
      snapshot.lidar_pose_in_map.position.y = p_map_lidar.y();
      snapshot.lidar_pose_in_map.position.z = p_map_lidar.z();
      snapshot.lidar_pose_in_map.orientation.x = q_map_lidar.x();
      snapshot.lidar_pose_in_map.orientation.y = q_map_lidar.y();
      snapshot.lidar_pose_in_map.orientation.z = q_map_lidar.z();
      snapshot.lidar_pose_in_map.orientation.w = q_map_lidar.w();
      snapshot.cloud = map_message_;
      map_snapshot_publisher_.publish(snapshot);
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  Config config_;
  livox_reflective_marker::ReflectivityFilter reflectivity_filter_;
  MotionCompensator motion_;
  ReflectiveVoxelMap map_;
  OdomBuffer odom_buffer_;
  MapElementFilter map_element_filter_;
  Eigen::Matrix3d imu_from_lidar_rotation_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d imu_from_lidar_translation_ = Eigen::Vector3d::Zero();
  std::vector<PendingScan> pending_scans_;
  std::deque<uint32_t> scan_order_;
  std::deque<PoseCorrection> pending_poses_;
  std::optional<PoseCorrection> last_external_pose_;
  std::optional<MapFrameAnchor> map_anchor_;
  bool map_frame_tf_published_ = false;
  NominalState scratch_start_;
  NominalState scratch_end_;
  ros::Subscriber lidar_subscriber_;
  ros::Subscriber imu_subscriber_;
  ros::Subscriber pose_subscriber_;
	  ros::Publisher map_publisher_;
	  ros::Publisher map_snapshot_publisher_;
	  ros::Publisher deskewed_high_ref_cloud_publisher_;
  ros::Publisher all_high_ref_history_publisher_;
	  ros::Publisher observation_frame_publisher_;
	  ros::Publisher path_since_start_publisher_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_;
  ros::Timer worker_timer_;
  ros::Timer map_publish_timer_;
  sensor_msgs::PointCloud2 map_message_;
  std::vector<DeskewedHighPoint> all_high_ref_history_;
  nav_msgs::Path path_since_start_;
  double latest_imu_time_ = -std::numeric_limits<double>::infinity();
  std::optional<PoseCorrection> latest_map_correction_;
	  bool map_dirty_ = false;
	  uint64_t next_scan_id_ = 0;
	  uint64_t next_observation_scan_id_ = 0;
	  uint64_t committed_scans_ = 0;
  uint64_t initialization_scans_ = 0;
  uint64_t epoch_resets_ = 0;
  uint64_t dropped_scan_overflow_ = 0;
  uint64_t dropped_pose_overflow_ = 0;
  uint64_t dropped_pose_timeout_ = 0;
  uint64_t dropped_pose_mismatch_ = 0;
  uint64_t dropped_stale_pose_ = 0;
  uint64_t dropped_missing_imu_ = 0;
  uint64_t dropped_odom_wait_timeout_ = 0;
  uint64_t dropped_odom_gap_ = 0;
  uint64_t dropped_odom_jump_ = 0;
  uint64_t dropped_time_invalid_ = 0;
  uint64_t dropped_empty_scan_ = 0;
  uint64_t dropped_poseat_invalid_ = 0;
	  uint64_t dropped_point_budget_ = 0;
	  uint64_t dropped_point_missing_state_ = 0;
	  uint64_t dropped_observation_point_budget_ = 0;
	};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "dynamic_reflective_mapping");
  try {
    DynamicReflectiveMappingNode node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("dynamic_reflective_mapping initialization failed: %s", error.what());
    return 1;
  }
  return 0;
}
