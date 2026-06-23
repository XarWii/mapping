#include <Eigen/Dense>

#include <livox_ros_driver2/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

#include <livox_reflective_marker/reflectivity_filter.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
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

struct PendingPoint {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint32_t offset_time = 0;
  uint8_t reflectivity = 0;
};

struct PendingScan {
  bool active = false;
  uint64_t scan_id = 0;
  double lidar_base_time = 0.0;
  double imu_start_time = 0.0;
  double imu_end_time = 0.0;
  uint64_t end_tick = 0;
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
  std::string lidar_time_base = "header_stamp";
  std::string map_frame = "reflective_odom";
  // livox_ros_driver2 forwards its native accelerometer measurement in g,
  // even though sensor_msgs/Imu conventionally uses m/s^2.
  double imu_acceleration_scale = 9.80665;
  int reflectivity_threshold = 160;
  float min_distance_m = 0.1f;
  float max_distance_m = 30.0f;
  double offset_time_scale_sec = 1e-9;
  double lidar_to_imu_time_offset_sec = 0.0;
  double pose_match_epsilon_sec = 0.002;
  double pose_timeout_sec = 0.5;
  double max_external_velocity_mps = 5.0;
  double max_external_angular_velocity_rps = 2.0;
  double external_position_margin_m = 0.5;
  double external_rotation_margin_rad = 0.3;
  int max_pending_scans = 4;
  int max_pending_poses = 8;
  int max_high_points_per_scan = 8192;
  double wheel_tick_sec = 0.05;
  double ttl_sec = 1.0;
  int map_hash_capacity = 65536;
  int map_node_capacity = 32768;
  int wheel_bucket_count = 128;
  double voxel_size_m = 0.05;
  double merge_distance_m = 0.05;
  bool publish_debug_map = true;
  double debug_map_publish_rate_hz = 1.0;
  std::string debug_map_topic = "/reflective/rolling_map";
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
        motion_(config_.motion), map_(MakeMapOptions(config_)) {
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
    worker_timer_ = nh_.createTimer(ros::Duration(1.0 / config_.worker_rate_hz),
                                    &DynamicReflectiveMappingNode::Worker, this);

    ROS_INFO("Dynamic reflective mapper: lidar=%s imu=%s pose=%s map=%s",
             config_.lidar_topic.c_str(), config_.imu_topic.c_str(),
             config_.pose_topic.c_str(), config_.debug_map_topic.c_str());
  }

 private:
  static ReflectiveVoxelMap::Options MakeMapOptions(const Config& config) {
    ReflectiveVoxelMap::Options options;
    options.hash_capacity = config.map_hash_capacity;
    options.node_capacity = config.map_node_capacity;
    options.wheel_bucket_count = config.wheel_bucket_count;
    options.voxel_size = static_cast<float>(config.voxel_size_m);
    options.merge_distance = static_cast<float>(config.merge_distance_m);
    options.ttl_ticks = std::max<uint64_t>(1, static_cast<uint64_t>(
        std::ceil(config.ttl_sec / config.wheel_tick_sec)));
    return options;
  }

  Config LoadConfig() {
    Config config;
    private_nh_.param("lidar_topic", config.lidar_topic, config.lidar_topic);
    private_nh_.param("imu_topic", config.imu_topic, config.imu_topic);
    private_nh_.param("pose_topic", config.pose_topic, config.pose_topic);
    private_nh_.param("pose_input_frame", config.pose_input_frame,
                      config.pose_input_frame);
    private_nh_.param("lidar_time_base", config.lidar_time_base, config.lidar_time_base);
    private_nh_.param("map_frame", config.map_frame, config.map_frame);
    private_nh_.param("imu_acceleration_scale", config.imu_acceleration_scale,
                      config.imu_acceleration_scale);
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
    private_nh_.param("pose_timeout_sec", config.pose_timeout_sec, config.pose_timeout_sec);
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
    private_nh_.param("publish_debug_map", config.publish_debug_map,
                      config.publish_debug_map);
    private_nh_.param("debug_map_publish_rate_hz", config.debug_map_publish_rate_hz,
                      config.debug_map_publish_rate_hz);
    private_nh_.param("debug_map_topic", config.debug_map_topic, config.debug_map_topic);
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
    private_nh_.param("initial_bias_window_sec", config.motion.initial_bias_window_sec,
                      config.motion.initial_bias_window_sec);
    return config;
  }

  void ValidateConfig() {
    if (config_.reflectivity_threshold < 0 || config_.reflectivity_threshold > 255 ||
        config_.max_pending_scans <= 0 || config_.max_pending_poses <= 0 ||
        config_.max_high_points_per_scan <= 0 || config_.wheel_tick_sec <= 0.0 ||
        config_.ttl_sec <= 0.0 || config_.wheel_bucket_count <= 1 ||
        config_.map_hash_capacity <= 1 || config_.map_node_capacity <= 0 ||
        config_.map_node_capacity >= config_.map_hash_capacity ||
        config_.voxel_size_m <= 0.0 || config_.merge_distance_m <= 0.0 ||
        config_.merge_distance_m > std::sqrt(3.0) * config_.voxel_size_m ||
        config_.offset_time_scale_sec <= 0.0 || config_.imu_acceleration_scale <= 0.0 ||
        config_.worker_rate_hz <= 0.0 ||
        config_.motion.max_imu_samples < 2 ||
        config_.imu_from_lidar_translation.size() != 3 ||
        config_.imu_from_lidar_rotation.size() != 9 ||
        (config_.pose_input_frame != "imu" && config_.pose_input_frame != "lidar") ||
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
    scan.imu_start_time = lidar_start + config_.lidar_to_imu_time_offset_sec;
    scan.imu_end_time = lidar_end + config_.lidar_to_imu_time_offset_sec;
    scan.end_tick = ToMapTick(lidar_end);
    scan.points.clear();

    for (const livox_ros_driver2::CustomPoint& point : message->points) {
      if (!reflectivity_filter_.Accepts(point)) continue;
      const Eigen::Vector3f location(point.x, point.y, point.z);
      const float distance = location.norm();
      if (!location.allFinite() || distance < config_.min_distance_m ||
          distance > config_.max_distance_m) {
        continue;
      }
      if (scan.points.size() == scan.points.capacity()) {
        ++dropped_point_budget_;
        continue;
      }
      scan.points.push_back(PendingPoint{point.x, point.y, point.z, point.offset_time,
                                         point.reflectivity});
    }
    scan_order_.push_back(scan_slot);
  }

  void HandleImu(const sensor_msgs::Imu::ConstPtr& message) {
    const double time = message->header.stamp.toSec();
    const Eigen::Vector3d gyro(message->angular_velocity.x, message->angular_velocity.y,
                               message->angular_velocity.z);
    const Eigen::Vector3d accel(message->linear_acceleration.x,
                                message->linear_acceleration.y,
                                message->linear_acceleration.z);
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
    motion_.PushImu(ImuSample{time, gyro, accel * config_.imu_acceleration_scale});
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
    pending_poses_.push_back(correction);
  }

  void Worker(const ros::TimerEvent&) {
    while (!scan_order_.empty()) {
      const uint32_t scan_slot = scan_order_.front();
      PendingScan& scan = pending_scans_[scan_slot];
      if (!scan.active) {
        scan_order_.pop_front();
        continue;
      }
      const std::optional<size_t> pose_index = FindMatchingPose(scan);
      if (!pose_index.has_value()) {
        DiscardStalePoses(scan);
        if (latest_imu_time_ > scan.imu_end_time + config_.pose_timeout_sec) {
          DropOldestScan();
          ++dropped_pose_timeout_;
          continue;
        }
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
      if (!motion_.StateAt(scan.imu_start_time, &scratch_start_) ||
          !motion_.StateAt(scan.imu_end_time, &scratch_end_) ||
          !motion_.StateAt(correction.time, &reference_state)) {
        if (latest_imu_time_ > scan.imu_end_time + config_.pose_timeout_sec) {
          pending_poses_.erase(pending_poses_.begin() + *pose_index);
          DropOldestScan();
          ++dropped_missing_imu_;
          continue;
        }
        break;
      }
      CommitScan(scan, correction, reference_state);
      if (!motion_.ApplyPoseCorrection(correction)) {
        ROS_ERROR_THROTTLE(1.0, "delayed pose replay failed after map insertion");
      }
      last_external_pose_ = correction;
      pending_poses_.erase(pending_poses_.begin() + *pose_index);
      DropOldestScan();
      ++committed_scans_;
      PublishMapIfDue(correction.time);
    }
    ROS_INFO_THROTTLE(2.0,
        "reflective map voxels=%zu committed=%llu pending_scans=%zu pending_poses=%zu drops=%llu",
        map_.occupied_count(), static_cast<unsigned long long>(committed_scans_),
        scan_order_.size(), pending_poses_.size(),
        static_cast<unsigned long long>(dropped_pose_timeout_ + dropped_missing_imu_));
  }

  std::optional<size_t> FindMatchingPose(const PendingScan& scan) const {
    std::optional<size_t> match;
    double closest_to_scan_end = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < pending_poses_.size(); ++i) {
      const double time = pending_poses_[i].time;
      if (time >= scan.imu_start_time - config_.pose_match_epsilon_sec &&
          time <= scan.imu_end_time + config_.pose_match_epsilon_sec) {
        // LIO odometry is normally stamped at scan end.  A Livox frame spans
        // about 100 ms and can contain two valid 10 Hz odometry samples, so
        // picking the first queued sample produces a full-scan pose error.
        const double distance_to_end = std::abs(time - scan.imu_end_time);
        if (distance_to_end < closest_to_scan_end) {
          closest_to_scan_end = distance_to_end;
          match = i;
        }
      }
    }
    return match;
  }

  void DiscardStalePoses(const PendingScan& scan) {
    while (!pending_poses_.empty() &&
           pending_poses_.front().time < scan.imu_start_time - config_.pose_match_epsilon_sec) {
      pending_poses_.pop_front();
      ++dropped_stale_pose_;
    }
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

  void CommitScan(const PendingScan& scan, const PoseCorrection& correction,
                  const NominalState& reference_state) {
    map_.Advance(scan.end_tick);
    const Eigen::Matrix3d predicted_reference_rotation =
        reference_state.q_odom_imu.toRotationMatrix() * imu_from_lidar_rotation_;
    const Eigen::Vector3d predicted_reference_translation =
        reference_state.q_odom_imu * imu_from_lidar_translation_ + reference_state.position;
    const Eigen::Matrix3d external_rotation =
        correction.q_odom_imu.toRotationMatrix() * imu_from_lidar_rotation_;
    const Eigen::Vector3d external_translation =
        correction.q_odom_imu * imu_from_lidar_translation_ + correction.position;

    for (const PendingPoint& point : scan.points) {
      const double point_imu_time = scan.lidar_base_time +
          point.offset_time * config_.offset_time_scale_sec +
          config_.lidar_to_imu_time_offset_sec;
      NominalState point_state;
      if (!motion_.StateAt(point_imu_time, &point_state)) {
        ++dropped_point_missing_state_;
        continue;
      }
      const Eigen::Vector3d point_lidar(point.x, point.y, point.z);
      const Eigen::Vector3d point_world_predicted =
          point_state.q_odom_imu * (imu_from_lidar_rotation_ * point_lidar +
                                    imu_from_lidar_translation_) + point_state.position;
      const Eigen::Vector3d point_lidar_reference =
          predicted_reference_rotation.transpose() *
          (point_world_predicted - predicted_reference_translation);
      const Eigen::Vector3d point_local =
          external_rotation * point_lidar_reference + external_translation;
      if (!IsFinite(point_local)) continue;
      map_.Insert(point_local.cast<float>(), scan.scan_id, scan.end_tick, point.reflectivity);
    }
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

  void ResetEpoch() {
    map_.Clear();
    motion_.Reset();
    latest_imu_time_ = -std::numeric_limits<double>::infinity();
    last_map_publish_time_ = -std::numeric_limits<double>::infinity();
    last_external_pose_.reset();
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
    map_message_.point_step = 16;
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
  }

  void PublishMapIfDue(double stamp) {
    if (!config_.publish_debug_map ||
        stamp - last_map_publish_time_ < 1.0 / config_.debug_map_publish_rate_hz) {
      return;
    }
    last_map_publish_time_ = stamp;
    map_message_.header.stamp.fromSec(stamp);
    map_message_.header.frame_id = config_.map_frame;
    map_message_.width = static_cast<uint32_t>(map_.occupied_count());
    map_message_.row_step = map_message_.width * map_message_.point_step;
    map_message_.data.resize(map_message_.row_step);
    size_t output_index = 0;
    for (const VoxelNode& node : map_.nodes()) {
      if (!node.occupied) continue;
      uint8_t* destination = map_message_.data.data() + output_index * map_message_.point_step;
      const float intensity = static_cast<float>(node.intensity_diag);
      std::memcpy(destination, &node.position.x(), sizeof(float));
      std::memcpy(destination + 4, &node.position.y(), sizeof(float));
      std::memcpy(destination + 8, &node.position.z(), sizeof(float));
      std::memcpy(destination + 12, &intensity, sizeof(float));
      ++output_index;
    }
    map_publisher_.publish(map_message_);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  Config config_;
  livox_reflective_marker::ReflectivityFilter reflectivity_filter_;
  MotionCompensator motion_;
  ReflectiveVoxelMap map_;
  Eigen::Matrix3d imu_from_lidar_rotation_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d imu_from_lidar_translation_ = Eigen::Vector3d::Zero();
  std::vector<PendingScan> pending_scans_;
  std::deque<uint32_t> scan_order_;
  std::deque<PoseCorrection> pending_poses_;
  std::optional<PoseCorrection> last_external_pose_;
  NominalState scratch_start_;
  NominalState scratch_end_;
  ros::Subscriber lidar_subscriber_;
  ros::Subscriber imu_subscriber_;
  ros::Subscriber pose_subscriber_;
  ros::Publisher map_publisher_;
  ros::Timer worker_timer_;
  sensor_msgs::PointCloud2 map_message_;
  double latest_imu_time_ = -std::numeric_limits<double>::infinity();
  double last_map_publish_time_ = -std::numeric_limits<double>::infinity();
  uint64_t next_scan_id_ = 0;
  uint64_t committed_scans_ = 0;
  uint64_t initialization_scans_ = 0;
  uint64_t epoch_resets_ = 0;
  uint64_t dropped_scan_overflow_ = 0;
  uint64_t dropped_pose_overflow_ = 0;
  uint64_t dropped_pose_timeout_ = 0;
  uint64_t dropped_stale_pose_ = 0;
  uint64_t dropped_missing_imu_ = 0;
  uint64_t dropped_point_budget_ = 0;
  uint64_t dropped_point_missing_state_ = 0;
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
