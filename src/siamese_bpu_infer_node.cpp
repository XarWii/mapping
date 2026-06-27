/// @file siamese_bpu_infer_node.cpp
/// @brief Pure BPU scorer — extracts candidates, runs inference, publishes BpuScores.
///
/// Does NOT: track, confirm, handoff, audit, validate, or talk to EKF.
/// All arbitration is done by target_manager_node; final target commands and
/// EKF status are consumed here only to color the visualization cloud.

#include "siamese_bpu_infer.h"

#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <livox_ros_driver2/CustomMsg.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <livox_reflective_marker/BpuScores.h>
#include <livox_reflective_marker/BpuScoreEntry.h>
#include <livox_reflective_marker/ClusterCloud.h>
#include <livox_reflective_marker/RecognitionCommand.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Point types
// ---------------------------------------------------------------------------

struct PointXYZI {
  float x = 0.0f, y = 0.0f, z = 0.0f;
  uint8_t reflectivity = 0;
};

struct Cluster {
  std::vector<PointXYZI> points;
  std::vector<size_t> indices;  // indices into the source point cloud
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  float score = 0.0f;
  size_t size() const { return points.size(); }
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
  std::string input_topic = "/livox/lidar";
  std::string input_msg_type = "custom";
  std::string bpu_model_path;

  uint8_t reflectivity_threshold = 250;
  float min_distance_m = 0.1f, max_distance_m = 30.0f;

  float cluster_tolerance_m = 0.25f;
  size_t min_cluster_points = 3, max_cluster_points = 2000;
  size_t min_inference_cluster_points = 0;
  size_t inference_fps_target_points = 0;

  int max_accumulation_frames = 20;
  double publish_interval_sec = 0.1;
  bool publish_markers = true;
  std::string frame_id = "livox_frame";

  float roi_radius_m = 0.45f;

  bool enable_size_filter = true;
  bool enable_plane_filter = true;
  float size_filter_min_long_axis_m = 0.12f;
  float size_filter_max_long_axis_m = 0.35f;
  float size_filter_min_short_axis_m = 0.06f;
  float size_filter_max_short_axis_m = 0.30f;
  float plane_filter_max_thickness_m = 0.08f;

  bool debug_dump_candidates = false;
  std::string debug_dump_dir = "/tmp/siamese_bpu_candidate_dump";
  int debug_dump_top_k = 1;
  double debug_dump_interval_sec = 1.0;

  float decision_score_threshold = -0.004310191f;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool IsFinite(float v) { return std::isfinite(v); }

PointXYZI ToPoint(const livox_ros_driver2::CustomPoint& raw) {
  PointXYZI p;
  p.x = raw.x; p.y = raw.y; p.z = raw.z;
  p.reflectivity = raw.reflectivity;
  return p;
}

const sensor_msgs::PointField* FindField(const sensor_msgs::PointCloud2& cloud,
                                         const std::string& name) {
  for (const auto& field : cloud.fields) {
    if (field.name == name) return &field;
  }
  return nullptr;
}

float ReadFloat32(const uint8_t* ptr) {
  float value = 0.0f;
  std::memcpy(&value, ptr, sizeof(float));
  return value;
}

float ReadFieldAsFloat(const uint8_t* ptr, uint8_t datatype) {
  switch (datatype) {
    case sensor_msgs::PointField::INT8:
      return static_cast<float>(*reinterpret_cast<const int8_t*>(ptr));
    case sensor_msgs::PointField::UINT8:
      return static_cast<float>(*reinterpret_cast<const uint8_t*>(ptr));
    case sensor_msgs::PointField::INT16: {
      int16_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<float>(v);
    }
    case sensor_msgs::PointField::UINT16: {
      uint16_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<float>(v);
    }
    case sensor_msgs::PointField::INT32: {
      int32_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<float>(v);
    }
    case sensor_msgs::PointField::UINT32: {
      uint32_t v = 0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<float>(v);
    }
    case sensor_msgs::PointField::FLOAT32:
      return ReadFloat32(ptr);
    case sensor_msgs::PointField::FLOAT64: {
      double v = 0.0;
      std::memcpy(&v, ptr, sizeof(v));
      return static_cast<float>(v);
    }
    default:
      return std::numeric_limits<float>::quiet_NaN();
  }
}

uint8_t ReflectivityFromFloat(float value) {
  if (!std::isfinite(value)) return 0;
  const int rounded = static_cast<int>(std::lround(value));
  return static_cast<uint8_t>(std::max(0, std::min(255, rounded)));
}

float SquaredDistance(const PointXYZI& a, const PointXYZI& b) {
  float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

bool IsHighReflective(const PointXYZI& p, const Config& cfg) {
  float d2 = p.x * p.x + p.y * p.y + p.z * p.z;
  float d = std::sqrt(d2);
  return p.reflectivity >= cfg.reflectivity_threshold &&
         d >= cfg.min_distance_m && d <= cfg.max_distance_m &&
         IsFinite(p.x) && IsFinite(p.y) && IsFinite(p.z);
}

std::array<float, 3> SortedAxisAlignedExtents(
    const std::vector<Eigen::Vector3f>& points) {
  std::array<float, 3> extents{0.0f, 0.0f, 0.0f};
  if (points.empty()) return extents;
  Eigen::Vector3f min_p = points.front();
  Eigen::Vector3f max_p = points.front();
  for (const auto& p : points) {
    min_p = min_p.cwiseMin(p);
    max_p = max_p.cwiseMax(p);
  }
  const Eigen::Vector3f diff = max_p - min_p;
  extents = {diff.x(), diff.y(), diff.z()};
  std::sort(extents.begin(), extents.end(), std::greater<float>());
  return extents;
}

std::array<float, 3> EstimatePcaExtents(
    const std::vector<Eigen::Vector3f>& points) {
  if (points.size() < 3) return SortedAxisAlignedExtents(points);

  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  for (const auto& p : points) mean += p;
  mean /= static_cast<float>(points.size());

  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  for (const auto& p : points) {
    const Eigen::Vector3f d = p - mean;
    cov += d * d.transpose();
  }
  cov /= static_cast<float>(points.size() - 1);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
  if (es.info() != Eigen::Success) return SortedAxisAlignedExtents(points);

  Eigen::Matrix3f axes;
  axes.col(0) = es.eigenvectors().col(2);
  axes.col(1) = es.eigenvectors().col(1);
  axes.col(2) = es.eigenvectors().col(0);

  Eigen::Vector3f min_q(std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max(),
                        std::numeric_limits<float>::max());
  Eigen::Vector3f max_q(-std::numeric_limits<float>::max(),
                        -std::numeric_limits<float>::max(),
                        -std::numeric_limits<float>::max());
  for (const auto& p : points) {
    const Eigen::Vector3f q = axes.transpose() * (p - mean);
    min_q = min_q.cwiseMin(q);
    max_q = max_q.cwiseMax(q);
  }

  const Eigen::Vector3f diff = max_q - min_q;
  std::array<float, 3> extents{diff.x(), diff.y(), diff.z()};
  std::sort(extents.begin(), extents.end(), std::greater<float>());
  return extents;
}

bool PassSizeFilter(const Cluster& cluster, const Config& cfg) {
  if (!cfg.enable_size_filter && !cfg.enable_plane_filter) return true;
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(cluster.points.size());
  for (const auto& p : cluster.points) pts.emplace_back(p.x, p.y, p.z);
  const auto ext = EstimatePcaExtents(pts);
  const float long_axis = ext[0];
  const float short_axis = ext[1];
  const float thickness = ext[2];

  if (cfg.enable_plane_filter &&
      thickness > cfg.plane_filter_max_thickness_m) {
    return false;
  }
  if (cfg.enable_size_filter &&
      (long_axis < cfg.size_filter_min_long_axis_m ||
       long_axis > cfg.size_filter_max_long_axis_m ||
       short_axis < cfg.size_filter_min_short_axis_m ||
       short_axis > cfg.size_filter_max_short_axis_m)) {
    return false;
  }
  return true;
}

std::vector<Cluster> ApplyCandidateFilter(std::vector<Cluster>&& clusters,
                                          const Config& cfg,
                                          size_t* rejected) {
  if (rejected != nullptr) *rejected = 0;
  if ((!cfg.enable_size_filter && !cfg.enable_plane_filter) ||
      clusters.empty()) {
    return std::move(clusters);
  }

  std::vector<Cluster> kept;
  kept.reserve(clusters.size());
  for (auto& cluster : clusters) {
    if (PassSizeFilter(cluster, cfg)) {
      kept.push_back(std::move(cluster));
    } else if (rejected != nullptr) {
      ++(*rejected);
    }
  }
  return kept;
}

// ---------------------------------------------------------------------------
// Euclidean clustering
// ---------------------------------------------------------------------------

std::vector<Cluster> EuclideanCluster(const std::vector<PointXYZI>& points,
                                      const Config& cfg) {
  std::vector<Cluster> clusters;
  if (points.empty()) return clusters;

  const float tol_sq = cfg.cluster_tolerance_m * cfg.cluster_tolerance_m;
  std::vector<bool> visited(points.size(), false);
  std::vector<size_t> queue;
  queue.reserve(points.size());

  for (size_t i = 0; i < points.size(); ++i) {
    if (visited[i]) continue;
    visited[i] = true;
    queue.clear();
    queue.push_back(i);

    std::vector<size_t> indices;
    for (size_t head = 0; head < queue.size(); ++head) {
      size_t cur = queue[head];
      indices.push_back(cur);
      for (size_t j = 0; j < points.size(); ++j) {
        if (!visited[j] && SquaredDistance(points[cur], points[j]) <= tol_sq) {
          visited[j] = true;
          queue.push_back(j);
        }
      }
    }

    if (indices.size() >= cfg.min_cluster_points &&
        (cfg.max_cluster_points == 0 ||
         indices.size() <= cfg.max_cluster_points)) {
      Cluster c;
      c.indices = std::move(indices);
      c.center = Eigen::Vector3f::Zero();
      for (auto idx : c.indices) {
        c.points.push_back(points[idx]);
        c.center += Eigen::Vector3f(points[idx].x, points[idx].y, points[idx].z);
      }
      c.center /= static_cast<float>(c.indices.size());
      clusters.push_back(std::move(c));
    }
  }

  std::sort(clusters.begin(), clusters.end(),
            [](const Cluster& a, const Cluster& b) { return a.size() > b.size(); });
  return clusters;
}

std::vector<Eigen::Vector3f> ClusterToEigen(const Cluster& cluster) {
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(cluster.size());
  for (const auto& p : cluster.points)
    pts.emplace_back(p.x, p.y, p.z);
  return pts;
}

std::vector<siamese_bpu::BpuPoint> PointsToBpu(
    const std::vector<PointXYZI>& points) {
  std::vector<siamese_bpu::BpuPoint> pts;
  pts.reserve(points.size());
  for (const auto& p : points) {
    siamese_bpu::BpuPoint out;
    out.xyz = Eigen::Vector3f(p.x, p.y, p.z);
    out.reflectivity = static_cast<float>(p.reflectivity);
    pts.push_back(out);
  }
  return pts;
}

float PointDistanceSquared(const PointXYZI& a, const PointXYZI& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

std::vector<PointXYZI> FarthestPointSample(
    const std::vector<PointXYZI>& points, size_t target_count) {
  if (target_count == 0 || points.size() <= target_count) return points;

  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  for (const auto& p : points) center += Eigen::Vector3f(p.x, p.y, p.z);
  center /= static_cast<float>(points.size());

  size_t first = 0;
  float best_center_d2 = std::numeric_limits<float>::max();
  for (size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector3f d(points[i].x - center.x(),
                            points[i].y - center.y(),
                            points[i].z - center.z());
    const float d2 = d.squaredNorm();
    if (d2 < best_center_d2) {
      best_center_d2 = d2;
      first = i;
    }
  }

  std::vector<PointXYZI> sampled;
  sampled.reserve(target_count);
  std::vector<float> nearest_d2(points.size(),
                                std::numeric_limits<float>::max());
  std::vector<uint8_t> selected(points.size(), 0);

  size_t current = first;
  for (size_t out = 0; out < target_count; ++out) {
    sampled.push_back(points[current]);
    selected[current] = 1;

    size_t next = current;
    float farthest_d2 = -1.0f;
    for (size_t i = 0; i < points.size(); ++i) {
      const float d2 = PointDistanceSquared(points[i], points[current]);
      nearest_d2[i] = std::min(nearest_d2[i], d2);
      if (!selected[i] && nearest_d2[i] > farthest_d2) {
        farthest_d2 = nearest_d2[i];
        next = i;
      }
    }
    current = next;
  }
  return sampled;
}

std::vector<PointXYZI> InferencePoints(const Cluster& cluster,
                                       const Config& cfg) {
  (void)cfg;
  return cluster.points;
}

// ---------------------------------------------------------------------------
// ROS Node — pure BPU scorer
// ---------------------------------------------------------------------------

class SiameseBpuInferNode {
 public:
  SiameseBpuInferNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : nh_(nh), pnh_(pnh) {
    loadConfig();
    initBpu();
    initPublishers();
    if (cfg_.input_msg_type == "pointcloud2" ||
        cfg_.input_msg_type == "sensor_msgs/PointCloud2") {
      pointcloud2_sub_ = nh_.subscribe(
          cfg_.input_topic, 4,
          &SiameseBpuInferNode::pointCloud2Callback, this);
    } else {
      sub_ = nh_.subscribe(cfg_.input_topic, 4,
                           &SiameseBpuInferNode::pointCloudCallback, this);
    }
    recfg_sub_ = nh_.subscribe(
        "/target_manager/recognition_command", 4,
        &SiameseBpuInferNode::recognitionCommandCallback, this);
    ROS_INFO("[siamese_bpu] pure scorer ready | input_type=%s model: %s",
             cfg_.input_msg_type.c_str(), cfg_.bpu_model_path.c_str());
  }

 private:
  void loadConfig() {
#define LOAD(ns, name, var) ns.param(name, var, var)
    pnh_.param<std::string>("input_topic", cfg_.input_topic, cfg_.input_topic);
    pnh_.param<std::string>("input_msg_type", cfg_.input_msg_type,
                            cfg_.input_msg_type);
    pnh_.param<std::string>("bpu_model_path", cfg_.bpu_model_path, cfg_.bpu_model_path);

    int rf = static_cast<int>(cfg_.reflectivity_threshold);
    pnh_.param("reflectivity_threshold", rf, rf);
    cfg_.reflectivity_threshold = static_cast<uint8_t>(rf);

    LOAD(pnh_, "min_distance_m", cfg_.min_distance_m);
    LOAD(pnh_, "max_distance_m", cfg_.max_distance_m);
    LOAD(pnh_, "cluster_tolerance_m", cfg_.cluster_tolerance_m);

    int mn = static_cast<int>(cfg_.min_cluster_points);
    int mx = static_cast<int>(cfg_.max_cluster_points);
    int min_infer = static_cast<int>(cfg_.min_inference_cluster_points);
    int fps_target = static_cast<int>(cfg_.inference_fps_target_points);
    pnh_.param("min_cluster_points", mn, mn);
    pnh_.param("max_cluster_points", mx, mx);
    pnh_.param("min_inference_cluster_points", min_infer, min_infer);
    pnh_.param("inference_fps_target_points", fps_target, fps_target);
    cfg_.min_cluster_points = static_cast<size_t>(mn);
    cfg_.max_cluster_points = static_cast<size_t>(mx);
    cfg_.min_inference_cluster_points =
        static_cast<size_t>(std::max(0, min_infer));
    cfg_.inference_fps_target_points =
        static_cast<size_t>(std::max(0, fps_target));

    LOAD(pnh_, "max_accumulation_frames", cfg_.max_accumulation_frames);
    LOAD(pnh_, "publish_interval_sec", cfg_.publish_interval_sec);
    LOAD(pnh_, "publish_markers", cfg_.publish_markers);
    LOAD(pnh_, "roi_radius_m", cfg_.roi_radius_m);
    LOAD(pnh_, "enable_size_filter", cfg_.enable_size_filter);
    LOAD(pnh_, "enable_plane_filter", cfg_.enable_plane_filter);
    LOAD(pnh_, "size_filter_min_long_axis_m",
         cfg_.size_filter_min_long_axis_m);
    LOAD(pnh_, "size_filter_max_long_axis_m",
         cfg_.size_filter_max_long_axis_m);
    LOAD(pnh_, "size_filter_min_short_axis_m",
         cfg_.size_filter_min_short_axis_m);
    LOAD(pnh_, "size_filter_max_short_axis_m",
         cfg_.size_filter_max_short_axis_m);
    LOAD(pnh_, "plane_filter_max_thickness_m",
         cfg_.plane_filter_max_thickness_m);
    LOAD(pnh_, "debug_dump_candidates", cfg_.debug_dump_candidates);
    LOAD(pnh_, "decision_score_threshold", cfg_.decision_score_threshold);
    pnh_.param<std::string>("debug_dump_dir", cfg_.debug_dump_dir,
                            cfg_.debug_dump_dir);
    LOAD(pnh_, "debug_dump_top_k", cfg_.debug_dump_top_k);
    LOAD(pnh_, "debug_dump_interval_sec", cfg_.debug_dump_interval_sec);
    pnh_.param<std::string>("frame_id", cfg_.frame_id, cfg_.frame_id);

    cfg_.debug_dump_top_k = std::max(1, cfg_.debug_dump_top_k);
    cfg_.debug_dump_interval_sec = std::max(0.0, cfg_.debug_dump_interval_sec);

    if (cfg_.bpu_model_path.empty()) {
      ROS_ERROR("[siamese_bpu] bpu_model_path is required!");
      ros::shutdown();
    }
#undef LOAD
  }

  void initBpu() {
    ROS_INFO("[siamese_bpu] loading BPU model: %s", cfg_.bpu_model_path.c_str());
    model_.init(cfg_.bpu_model_path);
    ROS_INFO("[siamese_bpu] BPU model loaded");
  }

  void initPublishers() {
    bpu_scores_pub_ = pnh_.advertise<livox_reflective_marker::BpuScores>(
        "bpu_scores", 4);
    if (cfg_.publish_markers) {
      marker_pub_ = pnh_.advertise<visualization_msgs::MarkerArray>(
          "candidate_markers", 4, true);
    }
    cluster_cloud_pub_ = pnh_.advertise<livox_reflective_marker::ClusterCloud>(
        "latest_cluster_cloud", 4);
  }

  void recognitionCommandCallback(
      const livox_reflective_marker::RecognitionCommand::ConstPtr& cmd) {
    if (cmd->max_accumulation_frames > 0) {
      cfg_.max_accumulation_frames = cmd->max_accumulation_frames;
      frame_buffer_.clear();
    }
    if (cmd->publish_interval_sec > 0.0) {
      cfg_.publish_interval_sec = cmd->publish_interval_sec;
    }
    ROS_INFO("[siamese_bpu] recfg: accum=%d frames  interval=%.2fs",
             cfg_.max_accumulation_frames, cfg_.publish_interval_sec);
  }

  // ---- Main callback ---------------------------------------------------

  void pointCloudCallback(const livox_ros_driver2::CustomMsg::ConstPtr& msg) {
    std::vector<PointXYZI> points;
    points.reserve(msg->points.size());
    for (const auto& raw : msg->points) {
      points.push_back(ToPoint(raw));
    }
    processFrame(msg->header.stamp, points);
  }

  void pointCloud2Callback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    if (msg->is_bigendian) {
      ROS_WARN_THROTTLE(2.0, "[siamese_bpu] big-endian PointCloud2 is not supported");
      return;
    }

    const sensor_msgs::PointField* x_field = FindField(*msg, "x");
    const sensor_msgs::PointField* y_field = FindField(*msg, "y");
    const sensor_msgs::PointField* z_field = FindField(*msg, "z");
    const sensor_msgs::PointField* i_field = FindField(*msg, "reflectivity");
    if (!i_field) i_field = FindField(*msg, "intensity");
    if (!x_field || !y_field || !z_field || !i_field) {
      ROS_WARN_THROTTLE(2.0,
                        "[siamese_bpu] PointCloud2 missing x/y/z/intensity fields");
      return;
    }

    std::vector<PointXYZI> points;
    points.reserve(static_cast<size_t>(msg->width) * msg->height);
    for (uint32_t row = 0; row < msg->height; ++row) {
      const size_t row_base = static_cast<size_t>(row) * msg->row_step;
      for (uint32_t col = 0; col < msg->width; ++col) {
        const uint8_t* ptr =
            msg->data.data() + row_base + static_cast<size_t>(col) * msg->point_step;
        PointXYZI p;
        p.x = ReadFieldAsFloat(ptr + x_field->offset, x_field->datatype);
        p.y = ReadFieldAsFloat(ptr + y_field->offset, y_field->datatype);
        p.z = ReadFieldAsFloat(ptr + z_field->offset, z_field->datatype);
        p.reflectivity = ReflectivityFromFloat(
            ReadFieldAsFloat(ptr + i_field->offset, i_field->datatype));
        points.push_back(p);
      }
    }
    processFrame(msg->header.stamp, points);
  }

  void processFrame(const ros::Time& now,
                    const std::vector<PointXYZI>& raw_points) {
    // 1. Extract high-reflectivity points
    std::vector<PointXYZI> high_pts;
    high_pts.reserve(raw_points.size());
    for (const auto& p : raw_points) {
      if (IsHighReflective(p, cfg_)) high_pts.push_back(p);
    }

    // 2. Accumulate
    frame_buffer_.push_back(std::move(high_pts));
    if (static_cast<int>(frame_buffer_.size()) > cfg_.max_accumulation_frames)
      frame_buffer_.pop_front();

    // 3. Throttle
    if (!last_publish_time_.isZero() &&
        (now - last_publish_time_).toSec() < cfg_.publish_interval_sec)
      return;
    last_publish_time_ = now;

    // Bump window id only when we actually publish.
    window_id_++;

    // 4. Flatten
    std::vector<PointXYZI> accumulated;
    size_t total = 0;
    for (const auto& fp : frame_buffer_) total += fp.size();
    accumulated.reserve(total);
    for (const auto& fp : frame_buffer_)
      accumulated.insert(accumulated.end(), fp.begin(), fp.end());

    // 5. Cluster accumulated points & BPU score
    std::vector<Cluster> clusters;
    size_t raw_cluster_count = 0;
    size_t rejected_by_filter = 0;
    if (!accumulated.empty()) {
      clusters = EuclideanCluster(accumulated, cfg_);
      raw_cluster_count = clusters.size();
      clusters = ApplyCandidateFilter(std::move(clusters), cfg_,
                                      &rejected_by_filter);
      if (cfg_.min_inference_cluster_points > 0) {
        const size_t before = clusters.size();
        clusters.erase(
            std::remove_if(
                clusters.begin(), clusters.end(),
                [&](const Cluster& c) {
                  return c.size() < cfg_.min_inference_cluster_points;
                }),
            clusters.end());
        rejected_by_filter += before - clusters.size();
      }
      if ((cfg_.enable_size_filter || cfg_.enable_plane_filter) &&
          rejected_by_filter > 0) {
        ROS_INFO_THROTTLE(
            1.0,
            "[siamese_bpu] candidate_filter accumulated raw=%zu kept=%zu rejected=%zu",
            raw_cluster_count, clusters.size(), rejected_by_filter);
      }
      for (auto& c : clusters) {
        const std::vector<PointXYZI> inference_points =
            InferencePoints(c, cfg_);
        auto bpu_pts = PointsToBpu(inference_points);
        preproc_.prepare(bpu_pts,
                         static_cast<float>(cfg_.reflectivity_threshold));
        c.score = model_.infer(preproc_.candidatePoint(),
                               preproc_.candidateMask(),
                               preproc_.candidateMeta(),
                               preproc_.candidateCountScale());
      }
    }
    dumpDebugCandidates(clusters, now);

    // 6. Cluster the latest frame once; reuse for centre refresh AND cluster-level coloring.
    std::vector<Cluster> latest_clusters;
    size_t raw_latest_count = 0;
    size_t latest_rejected_by_filter = 0;
    if (!frame_buffer_.empty() && !frame_buffer_.back().empty()) {
      latest_clusters = EuclideanCluster(frame_buffer_.back(), cfg_);
      raw_latest_count = latest_clusters.size();
      latest_clusters = ApplyCandidateFilter(
          std::move(latest_clusters), cfg_, &latest_rejected_by_filter);
      if ((cfg_.enable_size_filter || cfg_.enable_plane_filter) &&
          latest_rejected_by_filter > 0) {
        ROS_INFO_THROTTLE(
            1.0,
            "[siamese_bpu] candidate_filter latest raw=%zu kept=%zu rejected=%zu",
            raw_latest_count, latest_clusters.size(), latest_rejected_by_filter);
      }
    }
    if (!latest_clusters.empty()) {
      const float match_sq = cfg_.cluster_tolerance_m * cfg_.cluster_tolerance_m;
      for (auto& c : clusters) {
        int best = -1;
        float best_d2 = match_sq;
        for (size_t j = 0; j < latest_clusters.size(); ++j) {
          float d2 = (c.center - latest_clusters[j].center).squaredNorm();
          if (d2 < best_d2) { best_d2 = d2; best = static_cast<int>(j); }
        }
        if (best >= 0)
          c.center = latest_clusters[best].center;
      }
    }

    // 7. Publish BpuScores
    livox_reflective_marker::BpuScores scores_msg;
    scores_msg.header.stamp = now;
    scores_msg.header.frame_id = cfg_.frame_id;
    scores_msg.window_id = window_id_;

    for (size_t i = 0; i < clusters.size(); ++i) {
      livox_reflective_marker::BpuScoreEntry entry;
      entry.candidate_id = static_cast<uint32_t>(i);
      entry.score = clusters[i].score;
      entry.center.x = clusters[i].center.x();
      entry.center.y = clusters[i].center.y();
      entry.center.z = clusters[i].center.z();
      entry.roi_radius = cfg_.roi_radius_m;
      scores_msg.entries.push_back(entry);
    }
    bpu_scores_pub_.publish(scores_msg);
    logWindowSummary(now, raw_points.size(), frame_buffer_.empty()
                                      ? 0
                                      : frame_buffer_.back().size(),
                     accumulated.size(), raw_cluster_count, clusters.size(),
                     rejected_by_filter, raw_latest_count, latest_clusters.size(),
                     latest_rejected_by_filter, clusters);

    // 8. Markers
    if (cfg_.publish_markers)
      publishMarkers(clusters, now);

    // 9. Publish filtered latest-frame clusters for target_manager's visualisation.
    publishClusterCloud(latest_clusters, now);
  }

  void logWindowSummary(ros::Time stamp, size_t raw_points, size_t latest_high,
                        size_t accumulated_high, size_t raw_clusters,
                        size_t kept_clusters, size_t rejected_clusters,
                        size_t raw_latest_clusters, size_t kept_latest_clusters,
                        size_t rejected_latest_clusters,
                        const std::vector<Cluster>& clusters) const {
    int top1 = -1, top2 = -1, top3 = -1;
    for (size_t i = 0; i < clusters.size(); ++i) {
      if (top1 < 0 || clusters[i].score > clusters[top1].score) {
        top3 = top2;
        top2 = top1;
        top1 = static_cast<int>(i);
      } else if (top2 < 0 || clusters[i].score > clusters[top2].score) {
        top3 = top2;
        top2 = static_cast<int>(i);
      } else if (top3 < 0 || clusters[i].score > clusters[top3].score) {
        top3 = static_cast<int>(i);
      }
    }

    const float top1_score = top1 >= 0 ? clusters[top1].score : -1.0f;
    const float top2_score = top2 >= 0 ? clusters[top2].score : -1.0f;
    const float top3_score = top3 >= 0 ? clusters[top3].score : -1.0f;
    const float margin =
        (top1 >= 0 && top2 >= 0) ? (top1_score - top2_score) : 0.0f;
    const Eigen::Vector3f top1_center =
        top1 >= 0 ? clusters[top1].center : Eigen::Vector3f::Zero();
    const size_t top1_points =
        top1 >= 0 ? clusters[top1].points.size() : static_cast<size_t>(0);

    ROS_INFO_THROTTLE(
        1.0,
        "[siamese_bpu] window=%u raw_pts=%zu latest_high=%zu accumulated_high=%zu "
        "clusters=%zu kept=%zu rejected=%zu latest_clusters=%zu kept=%zu rejected=%zu "
        "top1_id=%d top1_score=%.4f top1_margin=%.4f top1_pts=%zu "
        "top1_center=[%.2f,%.2f,%.2f] top2_id=%d top2_score=%.4f top3_id=%d top3_score=%.4f",
        window_id_, raw_points, latest_high, accumulated_high, raw_clusters,
        kept_clusters, rejected_clusters, raw_latest_clusters,
        kept_latest_clusters, rejected_latest_clusters, top1, top1_score,
        margin, top1_points, static_cast<double>(top1_center.x()),
        static_cast<double>(top1_center.y()),
        static_cast<double>(top1_center.z()), top2, top2_score, top3,
        top3_score);
  }

  void publishClusterCloud(const std::vector<Cluster>& latest_clusters,
                           ros::Time stamp) {
    if (cluster_cloud_pub_.getNumSubscribers() == 0) return;

    livox_reflective_marker::ClusterCloud msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = cfg_.frame_id;
    msg.window_id = window_id_;

    size_t filtered_point_count = 0;
    for (const auto& cluster : latest_clusters) {
      filtered_point_count += cluster.points.size();
    }

    // Build a raw xyz PointCloud2 from filtered latest clusters only.
    sensor_msgs::PointCloud2 cloud;
    cloud.header = msg.header;
    cloud.height = 1;
    cloud.is_bigendian = false;
    cloud.is_dense = true;

    sensor_msgs::PointField f;
    f.datatype = sensor_msgs::PointField::FLOAT32;
    f.count = 1;
    f.name = "x";  f.offset = 0;  cloud.fields.push_back(f);
    f.name = "y";  f.offset = 4;  cloud.fields.push_back(f);
    f.name = "z";  f.offset = 8;  cloud.fields.push_back(f);

    cloud.point_step = 12;
    cloud.width = static_cast<uint32_t>(filtered_point_count);
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step);

    float* buf = reinterpret_cast<float*>(cloud.data.data());
    msg.per_point_cluster_id.reserve(filtered_point_count);
    size_t out_i = 0;
    for (size_t ci = 0; ci < latest_clusters.size(); ++ci) {
      for (const auto& p : latest_clusters[ci].points) {
        buf[out_i * 3 + 0] = p.x;
        buf[out_i * 3 + 1] = p.y;
        buf[out_i * 3 + 2] = p.z;
        msg.per_point_cluster_id.push_back(static_cast<int32_t>(ci));
        ++out_i;
      }
    }
    msg.cloud = cloud;

    // cluster_centers
    msg.cluster_centers.resize(latest_clusters.size());
    for (size_t ci = 0; ci < latest_clusters.size(); ++ci) {
      msg.cluster_centers[ci].x = latest_clusters[ci].center.x();
      msg.cluster_centers[ci].y = latest_clusters[ci].center.y();
      msg.cluster_centers[ci].z = latest_clusters[ci].center.z();
    }

    cluster_cloud_pub_.publish(msg);
  }

  void publishMarkers(const std::vector<Cluster>& clusters, ros::Time stamp) {
    visualization_msgs::MarkerArray arr;

    visualization_msgs::Marker del;
    del.action = visualization_msgs::Marker::DELETEALL;
    arr.markers.push_back(del);

    for (size_t i = 0; i < clusters.size(); ++i) {
      visualization_msgs::Marker m;
      m.header.frame_id = cfg_.frame_id;
      m.header.stamp = stamp;
      m.ns = "siamese_bpu";
      m.id = static_cast<int>(i);
      m.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      m.action = visualization_msgs::Marker::ADD;
      m.pose.position.x = clusters[i].center.x();
      m.pose.position.y = clusters[i].center.y();
      m.pose.position.z = clusters[i].center.z() + 0.25f;
      m.scale.z = 0.18f;

      float s = clusters[i].score;
      if (s > cfg_.decision_score_threshold) {
        m.color.r = 0.2f; m.color.g = 1.0f; m.color.b = 0.3f;
      } else if (s > cfg_.decision_score_threshold - 0.25f) {
        m.color.r = 1.0f; m.color.g = 1.0f; m.color.b = 0.2f;
      } else {
        m.color.r = 1.0f; m.color.g = 0.55f; m.color.b = 0.0f;
      }
      m.color.a = 1.0f;

      char buf[64];
      std::snprintf(buf, sizeof(buf), "id=%zu:%.3f", i, static_cast<double>(s));
      m.text = buf;
      arr.markers.push_back(m);
    }
    marker_pub_.publish(arr);
  }

  void dumpDebugCandidates(const std::vector<Cluster>& clusters,
                           const ros::Time& stamp) {
    if (!cfg_.debug_dump_candidates || clusters.empty()) return;
    if (!last_debug_dump_time_.isZero() &&
        (stamp - last_debug_dump_time_).toSec() < cfg_.debug_dump_interval_sec) {
      return;
    }
    last_debug_dump_time_ = stamp;

    std::vector<size_t> order(clusters.size());
    for (size_t i = 0; i < clusters.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return clusters[a].score > clusters[b].score;
    });

    std::error_code ec;
    std::filesystem::create_directories(cfg_.debug_dump_dir, ec);
    if (ec) {
      ROS_WARN_THROTTLE(2.0, "[siamese_bpu] debug dump mkdir failed: %s",
                        ec.message().c_str());
      return;
    }

    const std::filesystem::path root(cfg_.debug_dump_dir);
    const std::filesystem::path summary_path = root / "summary.csv";
    const bool write_header = !std::filesystem::exists(summary_path);
    std::ofstream summary(summary_path, std::ios::app);
    if (!summary) {
      ROS_WARN_THROTTLE(2.0, "[siamese_bpu] debug dump summary open failed");
      return;
    }
    if (write_header) {
      summary << "seq,stamp,window,candidate,rank,score,raw_points,inference_points,center_x,center_y,center_z,"
                 "valid_count,valid_ratio,count_scale,dump_dir\n";
    }

    const int n_dump = std::min<int>(
        cfg_.debug_dump_top_k, static_cast<int>(order.size()));
    for (int rank = 0; rank < n_dump; ++rank) {
      const size_t ci = order[rank];
      const Cluster& cluster = clusters[ci];

      const std::vector<PointXYZI> inference_points =
          InferencePoints(cluster, cfg_);
      std::vector<siamese_bpu::BpuPoint> bpu_pts = PointsToBpu(inference_points);
      siamese_bpu::BpuPreprocessor local_preproc;
      local_preproc.prepare(bpu_pts,
                            static_cast<float>(cfg_.reflectivity_threshold));

      std::ostringstream name;
      name << "seq_" << std::setw(6) << std::setfill('0') << debug_dump_seq_
           << "_w" << window_id_ << "_r" << rank << "_c" << ci;
      const std::filesystem::path dir = root / name.str();
      std::filesystem::create_directories(dir, ec);
      if (ec) continue;

      writeRawPointsCsv(dir / "raw_points.csv", cluster);
      writePointsCsv(dir / "inference_points.csv", inference_points);
      writeCandidatePointCsv(dir / "candidate_point.csv",
                             local_preproc.candidatePoint());
      writeMaskCsv(dir / "candidate_mask.csv",
                   local_preproc.candidateMask());
      writeMetaCsv(dir / "candidate_meta.csv",
                   local_preproc.candidateMeta());
      writeMetaYaml(dir / "meta.yaml", stamp, ci, rank, cluster,
                    inference_points.size(),
                    local_preproc.validCount(),
                    local_preproc.validRatio(),
                    local_preproc.candidateCountScale());

      summary << debug_dump_seq_ << "," << std::fixed << std::setprecision(6)
              << stamp.toSec() << "," << window_id_ << "," << ci << ","
              << rank << "," << std::setprecision(8) << cluster.score << ","
              << cluster.points.size() << "," << inference_points.size()
              << "," << std::setprecision(6)
              << cluster.center.x() << "," << cluster.center.y() << ","
              << cluster.center.z() << "," << local_preproc.validCount()
              << "," << local_preproc.validRatio()
              << "," << local_preproc.candidateCountScale()
              << "," << dir.string() << "\n";
      ++debug_dump_seq_;
    }

    ROS_INFO_THROTTLE(2.0, "[siamese_bpu] debug dumped %d candidate(s) to %s",
                      n_dump, cfg_.debug_dump_dir.c_str());
  }

  static void writeRawPointsCsv(const std::filesystem::path& path,
                                const Cluster& cluster) {
    writePointsCsv(path, cluster.points);
  }

  static void writePointsCsv(const std::filesystem::path& path,
                             const std::vector<PointXYZI>& points) {
    std::ofstream out(path);
    out << "x,y,z,reflectivity\n";
    out << std::fixed << std::setprecision(8);
    for (const auto& p : points) {
      out << p.x << "," << p.y << "," << p.z << ","
          << static_cast<int>(p.reflectivity) << "\n";
    }
  }

  static void writeCandidatePointCsv(const std::filesystem::path& path,
                                     const float* point) {
    std::ofstream out(path);
    out << "idx,x,y,z,rho,reflectivity_abs,reflectivity_rel\n";
    out << std::fixed << std::setprecision(8);
    for (int i = 0; i < siamese_bpu::BpuPreprocessor::kNumPoints; ++i) {
      out << i;
      for (int c = 0; c < siamese_bpu::BpuPreprocessor::kPointChannels; ++c) {
        out << "," << point[c * siamese_bpu::BpuPreprocessor::kNumPoints + i];
      }
      out << "\n";
    }
  }

  static void writeMaskCsv(const std::filesystem::path& path,
                           const float* mask) {
    std::ofstream out(path);
    out << "idx,value\n";
    out << std::fixed << std::setprecision(8);
    for (int i = 0; i < siamese_bpu::BpuPreprocessor::kNumPoints; ++i) {
      out << i << "," << mask[i] << "\n";
    }
  }

  static void writeMetaCsv(const std::filesystem::path& path,
                           const float* meta) {
    std::ofstream out(path);
    out << "idx,value\n";
    out << std::fixed << std::setprecision(8);
    for (int i = 0; i < siamese_bpu::BpuPreprocessor::kMetaDim; ++i) {
      out << i << "," << meta[i] << "\n";
    }
  }

  static void writeMetaYaml(const std::filesystem::path& path,
                            const ros::Time& stamp,
                            size_t candidate_idx,
                            int rank,
                            const Cluster& cluster,
                            size_t inference_points,
                            size_t valid_count,
                            float valid_ratio,
                            float count_scale) {
    std::ofstream out(path);
    out << std::fixed << std::setprecision(8);
    out << "stamp: " << stamp.toSec() << "\n";
    out << "candidate_index: " << candidate_idx << "\n";
    out << "rank: " << rank << "\n";
    out << "score: " << cluster.score << "\n";
    out << "raw_points: " << cluster.points.size() << "\n";
    out << "inference_points: " << inference_points << "\n";
    out << "valid_count: " << valid_count << "\n";
    out << "valid_ratio: " << valid_ratio << "\n";
    out << "count_scale: " << count_scale << "\n";
    out << "center: [" << cluster.center.x() << ", " << cluster.center.y()
        << ", " << cluster.center.z() << "]\n";
  }

  // ---- Members ----
  ros::NodeHandle nh_, pnh_;
  Config cfg_;
  siamese_bpu::BpuModel model_;
  siamese_bpu::BpuPreprocessor preproc_;

  ros::Subscriber sub_;
  ros::Subscriber pointcloud2_sub_;
  ros::Subscriber recfg_sub_;
  ros::Publisher bpu_scores_pub_;
  ros::Publisher marker_pub_;
  ros::Publisher cluster_cloud_pub_;

  ros::Time last_publish_time_;
  ros::Time last_debug_dump_time_;
  std::deque<std::vector<PointXYZI>> frame_buffer_;
  uint32_t window_id_ = 0;
  uint64_t debug_dump_seq_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "siamese_bpu_infer_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  SiameseBpuInferNode node(nh, pnh);
  ros::spin();
  return 0;
}
