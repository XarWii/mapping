#include <livox_reflective_marker/RecognitionScoreEntry.h>
#include <livox_reflective_marker/RecognitionScores.h>
#include <livox_reflective_marker/ReflectiveCandidate.h>
#include <livox_reflective_marker/ReflectiveCandidates.h>
#include <livox_reflective_marker/ReflectiveObservationFrame.h>
#include <livox_reflective_marker/ReflectiveRecognitionRequest.h>
#include <livox_reflective_marker/pointcloud2_codec.h>
#include <ros/ros.h>

#include "siamese_bpu_infer.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint8_t kStatusScored = 0;
constexpr uint8_t kStatusRawUnavailable = 1;
constexpr uint8_t kStatusInsufficientWindows = 2;
constexpr uint8_t kStatusInsufficientPoints = 3;
constexpr uint8_t kStatusNoMatchingCluster = 4;
constexpr uint8_t kStatusBudgetDeferred = 5;
constexpr uint8_t kStatusStaleVersion = 7;
constexpr uint8_t kStatusWindowBudgetDeferred = 8;

struct Config {
  std::string candidates_topic = "/reflective/candidates";
  std::string observation_frame_topic = "/reflective/observation_frame";
  std::string recognition_request_topic = "/reflective/recognition_request";
  std::string scores_topic = "/reflective/recognition_scores";
  std::string bpu_model_path;
  uint64_t expected_observation_source_contract_hash = 0;
  float reflectivity_threshold = 160.0f;
  int raw_ring_max_frames = 80;
  int window_size_frames = 10;
  int max_candidate_window_pairs_per_request = 64;
  int max_window_points = 5000;
  int min_cluster_points = 3;
  int max_cluster_points = 2000;
  int min_inference_points = 8;
  int min_roi_cluster_points = 3;
  float cluster_tolerance_m = 0.25f;
  float roi_padding_m = 0.03f;
  bool allow_truncated_frames = false;
};

struct Point {
  Eigen::Vector3f anchor = Eigen::Vector3f::Zero();
  Eigen::Vector3f map = Eigen::Vector3f::Zero();
  float reflectivity = 0.0f;
};

struct Cluster {
  std::vector<size_t> indices;
  Eigen::Vector3f center_anchor = Eigen::Vector3f::Zero();
  Eigen::Vector3f center_map = Eigen::Vector3f::Zero();
};

struct ObservationPoint {
  Eigen::Vector3f lidar = Eigen::Vector3f::Zero();
  Eigen::Vector3f map = Eigen::Vector3f::Zero();
  float reflectivity = 0.0f;
};

struct ObservationFrame {
  std_msgs::Header header;
  uint32_t map_epoch = 0;
  uint64_t scan_id = 0;
  uint64_t contract_hash = 0;
  Eigen::Isometry3f t_map_lidar = Eigen::Isometry3f::Identity();
  Eigen::Vector3f aabb_min = Eigen::Vector3f::Zero();
  Eigen::Vector3f aabb_max = Eigen::Vector3f::Zero();
  bool truncated = false;
  std::vector<ObservationPoint> points;
};

struct CandidateLite {
  uint32_t candidate_id = 0;
  uint32_t support_revision = 0;
  geometry_msgs::Point center_map_msg;
  Eigen::Vector3f center_map = Eigen::Vector3f::Zero();
  Eigen::Vector3f roi_min = Eigen::Vector3f::Zero();
  Eigen::Vector3f roi_max = Eigen::Vector3f::Zero();
  uint32_t voxel_count = 0;
  uint32_t evidence_count = 0;
};

Eigen::Vector3f ToEigen(const geometry_msgs::Point& point) {
  return Eigen::Vector3f(static_cast<float>(point.x),
                         static_cast<float>(point.y),
                         static_cast<float>(point.z));
}

geometry_msgs::Point ToMsg(const Eigen::Vector3f& point) {
  geometry_msgs::Point out;
  out.x = point.x();
  out.y = point.y();
  out.z = point.z();
  return out;
}

Eigen::Isometry3f PoseToIsometry(const geometry_msgs::Pose& pose) {
  Eigen::Quaternionf q(static_cast<float>(pose.orientation.w),
                       static_cast<float>(pose.orientation.x),
                       static_cast<float>(pose.orientation.y),
                       static_cast<float>(pose.orientation.z));
  if (!std::isfinite(q.norm()) || q.norm() < 1.0e-6f) {
    q = Eigen::Quaternionf::Identity();
  } else {
    q.normalize();
  }
  Eigen::Isometry3f out = Eigen::Isometry3f::Identity();
  out.linear() = q.toRotationMatrix();
  out.translation() = Eigen::Vector3f(static_cast<float>(pose.position.x),
                                      static_cast<float>(pose.position.y),
                                      static_cast<float>(pose.position.z));
  return out;
}

bool IntersectsAabb(const Eigen::Vector3f& a_min, const Eigen::Vector3f& a_max,
                    const Eigen::Vector3f& b_min, const Eigen::Vector3f& b_max) {
  return a_min.x() <= b_max.x() && a_max.x() >= b_min.x() &&
         a_min.y() <= b_max.y() && a_max.y() >= b_min.y() &&
         a_min.z() <= b_max.z() && a_max.z() >= b_min.z();
}

bool InsideAabb(const Eigen::Vector3f& p, const Eigen::Vector3f& min_p,
                const Eigen::Vector3f& max_p) {
  return p.x() >= min_p.x() && p.x() <= max_p.x() &&
         p.y() >= min_p.y() && p.y() <= max_p.y() &&
         p.z() >= min_p.z() && p.z() <= max_p.z();
}

bool SameSnapshot(const std_msgs::Header& first, uint32_t first_epoch,
                  const std_msgs::Header& second, uint32_t second_epoch) {
  return first_epoch == second_epoch && first.stamp == second.stamp &&
         first.frame_id == second.frame_id;
}

float SquaredDistance(const Point& a, const Point& b) {
  return (a.anchor - b.anchor).squaredNorm();
}

std::vector<Cluster> EuclideanCluster(const std::vector<Point>& points,
                                      float tolerance_m,
                                      int min_cluster_points,
                                      int max_cluster_points) {
  std::vector<Cluster> clusters;
  if (points.empty()) return clusters;

  const float tol_sq = tolerance_m * tolerance_m;
  std::vector<uint8_t> visited(points.size(), 0);
  std::vector<size_t> queue;
  queue.reserve(points.size());

  for (size_t i = 0; i < points.size(); ++i) {
    if (visited[i]) continue;
    visited[i] = 1;
    queue.clear();
    queue.push_back(i);

    std::vector<size_t> indices;
    for (size_t head = 0; head < queue.size(); ++head) {
      const size_t current = queue[head];
      indices.push_back(current);
      for (size_t j = 0; j < points.size(); ++j) {
        if (!visited[j] && SquaredDistance(points[current], points[j]) <= tol_sq) {
          visited[j] = 1;
          queue.push_back(j);
        }
      }
    }

    if (static_cast<int>(indices.size()) < min_cluster_points) continue;
    if (max_cluster_points > 0 &&
        static_cast<int>(indices.size()) > max_cluster_points) {
      continue;
    }

    Cluster cluster;
    cluster.indices = std::move(indices);
    for (const size_t idx : cluster.indices) {
      cluster.center_anchor += points[idx].anchor;
      cluster.center_map += points[idx].map;
    }
    const float inv_count = 1.0f / static_cast<float>(cluster.indices.size());
    cluster.center_anchor *= inv_count;
    cluster.center_map *= inv_count;
    clusters.push_back(std::move(cluster));
  }

  std::sort(clusters.begin(), clusters.end(), [](const Cluster& a, const Cluster& b) {
    return a.indices.size() > b.indices.size();
  });
  return clusters;
}

class ReflectiveObservationBpuScorer {
 public:
  ReflectiveObservationBpuScorer() : private_nh_("~"), config_(LoadConfig()) {
    ValidateConfig();
    ROS_INFO("reflective observation BPU scorer loading model: %s",
             config_.bpu_model_path.c_str());
    model_.init(config_.bpu_model_path);

    candidates_subscriber_ = nh_.subscribe(
        config_.candidates_topic, 1,
        &ReflectiveObservationBpuScorer::HandleCandidates, this,
        ros::TransportHints().tcpNoDelay());
    observation_subscriber_ = nh_.subscribe(
        config_.observation_frame_topic, 8,
        &ReflectiveObservationBpuScorer::HandleObservationFrame, this,
        ros::TransportHints().tcpNoDelay());
    request_subscriber_ = nh_.subscribe(
        config_.recognition_request_topic, 2,
        &ReflectiveObservationBpuScorer::HandleRecognitionRequest, this,
        ros::TransportHints().tcpNoDelay());
    scores_publisher_ = nh_.advertise<livox_reflective_marker::RecognitionScores>(
        config_.scores_topic, 1, true);

    ROS_INFO("reflective observation BPU scorer ready: candidates=%s observations=%s request=%s",
             config_.candidates_topic.c_str(),
             config_.observation_frame_topic.c_str(),
             config_.recognition_request_topic.c_str());
  }

 private:
  Config LoadConfig() {
    Config config;
    private_nh_.param("candidates_topic", config.candidates_topic,
                      config.candidates_topic);
    private_nh_.param("observation_frame_topic", config.observation_frame_topic,
                      config.observation_frame_topic);
    private_nh_.param("recognition_request_topic",
                      config.recognition_request_topic,
                      config.recognition_request_topic);
    private_nh_.param("scores_topic", config.scores_topic, config.scores_topic);
    private_nh_.param("bpu_model_path", config.bpu_model_path,
                      config.bpu_model_path);
    int contract_hash = static_cast<int>(config.expected_observation_source_contract_hash);
    private_nh_.param("expected_observation_source_contract_hash",
                      contract_hash, contract_hash);
    config.expected_observation_source_contract_hash =
        static_cast<uint64_t>(std::max(0, contract_hash));
    private_nh_.param("reflectivity_threshold", config.reflectivity_threshold,
                      config.reflectivity_threshold);
    private_nh_.param("raw_ring_max_frames", config.raw_ring_max_frames,
                      config.raw_ring_max_frames);
    private_nh_.param("window_size_frames", config.window_size_frames,
                      config.window_size_frames);
    private_nh_.param("max_candidate_window_pairs_per_request",
                      config.max_candidate_window_pairs_per_request,
                      config.max_candidate_window_pairs_per_request);
    private_nh_.param("max_window_points", config.max_window_points,
                      config.max_window_points);
    private_nh_.param("min_cluster_points", config.min_cluster_points,
                      config.min_cluster_points);
    private_nh_.param("max_cluster_points", config.max_cluster_points,
                      config.max_cluster_points);
    private_nh_.param("min_inference_points", config.min_inference_points,
                      config.min_inference_points);
    private_nh_.param("min_roi_cluster_points", config.min_roi_cluster_points,
                      config.min_roi_cluster_points);
    private_nh_.param("cluster_tolerance_m", config.cluster_tolerance_m,
                      config.cluster_tolerance_m);
    private_nh_.param("roi_padding_m", config.roi_padding_m,
                      config.roi_padding_m);
    private_nh_.param("allow_truncated_frames", config.allow_truncated_frames,
                      config.allow_truncated_frames);
    return config;
  }

  void ValidateConfig() const {
    if (config_.bpu_model_path.empty() || config_.raw_ring_max_frames < 1 ||
        config_.window_size_frames < 1 || config_.min_cluster_points < 1 ||
        config_.min_inference_points < 1 || config_.cluster_tolerance_m <= 0.0f ||
        config_.max_candidate_window_pairs_per_request < 1 ||
        config_.max_window_points < 1) {
      throw std::runtime_error("invalid reflective_observation_bpu_scorer parameters");
    }
  }

  bool DecodeCloud(const livox_reflective_marker::ReflectiveObservationFrame& msg,
                   std::vector<ObservationPoint>* points) const {
    const sensor_msgs::PointCloud2& cloud = msg.cloud_lidar;
    if (cloud.is_bigendian || !livox_reflective_marker::pointcloud2::HasCompleteRows(cloud)) {
      return false;
    }
    const sensor_msgs::PointField* x =
        livox_reflective_marker::pointcloud2::FindField(cloud, "x");
    const sensor_msgs::PointField* y =
        livox_reflective_marker::pointcloud2::FindField(cloud, "y");
    const sensor_msgs::PointField* z =
        livox_reflective_marker::pointcloud2::FindField(cloud, "z");
    const sensor_msgs::PointField* intensity =
        livox_reflective_marker::pointcloud2::FindField(cloud, "intensity");
    if (!intensity) {
      intensity = livox_reflective_marker::pointcloud2::FindField(cloud, "reflectivity");
    }
    if (!x || !y || !z || !intensity) return false;

    const Eigen::Isometry3f t_map_lidar = PoseToIsometry(msg.lidar_pose_in_map);
    points->clear();
    points->reserve(static_cast<size_t>(cloud.width) * cloud.height);
    for (uint32_t row = 0; row < cloud.height; ++row) {
      const size_t row_start = static_cast<size_t>(row) * cloud.row_step;
      for (uint32_t col = 0; col < cloud.width; ++col) {
        const uint8_t* source =
            cloud.data.data() + row_start + static_cast<size_t>(col) * cloud.point_step;
        float px = 0.0f;
        float py = 0.0f;
        float pz = 0.0f;
        float reflectivity = 0.0f;
        if (!livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *x, &px) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *y, &py) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *z, &pz) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *intensity,
                                                                     &reflectivity)) {
          continue;
        }
        ObservationPoint point;
        point.lidar = Eigen::Vector3f(px, py, pz);
        point.map = t_map_lidar * point.lidar;
        point.reflectivity = reflectivity;
        points->push_back(point);
      }
    }
    return true;
  }

  void HandleObservationFrame(
      const livox_reflective_marker::ReflectiveObservationFrame::ConstPtr& msg) {
    ObservationFrame frame;
    frame.header = msg->header;
    frame.map_epoch = msg->map_epoch;
    frame.scan_id = msg->scan_id;
    frame.contract_hash = msg->observation_source_contract_hash;
    frame.t_map_lidar = PoseToIsometry(msg->lidar_pose_in_map);
    frame.aabb_min = ToEigen(msg->map_aabb_min);
    frame.aabb_max = ToEigen(msg->map_aabb_max);
    frame.truncated = msg->truncated;
    if (!DecodeCloud(*msg, &frame.points)) {
      ROS_WARN_THROTTLE(2.0, "reflective observation scorer rejected malformed observation frame");
      return;
    }
    observation_ring_.push_back(std::move(frame));
    while (static_cast<int>(observation_ring_.size()) > config_.raw_ring_max_frames) {
      observation_ring_.pop_front();
    }
  }

  void HandleCandidates(
      const livox_reflective_marker::ReflectiveCandidates::ConstPtr& candidates) {
    latest_candidates_ = *candidates;
    candidate_cache_.clear();
    candidate_cache_.reserve(candidates->candidates.size());
    for (const auto& candidate : candidates->candidates) {
      CandidateLite lite;
      lite.candidate_id = candidate.candidate_id;
      lite.support_revision = candidate.support_revision;
      lite.center_map_msg = candidate.center_map;
      lite.center_map = ToEigen(candidate.center_map);
      lite.roi_min = ToEigen(candidate.roi_min_map);
      lite.roi_max = ToEigen(candidate.roi_max_map);
      lite.voxel_count = candidate.voxel_count;
      lite.evidence_count = candidate.evidence_count;
      candidate_cache_.push_back(lite);
    }
    have_candidates_ = true;
    TryScorePendingRequest();
  }

  void HandleRecognitionRequest(
      const livox_reflective_marker::ReflectiveRecognitionRequest::ConstPtr& request) {
    pending_request_ = *request;
    have_pending_request_ = true;
    TryScorePendingRequest();
  }

  void TryScorePendingRequest() {
    if (!have_pending_request_ || !have_candidates_) return;
    if (latest_candidates_.candidate_snapshot_id !=
            pending_request_.candidate_snapshot_id ||
        !SameSnapshot(latest_candidates_.header, latest_candidates_.map_epoch,
                      pending_request_.header, pending_request_.map_epoch)) {
      return;
    }
    const livox_reflective_marker::ReflectiveRecognitionRequest request =
        pending_request_;
    have_pending_request_ = false;
    ScoreRequest(request);
  }

  const CandidateLite* FindCandidate(uint32_t candidate_id,
                                     uint32_t support_revision) const {
    for (const auto& candidate : candidate_cache_) {
      if (candidate.candidate_id == candidate_id &&
          candidate.support_revision == support_revision) {
        return &candidate;
      }
    }
    return nullptr;
  }

  bool RequestIncludesCandidate(
      const livox_reflective_marker::ReflectiveRecognitionRequest& request,
      size_t index,
      uint32_t* candidate_id,
      uint32_t* support_revision) const {
    if (index >= request.candidate_ids.size() ||
        index >= request.support_revisions.size()) {
      return false;
    }
    *candidate_id = request.candidate_ids[index];
    *support_revision = request.support_revisions[index];
    return true;
  }

  bool FrameCompatible(const ObservationFrame& frame, uint32_t map_epoch) const {
    if (frame.map_epoch != map_epoch) return false;
    if (!config_.allow_truncated_frames && frame.truncated) return false;
    if (config_.expected_observation_source_contract_hash != 0 &&
        frame.contract_hash != config_.expected_observation_source_contract_hash) {
      return false;
    }
    return true;
  }

  bool BuildWindow(size_t end_index, const CandidateLite& candidate,
                   uint32_t map_epoch, std::vector<Point>* points,
                   uint32_t* windows_seen) const {
    if (end_index + 1 < static_cast<size_t>(config_.window_size_frames)) {
      return false;
    }
    const size_t begin = end_index + 1 -
                         static_cast<size_t>(config_.window_size_frames);
    const Eigen::Vector3f padding =
        Eigen::Vector3f::Constant(config_.roi_padding_m);
    const Eigen::Vector3f roi_min = candidate.roi_min - padding;
    const Eigen::Vector3f roi_max = candidate.roi_max + padding;

    bool has_roi_overlap = false;
    size_t point_count = 0;
    for (size_t i = begin; i <= end_index; ++i) {
      const ObservationFrame& frame = observation_ring_[i];
      if (!FrameCompatible(frame, map_epoch)) return false;
      if (i > begin && observation_ring_[i - 1].scan_id + 1 != frame.scan_id) {
        return false;
      }
      if (IntersectsAabb(frame.aabb_min, frame.aabb_max, roi_min, roi_max)) {
        has_roi_overlap = true;
      }
      point_count += frame.points.size();
    }
    if (windows_seen != nullptr) ++(*windows_seen);
    if (!has_roi_overlap) return false;
    if (point_count > static_cast<size_t>(config_.max_window_points)) {
      points->clear();
      return true;
    }

    const Eigen::Isometry3f t_anchor_map =
        observation_ring_[end_index].t_map_lidar.inverse();
    points->clear();
    points->reserve(point_count);
    for (size_t i = begin; i <= end_index; ++i) {
      for (const auto& source : observation_ring_[i].points) {
        Point point;
        point.map = source.map;
        point.anchor = t_anchor_map * source.map;
        point.reflectivity = source.reflectivity;
        points->push_back(point);
      }
    }
    return true;
  }

  bool SelectClusterForCandidate(const CandidateLite& candidate,
                                 const std::vector<Point>& points,
                                 Cluster* selected,
                                 size_t* roi_points) const {
    const std::vector<Cluster> clusters =
        EuclideanCluster(points, config_.cluster_tolerance_m,
                         config_.min_cluster_points,
                         config_.max_cluster_points);
    const Eigen::Vector3f padding =
        Eigen::Vector3f::Constant(config_.roi_padding_m);
    const Eigen::Vector3f roi_min = candidate.roi_min - padding;
    const Eigen::Vector3f roi_max = candidate.roi_max + padding;

    int best_index = -1;
    size_t best_overlap = 0;
    for (size_t i = 0; i < clusters.size(); ++i) {
      size_t overlap = 0;
      for (const size_t point_index : clusters[i].indices) {
        if (InsideAabb(points[point_index].map, roi_min, roi_max)) {
          ++overlap;
        }
      }
      if (overlap > best_overlap) {
        best_overlap = overlap;
        best_index = static_cast<int>(i);
      }
    }
    if (best_index < 0 ||
        best_overlap < static_cast<size_t>(config_.min_roi_cluster_points)) {
      return false;
    }
    *selected = clusters[static_cast<size_t>(best_index)];
    if (roi_points != nullptr) *roi_points = best_overlap;
    return true;
  }

  livox_reflective_marker::RecognitionScoreEntry MakeStatusEntry(
      const CandidateLite* candidate, uint32_t candidate_id,
      uint32_t support_revision, uint8_t status) const {
    livox_reflective_marker::RecognitionScoreEntry entry;
    entry.candidate_id = candidate_id;
    entry.support_revision = support_revision;
    entry.status = status;
    entry.score_valid = false;
    entry.windows_used = 0;
    if (candidate != nullptr) {
      entry.center_map = candidate->center_map_msg;
      entry.center_sensor = candidate->center_map_msg;
      entry.voxel_count = candidate->voxel_count;
      entry.evidence_count = candidate->evidence_count;
    }
    entry.center_sensor_valid = false;
    return entry;
  }

  livox_reflective_marker::RecognitionScoreEntry ScoreCandidate(
      const CandidateLite& candidate, uint32_t map_epoch,
      int* pair_budget_remaining) {
    if (*pair_budget_remaining <= 0) {
      return MakeStatusEntry(&candidate, candidate.candidate_id,
                             candidate.support_revision, kStatusBudgetDeferred);
    }

    bool saw_compatible_window = false;
    bool saw_roi_window = false;
    bool budget_deferred = false;
    uint32_t compatible_windows = 0;
    std::vector<Point> window_points;

    for (size_t offset = 0; offset < observation_ring_.size(); ++offset) {
      const size_t end_index = observation_ring_.size() - 1 - offset;
      if (!FrameCompatible(observation_ring_[end_index], map_epoch)) continue;
      saw_compatible_window = true;

      window_points.clear();
      const bool built = BuildWindow(end_index, candidate, map_epoch,
                                     &window_points, &compatible_windows);
      if (!built) continue;
      saw_roi_window = true;
      if (window_points.empty()) {
        budget_deferred = true;
        continue;
      }
      --(*pair_budget_remaining);

      Cluster cluster;
      size_t roi_points = 0;
      if (!SelectClusterForCandidate(candidate, window_points, &cluster,
                                     &roi_points)) {
        continue;
      }

      std::vector<siamese_bpu::BpuPoint> bpu_points;
      bpu_points.reserve(cluster.indices.size());
      for (const size_t point_index : cluster.indices) {
        siamese_bpu::BpuPoint point;
        point.xyz = window_points[point_index].anchor;
        point.reflectivity = window_points[point_index].reflectivity;
        bpu_points.push_back(point);
      }
      preprocessor_.prepare(bpu_points, config_.reflectivity_threshold);
      if (preprocessor_.validCount() <
          static_cast<size_t>(config_.min_inference_points)) {
        livox_reflective_marker::RecognitionScoreEntry entry =
            MakeStatusEntry(&candidate, candidate.candidate_id,
                            candidate.support_revision,
                            kStatusInsufficientPoints);
        entry.center_sensor = ToMsg(cluster.center_anchor);
        entry.center_sensor_valid = true;
        entry.center_sensor_frame_id = observation_ring_[end_index].header.frame_id;
        entry.center_sensor_stamp = observation_ring_[end_index].header.stamp;
        return entry;
      }

      livox_reflective_marker::RecognitionScoreEntry entry;
      entry.candidate_id = candidate.candidate_id;
      entry.support_revision = candidate.support_revision;
      entry.status = kStatusScored;
      entry.score_valid = true;
      entry.windows_used = 1;
      entry.score = model_.infer(preprocessor_.candidatePoint(),
                                 preprocessor_.candidateMask(),
                                 preprocessor_.candidateMeta(),
                                 preprocessor_.candidateCountScale());
      entry.center_map = ToMsg(cluster.center_map);
      entry.center_sensor = ToMsg(cluster.center_anchor);
      entry.center_sensor_valid = true;
      entry.center_sensor_frame_id = observation_ring_[end_index].header.frame_id;
      entry.center_sensor_stamp = observation_ring_[end_index].header.stamp;
      entry.voxel_count = candidate.voxel_count;
      entry.evidence_count = candidate.evidence_count;
      ROS_INFO("observation BPU candidate=%u window_end_scan=%lu cluster_points=%zu "
               "roi_points=%zu valid_points=%zu score=%.4f",
               candidate.candidate_id,
               static_cast<unsigned long>(observation_ring_[end_index].scan_id),
               cluster.indices.size(), roi_points, preprocessor_.validCount(),
               static_cast<double>(entry.score));
      return entry;
    }

    if (budget_deferred) {
      return MakeStatusEntry(&candidate, candidate.candidate_id,
                             candidate.support_revision,
                             kStatusWindowBudgetDeferred);
    }
    if (!saw_compatible_window) {
      return MakeStatusEntry(&candidate, candidate.candidate_id,
                             candidate.support_revision, kStatusRawUnavailable);
    }
    if (!saw_roi_window || compatible_windows == 0) {
      return MakeStatusEntry(&candidate, candidate.candidate_id,
                             candidate.support_revision,
                             kStatusInsufficientWindows);
    }
    return MakeStatusEntry(&candidate, candidate.candidate_id,
                           candidate.support_revision, kStatusNoMatchingCluster);
  }

  void ScoreRequest(
      const livox_reflective_marker::ReflectiveRecognitionRequest& request) {
    livox_reflective_marker::RecognitionScores scores;
    scores.header = latest_candidates_.header;
    scores.map_epoch = latest_candidates_.map_epoch;
    scores.request_id = request.request_id;
    scores.candidate_snapshot_id = request.candidate_snapshot_id;

    int pair_budget_remaining = config_.max_candidate_window_pairs_per_request;
    const size_t count =
        std::min(request.candidate_ids.size(), request.support_revisions.size());
    scores.entries.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      uint32_t candidate_id = 0;
      uint32_t support_revision = 0;
      if (!RequestIncludesCandidate(request, i, &candidate_id,
                                    &support_revision)) {
        continue;
      }
      const CandidateLite* candidate =
          FindCandidate(candidate_id, support_revision);
      if (candidate == nullptr) {
        scores.entries.push_back(MakeStatusEntry(
            nullptr, candidate_id, support_revision, kStatusStaleVersion));
        continue;
      }
      scores.entries.push_back(ScoreCandidate(*candidate, request.map_epoch,
                                              &pair_budget_remaining));
    }
    scores_publisher_.publish(scores);
    ROS_INFO("observation BPU request=%lu snapshot=%lu entries=%zu ring=%zu budget_left=%d",
             static_cast<unsigned long>(request.request_id),
             static_cast<unsigned long>(request.candidate_snapshot_id),
             scores.entries.size(), observation_ring_.size(),
             pair_budget_remaining);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  Config config_;
  ros::Subscriber candidates_subscriber_;
  ros::Subscriber observation_subscriber_;
  ros::Subscriber request_subscriber_;
  ros::Publisher scores_publisher_;
  bool have_candidates_ = false;
  bool have_pending_request_ = false;
  livox_reflective_marker::ReflectiveCandidates latest_candidates_;
  livox_reflective_marker::ReflectiveRecognitionRequest pending_request_;
  std::vector<CandidateLite> candidate_cache_;
  std::deque<ObservationFrame> observation_ring_;
  siamese_bpu::BpuModel model_;
  siamese_bpu::BpuPreprocessor preprocessor_;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "reflective_observation_bpu_scorer");
  try {
    ReflectiveObservationBpuScorer node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("reflective observation BPU scorer initialization failed: %s",
              error.what());
    return 1;
  }
  return 0;
}
