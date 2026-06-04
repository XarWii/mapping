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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
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
  std::string bpu_model_path;

  uint8_t reflectivity_threshold = 250;
  float min_distance_m = 0.1f, max_distance_m = 30.0f;

  float cluster_tolerance_m = 0.25f;
  size_t min_cluster_points = 3, max_cluster_points = 2000;

  int max_accumulation_frames = 20;
  double publish_interval_sec = 0.1;
  bool publish_markers = true;
  std::string frame_id = "livox_frame";

  float roi_radius_m = 0.45f;
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
    sub_ = nh_.subscribe(cfg_.input_topic, 4,
                         &SiameseBpuInferNode::pointCloudCallback, this);
    recfg_sub_ = nh_.subscribe(
        "/target_manager/recognition_command", 4,
        &SiameseBpuInferNode::recognitionCommandCallback, this);
    ROS_INFO("[siamese_bpu] pure scorer ready | model: %s",
             cfg_.bpu_model_path.c_str());
  }

 private:
  void loadConfig() {
#define LOAD(ns, name, var) ns.param(name, var, var)
    pnh_.param<std::string>("input_topic", cfg_.input_topic, cfg_.input_topic);
    pnh_.param<std::string>("bpu_model_path", cfg_.bpu_model_path, cfg_.bpu_model_path);

    int rf = static_cast<int>(cfg_.reflectivity_threshold);
    pnh_.param("reflectivity_threshold", rf, rf);
    cfg_.reflectivity_threshold = static_cast<uint8_t>(rf);

    LOAD(pnh_, "min_distance_m", cfg_.min_distance_m);
    LOAD(pnh_, "max_distance_m", cfg_.max_distance_m);
    LOAD(pnh_, "cluster_tolerance_m", cfg_.cluster_tolerance_m);

    int mn = static_cast<int>(cfg_.min_cluster_points);
    int mx = static_cast<int>(cfg_.max_cluster_points);
    pnh_.param("min_cluster_points", mn, mn);
    pnh_.param("max_cluster_points", mx, mx);
    cfg_.min_cluster_points = static_cast<size_t>(mn);
    cfg_.max_cluster_points = static_cast<size_t>(mx);

    LOAD(pnh_, "max_accumulation_frames", cfg_.max_accumulation_frames);
    LOAD(pnh_, "publish_interval_sec", cfg_.publish_interval_sec);
    LOAD(pnh_, "publish_markers", cfg_.publish_markers);
    LOAD(pnh_, "roi_radius_m", cfg_.roi_radius_m);
    pnh_.param<std::string>("frame_id", cfg_.frame_id, cfg_.frame_id);

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
    ros::Time now = msg->header.stamp;

    // 1. Extract high-reflectivity points
    std::vector<PointXYZI> high_pts;
    high_pts.reserve(msg->points.size());
    for (const auto& raw : msg->points) {
      PointXYZI p = ToPoint(raw);
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
    if (!accumulated.empty()) {
      clusters = EuclideanCluster(accumulated, cfg_);
      for (auto& c : clusters) {
        auto eigen_pts = ClusterToEigen(c);
        preproc_.normalize(eigen_pts);
        preproc_.fixedSize(eigen_pts);
        c.score = model_.infer(preproc_.candidateNchw(), preproc_.validRatio());
      }
    }

    // 6. Cluster the latest frame once; reuse for centre refresh AND cluster-level coloring.
    std::vector<Cluster> latest_clusters;
    if (!frame_buffer_.empty() && !frame_buffer_.back().empty()) {
      latest_clusters = EuclideanCluster(frame_buffer_.back(), cfg_);
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

    // 8. Markers
    if (cfg_.publish_markers)
      publishMarkers(clusters, now);

    // 9. Publish latest-frame cluster cloud for target_manager's visualisation.
    if (!latest_clusters.empty() || !frame_buffer_.empty()) {
      publishClusterCloud(latest_clusters,
                          frame_buffer_.empty()
                              ? std::vector<PointXYZI>()
                              : frame_buffer_.back(),
                          now);
    }
  }

  void publishClusterCloud(const std::vector<Cluster>& latest_clusters,
                           const std::vector<PointXYZI>& latest_points,
                           ros::Time stamp) {
    if (cluster_cloud_pub_.getNumSubscribers() == 0) return;

    livox_reflective_marker::ClusterCloud msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = cfg_.frame_id;
    msg.window_id = window_id_;

    // Build a raw xyz PointCloud2 from latest_points
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
    cloud.width = static_cast<uint32_t>(latest_points.size());
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step);

    float* buf = reinterpret_cast<float*>(cloud.data.data());
    for (size_t i = 0; i < latest_points.size(); ++i) {
      buf[i * 3 + 0] = latest_points[i].x;
      buf[i * 3 + 1] = latest_points[i].y;
      buf[i * 3 + 2] = latest_points[i].z;
    }
    msg.cloud = cloud;

    // per_point_cluster_id: -1 = orphan, 0..N-1 = valid cluster
    msg.per_point_cluster_id.resize(latest_points.size(), -1);
    for (size_t ci = 0; ci < latest_clusters.size(); ++ci) {
      for (size_t idx : latest_clusters[ci].indices) {
        if (idx < msg.per_point_cluster_id.size())
          msg.per_point_cluster_id[idx] = static_cast<int32_t>(ci);
      }
    }

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
      if (s >= 0.6f) {
        m.color.r = 0.2f; m.color.g = 1.0f; m.color.b = 0.3f;
      } else if (s >= 0.3825f) {
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

  // ---- Members ----
  ros::NodeHandle nh_, pnh_;
  Config cfg_;
  siamese_bpu::BpuModel model_;
  siamese_bpu::BpuPreprocessor preproc_;

  ros::Subscriber sub_;
  ros::Subscriber recfg_sub_;
  ros::Publisher bpu_scores_pub_;
  ros::Publisher marker_pub_;
  ros::Publisher cluster_cloud_pub_;

  ros::Time last_publish_time_;
  std::deque<std::vector<PointXYZI>> frame_buffer_;
  uint32_t window_id_ = 0;
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
