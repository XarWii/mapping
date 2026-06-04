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
#include <std_msgs/Empty.h>

#include <livox_reflective_marker/BpuScores.h>
#include <livox_reflective_marker/BpuScoreEntry.h>
#include <livox_reflective_marker/EkfStatus.h>
#include <livox_reflective_marker/TargetCommand.h>

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
  bool color_by_confirmed_target = true;
  float confirmed_target_radius_m = 0.45f;
  double confirmed_target_timeout_sec = 0.0;
  bool color_follow_ekf_status = true;
  std::string target_command_topic = "/target_manager/target_command";
  std::string tracking_lost_topic = "/siamese_bpu_infer_node/tracking_lost";
  std::string ekf_status_topic = "/ekf_pose_node/ekf_status";
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
      c.center = Eigen::Vector3f::Zero();
      for (auto idx : indices) {
        c.points.push_back(points[idx]);
        c.center += Eigen::Vector3f(points[idx].x, points[idx].y, points[idx].z);
      }
      c.center /= static_cast<float>(indices.size());
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
    target_command_sub_ = nh_.subscribe(
        cfg_.target_command_topic, 4,
        &SiameseBpuInferNode::targetCommandCallback, this);
    tracking_lost_sub_ = nh_.subscribe(
        cfg_.tracking_lost_topic, 4,
        &SiameseBpuInferNode::trackingLostCallback, this);
    ekf_status_sub_ = nh_.subscribe(
        cfg_.ekf_status_topic, 4,
        &SiameseBpuInferNode::ekfStatusCallback, this);
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
    cfg_.confirmed_target_radius_m = cfg_.roi_radius_m;
    LOAD(pnh_, "color_by_confirmed_target", cfg_.color_by_confirmed_target);
    LOAD(pnh_, "confirmed_target_radius_m", cfg_.confirmed_target_radius_m);
    LOAD(pnh_, "confirmed_target_timeout_sec",
         cfg_.confirmed_target_timeout_sec);
    LOAD(pnh_, "color_follow_ekf_status", cfg_.color_follow_ekf_status);
    LOAD(pnh_, "target_command_topic", cfg_.target_command_topic);
    LOAD(pnh_, "tracking_lost_topic", cfg_.tracking_lost_topic);
    LOAD(pnh_, "ekf_status_topic", cfg_.ekf_status_topic);
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
    colored_cloud_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>(
        "colored_high_points", 4, true);
  }

  // ---- Main callback ---------------------------------------------------

  void publishColoredCloud(
      const std::vector<PointXYZI>& latest_points,
      const std::vector<Cluster>& scored_clusters,
      ros::Time stamp) {
    if (colored_cloud_pub_.getNumSubscribers() == 0) return;

    const float match_sq = cfg_.cluster_tolerance_m * cfg_.cluster_tolerance_m;
    const bool use_confirmed_target =
        cfg_.color_by_confirmed_target && isConfirmedTargetActive(stamp);
    const float confirmed_target_sq =
        cfg_.confirmed_target_radius_m * cfg_.confirmed_target_radius_m;

    sensor_msgs::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = cfg_.frame_id;
    cloud.height = 1;
    cloud.is_bigendian = false;
    cloud.is_dense = true;

    // 4 fields: x, y, z, rgb
    sensor_msgs::PointField f;
    f.datatype = sensor_msgs::PointField::FLOAT32;
    f.count = 1;

    f.name = "x";        f.offset = 0;  cloud.fields.push_back(f);
    f.name = "y";        f.offset = 4;  cloud.fields.push_back(f);
    f.name = "z";        f.offset = 8;  cloud.fields.push_back(f);
    f.name = "rgb";      f.offset = 12; cloud.fields.push_back(f);

    cloud.point_step = 16;
    cloud.width = static_cast<uint32_t>(latest_points.size());
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step);

    float* buf = reinterpret_cast<float*>(cloud.data.data());

    for (size_t i = 0; i < latest_points.size(); ++i) {
      const auto& pt = latest_points[i];
      Eigen::Vector3f pos(pt.x, pt.y, pt.z);

      bool is_target = false;
      if (use_confirmed_target) {
        is_target = (pos - confirmed_target_center_).squaredNorm() <=
                    confirmed_target_sq;
      } else if (!cfg_.color_by_confirmed_target) {
        // Legacy visualization: green if the point belongs to a high-score BPU
        // cluster. Confirmed-target coloring leaves everything yellow until a
        // final target command is received.
        for (const auto& c : scored_clusters) {
          if (c.score >= 0.3825f) {
            float d2 = (pos - c.center).squaredNorm();
            if (d2 <= match_sq) { is_target = true; break; }
          }
        }
      }

      uint8_t r, g, b;
      if (is_target) {
        r = 0;   g = 255; b = 0;    // green
      } else {
        r = 255; g = 200; b = 0;    // orange-yellow
      }

      uint32_t rgb_packed = (static_cast<uint32_t>(r) << 16) |
                            (static_cast<uint32_t>(g) << 8) |
                            static_cast<uint32_t>(b);
      float rgb_float;
      std::memcpy(&rgb_float, &rgb_packed, sizeof(float));

      buf[i * 4 + 0] = pt.x;
      buf[i * 4 + 1] = pt.y;
      buf[i * 4 + 2] = pt.z;
      buf[i * 4 + 3] = rgb_float;
    }

    colored_cloud_pub_.publish(cloud);
  }

  void targetCommandCallback(
      const livox_reflective_marker::TargetCommand::ConstPtr& cmd) {
    confirmed_target_center_ = Eigen::Vector3f(
        static_cast<float>(cmd->pose.pose.position.x),
        static_cast<float>(cmd->pose.pose.position.y),
        static_cast<float>(cmd->pose.pose.position.z));
    confirmed_target_stamp_ =
        cmd->header.stamp.isZero() ? ros::Time::now() : cmd->header.stamp;
    have_confirmed_target_ = true;
    ROS_INFO("[siamese_bpu] confirmed target coloring center=[%.2f, %.2f, %.2f]",
             static_cast<double>(confirmed_target_center_.x()),
             static_cast<double>(confirmed_target_center_.y()),
             static_cast<double>(confirmed_target_center_.z()));
  }

  void trackingLostCallback(const std_msgs::Empty::ConstPtr&) {
    have_confirmed_target_ = false;
    confirmed_target_stamp_ = ros::Time();
    confirmed_target_center_ = Eigen::Vector3f::Zero();
    ROS_WARN("[siamese_bpu] confirmed target coloring cleared: tracking lost");
  }

  void ekfStatusCallback(
      const livox_reflective_marker::EkfStatus::ConstPtr& status) {
    if (!cfg_.color_follow_ekf_status) return;

    if (status->state == 1) {
      if (have_confirmed_target_) {
        have_confirmed_target_ = false;
        confirmed_target_stamp_ = ros::Time();
        confirmed_target_center_ = Eigen::Vector3f::Zero();
        ROS_WARN("[siamese_bpu] confirmed target coloring cleared: EKF lost");
      }
      return;
    }

    confirmed_target_center_ = Eigen::Vector3f(
        static_cast<float>(status->current_pose.pose.position.x),
        static_cast<float>(status->current_pose.pose.position.y),
        static_cast<float>(status->current_pose.pose.position.z));
    confirmed_target_stamp_ =
        status->header.stamp.isZero() ? ros::Time::now() : status->header.stamp;
    have_confirmed_target_ = true;
    ROS_INFO_THROTTLE(
        2.0,
        "[siamese_bpu] coloring follows EKF center=[%.2f, %.2f, %.2f]",
        static_cast<double>(confirmed_target_center_.x()),
        static_cast<double>(confirmed_target_center_.y()),
        static_cast<double>(confirmed_target_center_.z()));
  }

  bool isConfirmedTargetActive(const ros::Time& stamp) const {
    if (!have_confirmed_target_) return false;
    if (cfg_.confirmed_target_timeout_sec <= 0.0) return true;

    const ros::Time reference = stamp.isZero() ? ros::Time::now() : stamp;
    const double age = (reference - confirmed_target_stamp_).toSec();
    return age <= cfg_.confirmed_target_timeout_sec;
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
    window_id_++;

    // 3. Throttle
    if (!last_publish_time_.isZero() &&
        (now - last_publish_time_).toSec() < cfg_.publish_interval_sec)
      return;
    last_publish_time_ = now;

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

    // 6. Refresh centres from the latest frame to eliminate lag.
    std::vector<Eigen::Vector3f> latest_centers;
    if (!frame_buffer_.empty() && !frame_buffer_.back().empty()) {
      auto latest_clusters = EuclideanCluster(frame_buffer_.back(), cfg_);
      for (auto& c : latest_clusters)
        latest_centers.push_back(c.center);
    }
    if (!latest_centers.empty()) {
      const float match_sq = cfg_.cluster_tolerance_m * cfg_.cluster_tolerance_m;
      for (auto& c : clusters) {
        int best = -1;
        float best_d2 = match_sq;
        for (size_t j = 0; j < latest_centers.size(); ++j) {
          float d2 = (c.center - latest_centers[j]).squaredNorm();
          if (d2 < best_d2) { best_d2 = d2; best = static_cast<int>(j); }
        }
        if (best >= 0)
          c.center = latest_centers[best];
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

    // 9. Colored point cloud — latest-frame points, green for target / yellow for rest
    if (!frame_buffer_.empty())
      publishColoredCloud(frame_buffer_.back(), clusters, now);
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
  ros::Subscriber target_command_sub_;
  ros::Subscriber tracking_lost_sub_;
  ros::Subscriber ekf_status_sub_;
  ros::Publisher bpu_scores_pub_;
  ros::Publisher marker_pub_;
  ros::Publisher colored_cloud_pub_;

  ros::Time last_publish_time_;
  std::deque<std::vector<PointXYZI>> frame_buffer_;
  uint32_t window_id_ = 0;
  bool have_confirmed_target_ = false;
  Eigen::Vector3f confirmed_target_center_ = Eigen::Vector3f::Zero();
  ros::Time confirmed_target_stamp_;
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
