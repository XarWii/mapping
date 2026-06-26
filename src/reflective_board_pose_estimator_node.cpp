#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <geometry_msgs/PoseStamped.h>
#include <livox_reflective_marker/ReflectiveBoardPose.h>
#include <livox_reflective_marker/ReflectiveCandidate.h>
#include <livox_reflective_marker/ReflectiveTargetState.h>
#include <livox_reflective_marker/pointcloud2_codec.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <visualization_msgs/Marker.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint8_t kDecisionTracking = 1;

constexpr uint8_t kPoseValid = 0;
constexpr uint8_t kPoseStale = 1;
constexpr uint8_t kPoseLost = 2;
constexpr uint8_t kPoseInvalidGeometry = 3;

struct Config {
  std::string active_candidate_topic = "/reflective/active_candidate";
  std::string target_state_topic = "/reflective/target_state";
  std::string target_pose_topic = "/reflective/target_pose";
  std::string target_pose_status_topic = "/reflective/target_pose_status";
  std::string normal_marker_topic = "/reflective/target_normal_marker";
  double normal_marker_length_m = 0.35;
  int min_points = 8;
  float max_plane_rms_m = 0.06f;
  float min_inplane_eigen_ratio = 1.10f;
  float max_center_jump_m = 0.30f;
  float max_normal_change_deg = 35.0f;
  double stale_timeout_sec = 4.0;
};

struct FitResult {
  geometry_msgs::Pose pose;
  float plane_rms_m = 0.0f;
  float inplane_eigen_ratio = 0.0f;
  uint32_t point_count = 0;
  bool yaw_observable = false;
};

bool SameSnapshot(const std_msgs::Header& first, const std_msgs::Header& second) {
  return first.stamp == second.stamp && first.frame_id == second.frame_id;
}

double Clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(upper, value));
}

class ReflectiveBoardPoseEstimator {
 public:
  ReflectiveBoardPoseEstimator() : private_nh_("~"), config_(LoadConfig()) {
    if (config_.min_points < 3 || config_.max_plane_rms_m <= 0.0f ||
        config_.min_inplane_eigen_ratio < 1.0f ||
        config_.max_center_jump_m <= 0.0f ||
        config_.max_normal_change_deg <= 0.0f ||
        config_.stale_timeout_sec <= 0.0 || config_.normal_marker_length_m <= 0.0) {
      throw std::runtime_error("invalid reflective_board_pose_estimator parameters");
    }
    candidate_subscriber_ = nh_.subscribe(
        config_.active_candidate_topic, 2,
        &ReflectiveBoardPoseEstimator::HandleActiveCandidate, this,
        ros::TransportHints().tcpNoDelay());
    state_subscriber_ = nh_.subscribe(
        config_.target_state_topic, 2,
        &ReflectiveBoardPoseEstimator::HandleTargetState, this,
        ros::TransportHints().tcpNoDelay());
    pose_publisher_ = nh_.advertise<geometry_msgs::PoseStamped>(
        config_.target_pose_topic, 1);
    status_publisher_ = nh_.advertise<livox_reflective_marker::ReflectiveBoardPose>(
        config_.target_pose_status_topic, 1, true);
    normal_marker_publisher_ = nh_.advertise<visualization_msgs::Marker>(
        config_.normal_marker_topic, 1, true);
    stale_timer_ = private_nh_.createTimer(
        ros::Duration(0.25), &ReflectiveBoardPoseEstimator::HandleStaleTimer, this);
    ROS_INFO("reflective board pose estimator: candidate=%s state=%s pose=%s",
             config_.active_candidate_topic.c_str(), config_.target_state_topic.c_str(),
             config_.target_pose_topic.c_str());
  }

 private:
  Config LoadConfig() {
    Config config;
    private_nh_.param("active_candidate_topic", config.active_candidate_topic,
                      config.active_candidate_topic);
    private_nh_.param("target_state_topic", config.target_state_topic,
                      config.target_state_topic);
    private_nh_.param("target_pose_topic", config.target_pose_topic,
                      config.target_pose_topic);
    private_nh_.param("target_pose_status_topic", config.target_pose_status_topic,
                      config.target_pose_status_topic);
    private_nh_.param("normal_marker_topic", config.normal_marker_topic,
                      config.normal_marker_topic);
    private_nh_.param("normal_marker_length_m", config.normal_marker_length_m,
                      config.normal_marker_length_m);
    private_nh_.param("min_points", config.min_points, config.min_points);
    private_nh_.param("max_plane_rms_m", config.max_plane_rms_m,
                      config.max_plane_rms_m);
    private_nh_.param("min_inplane_eigen_ratio", config.min_inplane_eigen_ratio,
                      config.min_inplane_eigen_ratio);
    private_nh_.param("max_center_jump_m", config.max_center_jump_m,
                      config.max_center_jump_m);
    private_nh_.param("max_normal_change_deg", config.max_normal_change_deg,
                      config.max_normal_change_deg);
    private_nh_.param("stale_timeout_sec", config.stale_timeout_sec,
                      config.stale_timeout_sec);
    return config;
  }

  bool DecodeMapCloud(const sensor_msgs::PointCloud2& cloud,
                      std::vector<Eigen::Vector3f>* points) const {
    if (cloud.is_bigendian || cloud.point_step == 0) return false;
    const auto* x = livox_reflective_marker::pointcloud2::FindField(cloud, "x");
    const auto* y = livox_reflective_marker::pointcloud2::FindField(cloud, "y");
    const auto* z = livox_reflective_marker::pointcloud2::FindField(cloud, "z");
    if (!x || !y || !z) return false;
    const size_t count = static_cast<size_t>(cloud.width) * cloud.height;
    if (!livox_reflective_marker::pointcloud2::HasCompleteRows(cloud)) {
      return false;
    }
    points->clear();
    points->reserve(count);
    for (uint32_t row = 0; row < cloud.height; ++row) {
      const size_t row_offset = static_cast<size_t>(row) * cloud.row_step;
      for (uint32_t column = 0; column < cloud.width; ++column) {
        const uint8_t* source = cloud.data.data() + row_offset +
                                static_cast<size_t>(column) * cloud.point_step;
        float px = 0.0f;
        float py = 0.0f;
        float pz = 0.0f;
        if (!livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *x, &px) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *y, &py) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *z, &pz)) {
          continue;
        }
        points->emplace_back(px, py, pz);
      }
    }
    return true;
  }

  static void MakeDeterministicAxisSign(Eigen::Vector3f* axis) {
    int dominant = 0;
    if (std::fabs(axis->y()) > std::fabs(axis->x())) dominant = 1;
    if (std::fabs(axis->z()) > std::fabs((*axis)[dominant])) dominant = 2;
    if ((*axis)[dominant] < 0.0f) *axis = -*axis;
  }

  bool FitPose(const livox_reflective_marker::ReflectiveCandidate& candidate,
               FitResult* result, std::string* reason) const {
    std::vector<Eigen::Vector3f> points;
    if (!DecodeMapCloud(candidate.cloud_map, &points)) {
      *reason = "malformed_map_cloud";
      return false;
    }
    if (points.size() < static_cast<size_t>(config_.min_points)) {
      *reason = "not_enough_points";
      return false;
    }

    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& point : points) center += point;
    center /= static_cast<float>(points.size());
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& point : points) {
      const Eigen::Vector3f delta = point - center;
      covariance.noalias() += delta * delta.transpose();
    }
    covariance /= static_cast<float>(points.size());

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) {
      *reason = "plane_eigendecomposition_failed";
      return false;
    }
    const Eigen::Vector3f values = solver.eigenvalues();
    Eigen::Vector3f normal = solver.eigenvectors().col(0).normalized();
    Eigen::Vector3f major_axis = solver.eigenvectors().col(2).normalized();
    const float plane_rms = std::sqrt(std::max(0.0f, values.x()));
    if (!std::isfinite(plane_rms) || plane_rms > config_.max_plane_rms_m) {
      *reason = "plane_rms_too_large";
      return false;
    }

    const Eigen::Vector3f lidar_position(
        static_cast<float>(candidate.lidar_pose_in_map.position.x),
        static_cast<float>(candidate.lidar_pose_in_map.position.y),
        static_cast<float>(candidate.lidar_pose_in_map.position.z));
    const Eigen::Vector3f toward_lidar = lidar_position - center;
    if (toward_lidar.squaredNorm() > 1e-8f && normal.dot(toward_lidar) < 0.0f) {
      normal = -normal;
    }
    major_axis -= normal * normal.dot(major_axis);
    if (major_axis.squaredNorm() < 1e-8f) {
      *reason = "inplane_axis_degenerate";
      return false;
    }
    major_axis.normalize();
    if (HaveComparableLastPose(candidate.candidate_id)) {
      const Eigen::Quaternionf previous(
          static_cast<float>(last_pose_.orientation.w),
          static_cast<float>(last_pose_.orientation.x),
          static_cast<float>(last_pose_.orientation.y),
          static_cast<float>(last_pose_.orientation.z));
      if (previous.normalized().toRotationMatrix().col(0).dot(major_axis) < 0.0f) {
        major_axis = -major_axis;
      }
    } else {
      MakeDeterministicAxisSign(&major_axis);
    }
    Eigen::Vector3f minor_axis = normal.cross(major_axis);
    if (minor_axis.squaredNorm() < 1e-8f) {
      *reason = "minor_axis_degenerate";
      return false;
    }
    minor_axis.normalize();
    Eigen::Matrix3f rotation;
    rotation.col(0) = major_axis;
    rotation.col(1) = minor_axis;
    rotation.col(2) = normal;
    const Eigen::Quaternionf orientation(rotation);
    if (!orientation.coeffs().allFinite()) {
      *reason = "nonfinite_orientation";
      return false;
    }

    result->pose.position.x = center.x();
    result->pose.position.y = center.y();
    result->pose.position.z = center.z();
    result->pose.orientation.x = orientation.x();
    result->pose.orientation.y = orientation.y();
    result->pose.orientation.z = orientation.z();
    result->pose.orientation.w = orientation.w();
    result->plane_rms_m = plane_rms;
    result->point_count = static_cast<uint32_t>(points.size());
    result->inplane_eigen_ratio = values.z() /
                                  std::max(values.y(), std::numeric_limits<float>::epsilon());
    result->yaw_observable =
        result->inplane_eigen_ratio >= config_.min_inplane_eigen_ratio;
    return true;
  }

  bool HaveComparableLastPose(uint32_t candidate_id) const {
    return have_last_pose_ && last_candidate_id_ == candidate_id &&
           last_epoch_ == latest_state_.map_epoch;
  }

  bool PassesJumpGates(const FitResult& fit, std::string* reason) const {
    if (!HaveComparableLastPose(latest_candidate_.candidate_id)) return true;
    const Eigen::Vector3f current_position(
        static_cast<float>(fit.pose.position.x), static_cast<float>(fit.pose.position.y),
        static_cast<float>(fit.pose.position.z));
    const Eigen::Vector3f previous_position(
        static_cast<float>(last_pose_.position.x),
        static_cast<float>(last_pose_.position.y),
        static_cast<float>(last_pose_.position.z));
    if ((current_position - previous_position).norm() > config_.max_center_jump_m) {
      *reason = "center_jump_rejected";
      return false;
    }
    const Eigen::Quaternionf current_orientation(
        static_cast<float>(fit.pose.orientation.w), static_cast<float>(fit.pose.orientation.x),
        static_cast<float>(fit.pose.orientation.y), static_cast<float>(fit.pose.orientation.z));
    const Eigen::Quaternionf previous_orientation(
        static_cast<float>(last_pose_.orientation.w),
        static_cast<float>(last_pose_.orientation.x),
        static_cast<float>(last_pose_.orientation.y),
        static_cast<float>(last_pose_.orientation.z));
    const float cosine = Clamp(
        previous_orientation.normalized().toRotationMatrix().col(2).dot(
            current_orientation.normalized().toRotationMatrix().col(2)),
        -1.0, 1.0);
    const float angle_deg = static_cast<float>(std::acos(cosine) * 180.0 / M_PI);
    if (angle_deg > config_.max_normal_change_deg) {
      *reason = "normal_jump_rejected";
      return false;
    }
    return true;
  }

  void HandleActiveCandidate(
      const livox_reflective_marker::ReflectiveCandidate::ConstPtr& candidate) {
    latest_candidate_ = *candidate;
    have_candidate_ = true;
    TryEstimate();
  }

  void HandleTargetState(
      const livox_reflective_marker::ReflectiveTargetState::ConstPtr& state) {
    if (have_state_ && state->map_epoch != latest_state_.map_epoch) {
      have_last_pose_ = false;
      stale_reported_ = false;
    }
    latest_state_ = *state;
    have_state_ = true;
    if (state->state != kDecisionTracking) {
      PublishNormalMarkerDelete(state->header);
      if (have_last_pose_) {
        PublishStatus(state->header, state->map_epoch, last_candidate_id_, kPoseLost,
                      last_pose_, last_plane_rms_m_, last_point_count_,
                      last_inplane_eigen_ratio_, last_yaw_observable_, "decision_lost");
      } else {
        PublishStatus(state->header, state->map_epoch, state->candidate_id, kPoseLost,
                      geometry_msgs::Pose(), 0.0f, 0, 0.0f, false, "decision_lost");
      }
      return;
    }
    TryEstimate();
  }

  void TryEstimate() {
    if (!have_candidate_ || !have_state_ ||
        latest_state_.state != kDecisionTracking ||
        latest_candidate_.candidate_id != latest_state_.candidate_id ||
        !SameSnapshot(latest_candidate_.cloud_map.header, latest_state_.header)) {
      return;
    }
    FitResult fit;
    std::string reason;
    if (!FitPose(latest_candidate_, &fit, &reason) || !PassesJumpGates(fit, &reason)) {
      PublishNormalMarkerDelete(latest_state_.header);
      PublishStatus(latest_state_.header, latest_state_.map_epoch,
                    latest_state_.candidate_id, kPoseInvalidGeometry,
                    have_last_pose_ ? last_pose_ : geometry_msgs::Pose(),
                    have_last_pose_ ? last_plane_rms_m_ : 0.0f,
                    have_last_pose_ ? last_point_count_ : 0,
                    have_last_pose_ ? last_inplane_eigen_ratio_ : 0.0f,
                    have_last_pose_ && last_yaw_observable_, reason);
      ROS_WARN_THROTTLE(2.0, "reflective board pose rejected: %s", reason.c_str());
      return;
    }

    have_last_pose_ = true;
    last_epoch_ = latest_state_.map_epoch;
    last_candidate_id_ = latest_candidate_.candidate_id;
    last_pose_ = fit.pose;
    last_plane_rms_m_ = fit.plane_rms_m;
    last_point_count_ = fit.point_count;
    last_inplane_eigen_ratio_ = fit.inplane_eigen_ratio;
    last_yaw_observable_ = fit.yaw_observable;
    last_observation_time_ = ros::Time::now();
    stale_reported_ = false;

    geometry_msgs::PoseStamped pose;
    pose.header = latest_state_.header;
    pose.pose = fit.pose;
    pose_publisher_.publish(pose);
    PublishNormalMarker(pose);
    PublishStatus(latest_state_.header, latest_state_.map_epoch,
                  latest_candidate_.candidate_id, kPoseValid, fit.pose,
                  fit.plane_rms_m, fit.point_count, fit.inplane_eigen_ratio,
                  fit.yaw_observable,
                  fit.yaw_observable ? "valid" : "valid_yaw_ambiguous");
  }

  void HandleStaleTimer(const ros::TimerEvent&) {
    if (!have_state_ || latest_state_.state != kDecisionTracking ||
        !have_last_pose_ || stale_reported_ || last_observation_time_.isZero() ||
        (ros::Time::now() - last_observation_time_).toSec() <= config_.stale_timeout_sec) {
      return;
    }
    stale_reported_ = true;
    PublishNormalMarkerDelete(latest_state_.header);
    PublishStatus(latest_state_.header, last_epoch_, last_candidate_id_, kPoseStale,
                  last_pose_, last_plane_rms_m_, last_point_count_,
                  last_inplane_eigen_ratio_, last_yaw_observable_, "observation_timeout");
  }

  void PublishStatus(const std_msgs::Header& header, uint32_t epoch,
                     uint32_t candidate_id, uint8_t status,
                     const geometry_msgs::Pose& pose, float plane_rms_m,
                     uint32_t point_count, float inplane_eigen_ratio,
                     bool yaw_observable, const std::string& reason) {
    livox_reflective_marker::ReflectiveBoardPose output;
    output.header = header;
    output.map_epoch = epoch;
    output.candidate_id = candidate_id;
    output.status = status;
    output.pose = pose;
    output.plane_rms_m = plane_rms_m;
    output.point_count = point_count;
    output.inplane_eigen_ratio = inplane_eigen_ratio;
    output.yaw_observable = yaw_observable;
    output.reason = reason;
    status_publisher_.publish(output);
  }

  void PublishNormalMarker(const geometry_msgs::PoseStamped& pose) {
    const Eigen::Quaterniond orientation(
        pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y,
        pose.pose.orientation.z);
    if (orientation.norm() < 1e-8) return;
    const Eigen::Vector3d normal =
        orientation.normalized().toRotationMatrix().col(2);

    visualization_msgs::Marker marker;
    marker.header = pose.header;
    marker.ns = "reflective_target_normal";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.025;
    marker.scale.y = 0.060;
    marker.scale.z = 0.080;
    marker.color.r = 0.1f;
    marker.color.g = 0.95f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0f;

    geometry_msgs::Point start = pose.pose.position;
    geometry_msgs::Point end;
    end.x = start.x + normal.x() * config_.normal_marker_length_m;
    end.y = start.y + normal.y() * config_.normal_marker_length_m;
    end.z = start.z + normal.z() * config_.normal_marker_length_m;
    marker.points.push_back(start);
    marker.points.push_back(end);
    normal_marker_publisher_.publish(marker);
  }

  void PublishNormalMarkerDelete(const std_msgs::Header& header) {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "reflective_target_normal";
    marker.id = 0;
    marker.action = visualization_msgs::Marker::DELETE;
    marker.pose.orientation.w = 1.0;
    normal_marker_publisher_.publish(marker);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  Config config_;
  ros::Subscriber candidate_subscriber_;
  ros::Subscriber state_subscriber_;
  ros::Publisher pose_publisher_;
  ros::Publisher status_publisher_;
  ros::Publisher normal_marker_publisher_;
  ros::Timer stale_timer_;

  bool have_candidate_ = false;
  bool have_state_ = false;
  livox_reflective_marker::ReflectiveCandidate latest_candidate_;
  livox_reflective_marker::ReflectiveTargetState latest_state_;
  bool have_last_pose_ = false;
  uint32_t last_epoch_ = 0;
  uint32_t last_candidate_id_ = 0;
  geometry_msgs::Pose last_pose_;
  float last_plane_rms_m_ = 0.0f;
  uint32_t last_point_count_ = 0;
  float last_inplane_eigen_ratio_ = 0.0f;
  bool last_yaw_observable_ = false;
  ros::Time last_observation_time_;
  bool stale_reported_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "reflective_board_pose_estimator");
  try {
    ReflectiveBoardPoseEstimator node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("reflective board pose estimator initialization failed: %s", error.what());
    return 1;
  }
  return 0;
}
