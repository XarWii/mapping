/// @file traditional_validator_node.cpp
/// @brief Validates BPU-proposed candidates using traditional template matching.
///
/// Receives CandidateValidationRequest, accumulates points in the specified ROI,
/// runs PCA-based pose estimation + template matching, and publishes
/// CandidateValidationResult.  Operates in "validation-only" mode: it does NOT
/// do its own discovery or tracking.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <std_msgs/Header.h>

#include <Eigen/Dense>

#include <livox_ros_driver2/CustomMsg.h>
#include <livox_reflective_marker/CandidateValidationRequest.h>
#include <livox_reflective_marker/CandidateValidationResult.h>

namespace {

// ---------------------------------------------------------------------------
// Point / geometry helpers (reused from reflective_board_identifier_node)
// ---------------------------------------------------------------------------

struct PointXYZI {
  float x = 0.0f, y = 0.0f, z = 0.0f;
  uint8_t reflectivity = 0;
};

struct TimedPoint {
  PointXYZI point;
  ros::Time stamp;
};

struct PlaneTemplate {
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
  std::vector<PointXYZI> inliers;
  float rms_m = 0.0f;
};

struct BoardPose {
  geometry_msgs::Pose pose;
  Eigen::Vector3f x_axis = Eigen::Vector3f::UnitX();
  Eigen::Vector3f y_axis = Eigen::Vector3f::UnitY();
  Eigen::Vector3f z_axis = Eigen::Vector3f::UnitZ();
};

struct TemplateModel {
  bool loaded = false;
  std::vector<std::vector<float>> descriptor_variants;
  std::vector<std::vector<float>> mid_descriptor_variants;
  std::vector<std::vector<float>> coarse_descriptor_variants;
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct ValidatorConfig {
  std::string input_topic = "/livox/lidar";
  std::string frame_id = "livox_frame";
  std::string validation_request_topic = "/siamese_bpu_infer_node/validation_request";
  std::string validation_result_topic = "/siamese_bpu_infer_node/validation_result";

  uint8_t reflectivity_threshold = 250;
  float min_distance_m = 0.1f;
  float max_distance_m = 30.0f;

  double validation_accumulation_sec = 1.5;
  double validation_timeout_sec = 3.0;
  double validation_confirm_score_threshold = 0.80;
  double validation_confirm_margin_threshold = 0.30;

  std::string template_path;
  size_t template_grid_cols = 48;
  size_t template_grid_rows = 48;
  size_t template_mid_grid_cols = 32;
  size_t template_mid_grid_rows = 32;
  size_t template_coarse_grid_cols = 24;
  size_t template_coarse_grid_rows = 24;
  size_t min_template_points = 30;
  size_t target_template_match_points = 300;
  size_t far_template_match_points = 120;
  float far_template_distance_m = 8.0f;
  double min_template_score = 0.35;
  double far_min_template_score = 0.28;
  double good_template_score = 0.50;
  float template_plane_tolerance_m = 0.045f;
  double confirm_score_threshold = 0.75;
  double confirm_margin_threshold = 0.30;
};

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

inline Eigen::Vector3f ToEigen(const PointXYZI& p) {
  return {p.x, p.y, p.z};
}

inline bool IsFinite(float v) { return std::isfinite(v); }

inline bool IsValidPoint(const PointXYZI& p) {
  return IsFinite(p.x) && IsFinite(p.y) && IsFinite(p.z);
}

inline bool IsReflectiveCandidate(const PointXYZI& p, const ValidatorConfig& cfg) {
  if (!IsValidPoint(p)) return false;
  float d2 = p.x * p.x + p.y * p.y + p.z * p.z;
  float d = std::sqrt(d2);
  return p.reflectivity >= cfg.reflectivity_threshold &&
         d >= cfg.min_distance_m && d <= cfg.max_distance_m;
}

inline float Clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }

inline double Square(double x) { return x * x; }

int FieldOffset(const sensor_msgs::PointCloud2& cloud, const std::string& name) {
  for (const auto& field : cloud.fields) {
    if (field.name == name) return static_cast<int>(field.offset);
  }
  return -1;
}

bool ReadFloat32(const sensor_msgs::PointCloud2& cloud, size_t base, int offset, float* value) {
  if (value == nullptr || offset < 0 || cloud.is_bigendian) return false;
  const size_t pos = base + static_cast<size_t>(offset);
  if (pos + sizeof(float) > cloud.data.size()) return false;
  std::memcpy(value, cloud.data.data() + pos, sizeof(float));
  return std::isfinite(*value);
}

std::vector<PointXYZI> ExtractSeedCloudPoints(const sensor_msgs::PointCloud2& cloud,
                                              const Eigen::Vector3f& roi_center,
                                              float roi_radius,
                                              const ValidatorConfig& cfg) {
  std::vector<PointXYZI> points;
  if (cloud.width == 0 || cloud.height == 0 || cloud.point_step == 0 ||
      cloud.row_step == 0 || cloud.is_bigendian) {
    return points;
  }

  const int x_off = FieldOffset(cloud, "x");
  const int y_off = FieldOffset(cloud, "y");
  const int z_off = FieldOffset(cloud, "z");
  const int intensity_off = FieldOffset(cloud, "intensity");
  const int reflectivity_off = FieldOffset(cloud, "reflectivity");
  if (x_off < 0 || y_off < 0 || z_off < 0) return points;

  const float roi_sq = roi_radius * roi_radius;
  points.reserve(static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height));
  for (uint32_t row = 0; row < cloud.height; ++row) {
    const size_t row_base = static_cast<size_t>(row) * cloud.row_step;
    for (uint32_t col = 0; col < cloud.width; ++col) {
      const size_t base = row_base + static_cast<size_t>(col) * cloud.point_step;
      if (base + cloud.point_step > cloud.data.size()) continue;

      PointXYZI p;
      if (!ReadFloat32(cloud, base, x_off, &p.x) ||
          !ReadFloat32(cloud, base, y_off, &p.y) ||
          !ReadFloat32(cloud, base, z_off, &p.z)) {
        continue;
      }

      float intensity = 255.0f;
      if (intensity_off >= 0) {
        ReadFloat32(cloud, base, intensity_off, &intensity);
      } else if (reflectivity_off >= 0) {
        ReadFloat32(cloud, base, reflectivity_off, &intensity);
      }
      intensity = std::max(0.0f, std::min(255.0f, intensity));
      p.reflectivity = static_cast<uint8_t>(std::round(intensity));

      if (!IsReflectiveCandidate(p, cfg)) continue;
      if ((ToEigen(p) - roi_center).squaredNorm() > roi_sq) continue;
      points.push_back(p);
    }
  }
  return points;
}

float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0f;
  double dot = 0.0, an = 0.0, bn = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    an += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    bn += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  double denom = std::sqrt(an * bn);
  if (denom < 1e-12) return 0.0f;
  return static_cast<float>(Clamp01(static_cast<float>(dot / denom)));
}

std::vector<std::vector<float>> BuildDescriptorVariants(
    const std::vector<Eigen::Vector2f>& uv, size_t cols, size_t rows) {
  // Build a single-grid binary descriptor; "variants" = the original + flipped.
  std::vector<float> descriptor(cols * rows, 0.0f);
  if (uv.empty()) return {descriptor};

  float min_u = uv[0].x(), max_u = uv[0].x();
  float min_v = uv[0].y(), max_v = uv[0].y();
  for (const auto& p : uv) {
    min_u = std::min(min_u, p.x()); max_u = std::max(max_u, p.x());
    min_v = std::min(min_v, p.y()); max_v = std::max(max_v, p.y());
  }
  float range_u = max_u - min_u, range_v = max_v - min_v;
  if (range_u < 1e-6f) range_u = 1.0f;
  if (range_v < 1e-6f) range_v = 1.0f;

  for (const auto& p : uv) {
    int cu = static_cast<int>((p.x() - min_u) / range_u * static_cast<float>(cols));
    int cv = static_cast<int>((p.y() - min_v) / range_v * static_cast<float>(rows));
    cu = std::max(0, std::min(static_cast<int>(cols) - 1, cu));
    cv = std::max(0, std::min(static_cast<int>(rows) - 1, cv));
    descriptor[static_cast<size_t>(cv) * cols + static_cast<size_t>(cu)] = 1.0f;
  }

  auto flip_u = [&](const std::vector<float>& desc) {
    std::vector<float> flipped(desc.size());
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < cols; ++c)
        flipped[r * cols + c] = desc[r * cols + (cols - 1 - c)];
    return flipped;
  };
  auto flip_v = [&](const std::vector<float>& desc) {
    std::vector<float> flipped(desc.size());
    for (size_t r = 0; r < rows; ++r)
      for (size_t c = 0; c < cols; ++c)
        flipped[r * cols + c] = desc[(rows - 1 - r) * cols + c];
    return flipped;
  };

  return {descriptor, flip_u(descriptor), flip_v(descriptor),
          flip_u(flip_v(descriptor))};
}

geometry_msgs::Quaternion QuaternionFromAxes(const Eigen::Vector3f& x,
                                              const Eigen::Vector3f& y,
                                              const Eigen::Vector3f& z) {
  Eigen::Matrix3f rot;
  rot.col(0) = x; rot.col(1) = y; rot.col(2) = z;
  Eigen::Quaternionf q(rot);
  q.normalize();
  geometry_msgs::Quaternion msg;
  msg.x = q.x(); msg.y = q.y(); msg.z = q.z(); msg.w = q.w();
  return msg;
}

// ---------------------------------------------------------------------------
// Template model loading (supports points_local_uv YAML format and points_uv format)
// ---------------------------------------------------------------------------

// Trim leading whitespace from a string view
std::string TrimLeft(const std::string& s) {
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
  return s.substr(i);
}

// Parse a YAML list entry like "[0.056139, 0.105206, 255]" or "0.056139 0.105206"
bool ParseUVEntry(const std::string& line, float* u, float* v) {
  std::string trimmed = TrimLeft(line);
  if (trimmed.empty()) return false;

  // Format: "- [u, v, ...]"  (YAML points_local_uv)
  if (trimmed[0] == '-' && trimmed.size() > 1 && trimmed[1] != '-' &&
      trimmed.find('[') != std::string::npos) {
    auto lb = trimmed.find('[');
    auto rb = trimmed.find(']');
    if (lb == std::string::npos || rb == std::string::npos) return false;
    std::string inner = trimmed.substr(lb + 1, rb - lb - 1);
    // Replace commas with spaces
    for (auto& ch : inner)
      if (ch == ',') ch = ' ';
    std::istringstream iss(inner);
    return static_cast<bool>(iss >> *u >> *v);
  }

  // Format: "0.056139 0.105206" (space-separated UV)
  std::istringstream iss(trimmed);
  return static_cast<bool>(iss >> *u >> *v);
}

bool LoadTemplateModel(const ValidatorConfig& cfg, TemplateModel* model) {
  if (cfg.template_path.empty()) {
    ROS_WARN("[trad_validator] template_path is empty");
    return false;
  }
  std::ifstream f(cfg.template_path);
  if (!f.is_open()) {
    ROS_WARN("[trad_validator] cannot open template: %s", cfg.template_path.c_str());
    return false;
  }

  std::vector<Eigen::Vector2f> points_uv;
  std::string line;
  bool in_points = false;
  while (std::getline(f, line)) {
    // Skip empty and comment lines
    if (line.empty()) continue;
    std::string trimmed = TrimLeft(line);
    if (trimmed.empty() || trimmed[0] == '#') continue;

    // Detect section start
    if (trimmed.find("points_local_uv:") == 0 || trimmed == "points_uv") {
      in_points = true;
      continue;
    }

    if (!in_points) continue;

    // End-of-section detection
    if (trimmed == "end_points_uv" ||
        (!trimmed.empty() && trimmed[0] != '-' && trimmed.find(':') != std::string::npos)) {
      break;
    }

    float u = 0, v = 0;
    if (ParseUVEntry(line, &u, &v))
      points_uv.emplace_back(u, v);
  }
  f.close();

  if (points_uv.size() < 5) {
    ROS_WARN("[trad_validator] template has too few UV points (%zu)", points_uv.size());
    return false;
  }

  model->descriptor_variants =
      BuildDescriptorVariants(points_uv, cfg.template_grid_cols, cfg.template_grid_rows);
  model->mid_descriptor_variants =
      BuildDescriptorVariants(points_uv, cfg.template_mid_grid_cols, cfg.template_mid_grid_rows);
  model->coarse_descriptor_variants =
      BuildDescriptorVariants(points_uv, cfg.template_coarse_grid_cols, cfg.template_coarse_grid_rows);
  model->loaded = true;
  ROS_INFO("[trad_validator] template loaded (%zu UV points)", points_uv.size());
  return true;
}

// ---------------------------------------------------------------------------
// PCA-based pose estimation (from reflective_board_identifier_node)
// ---------------------------------------------------------------------------

bool EstimatePoseFromHighPoints(const std::vector<PointXYZI>& high_points,
                                BoardPose* pose, size_t min_points) {
  if (pose == nullptr || high_points.size() < min_points) return false;

  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  for (const auto& p : high_points) origin += ToEigen(p);
  origin /= static_cast<float>(high_points.size());

  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  for (const auto& p : high_points) {
    Eigen::Vector3f d = ToEigen(p) - origin;
    cov += d * d.transpose();
  }
  cov /= static_cast<float>(high_points.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
  if (solver.info() != Eigen::Success) return false;

  Eigen::Vector3f z_axis = solver.eigenvectors().col(0).normalized();
  Eigen::Vector3f x_axis = solver.eigenvectors().col(2).normalized();
  x_axis = x_axis - x_axis.dot(z_axis) * z_axis;
  if (x_axis.norm() < 1e-4f) return false;
  x_axis.normalize();

  Eigen::Vector3f y_axis = z_axis.cross(x_axis);
  if (y_axis.norm() < 1e-4f) return false;
  y_axis.normalize();

  Eigen::Vector3f to_lidar = -origin;
  if (to_lidar.norm() > 1e-4f && z_axis.dot(to_lidar.normalized()) < 0.0f) {
    z_axis = -z_axis;
    y_axis = -y_axis;
  }

  pose->pose.position.x = origin.x();
  pose->pose.position.y = origin.y();
  pose->pose.position.z = origin.z();
  pose->pose.orientation = QuaternionFromAxes(x_axis, y_axis, z_axis);
  pose->x_axis = x_axis;
  pose->y_axis = y_axis;
  pose->z_axis = z_axis;
  return true;
}

// ---------------------------------------------------------------------------
// Template scoring
// ---------------------------------------------------------------------------

double TemplateScore(const std::vector<PointXYZI>& high_points,
                     const BoardPose& board,
                     const TemplateModel& model,
                     const ValidatorConfig& cfg) {
  if (!model.loaded || high_points.size() < cfg.min_template_points) return 0.0;

  Eigen::Vector3f origin(board.pose.position.x, board.pose.position.y,
                         board.pose.position.z);
  std::vector<Eigen::Vector2f> uv;
  std::vector<Eigen::Vector2f> uv_swapped;  // 90° rotation: swap x/y axes
  uv.reserve(high_points.size());
  uv_swapped.reserve(high_points.size());
  float max_offset = std::max(0.08f, 2.0f * cfg.template_plane_tolerance_m);
  for (const auto& p : high_points) {
    Eigen::Vector3f rel = ToEigen(p) - origin;
    if (std::fabs(rel.dot(board.z_axis)) > max_offset) continue;
    uv.emplace_back(rel.dot(board.x_axis), rel.dot(board.y_axis));
    uv_swapped.emplace_back(rel.dot(board.y_axis), rel.dot(board.x_axis));
  }
  if (uv.size() < cfg.min_template_points) return 0.0;

  size_t cols = cfg.template_grid_cols, rows = cfg.template_grid_rows;
  const std::vector<std::vector<float>>* variants = &model.descriptor_variants;
  const std::vector<std::vector<float>>* variants_swapped = &model.descriptor_variants;
  if (uv.size() < 0.5 * cfg.target_template_match_points) {
    cols = cfg.template_coarse_grid_cols; rows = cfg.template_coarse_grid_rows;
    variants = &model.coarse_descriptor_variants;
    variants_swapped = variants;
  } else if (uv.size() < cfg.target_template_match_points) {
    cols = cfg.template_mid_grid_cols; rows = cfg.template_mid_grid_rows;
    variants = &model.mid_descriptor_variants;
    variants_swapped = variants;
  }

  auto candidates = BuildDescriptorVariants(uv, cols, rows);
  auto candidates_swapped = BuildDescriptorVariants(uv_swapped, cols, rows);
  double best = 0.0;
  for (const auto& c : candidates)
    for (const auto& t : *variants)
      best = std::max(best, static_cast<double>(CosineSimilarity(c, t)));
  for (const auto& c : candidates_swapped)
    for (const auto& t : *variants_swapped)
      best = std::max(best, static_cast<double>(CosineSimilarity(c, t)));
  return best;
}

double CalibratedTemplateScore(double raw, double min_score, const ValidatorConfig& cfg) {
  if (raw < min_score) return 0.0;
  double norm = (raw - min_score) / std::max(1e-6, cfg.good_template_score - min_score);
  double confidence = Clamp01(static_cast<float>(norm));
  return cfg.confirm_score_threshold +
         (1.0 - cfg.confirm_score_threshold) * confidence;
}

size_t RequiredTemplatePoints(double distance_m, const ValidatorConfig& cfg) {
  if (distance_m >= cfg.far_template_distance_m) return cfg.far_template_match_points;
  double t = Clamp01(static_cast<float>(distance_m / std::max(1e-3, static_cast<double>(cfg.far_template_distance_m))));
  return static_cast<size_t>(std::round((1.0 - t) * cfg.target_template_match_points +
                                        t * cfg.far_template_match_points));
}

double RequiredTemplateScore(double distance_m, const ValidatorConfig& cfg) {
  if (distance_m >= static_cast<double>(cfg.far_template_distance_m))
    return cfg.far_min_template_score;
  double t = Clamp01(static_cast<float>(distance_m / std::max(1e-3, static_cast<double>(cfg.far_template_distance_m))));
  return (1.0 - t) * cfg.min_template_score + t * cfg.far_min_template_score;
}

// ---------------------------------------------------------------------------
// Validator Node
// ---------------------------------------------------------------------------

class TraditionalValidator {
 public:
  TraditionalValidator(ros::NodeHandle nh, ros::NodeHandle pnh) : nh_(nh), pnh_(pnh) {
    LoadConfig();
    LoadTemplateModel(cfg_, &model_);

    validation_sub_ = nh_.subscribe(cfg_.validation_request_topic, 4,
                                    &TraditionalValidator::RequestCallback, this);
    validation_pub_ = nh_.advertise<livox_reflective_marker::CandidateValidationResult>(
        cfg_.validation_result_topic, 4, true);
    sub_ = nh_.subscribe(cfg_.input_topic, 4,
                         &TraditionalValidator::PointCloudCallback, this);
    ROS_INFO("[trad_validator] ready | accum=%.1fs timeout=%.1fs confirm=%.2f/%.2f",
             cfg_.validation_accumulation_sec, cfg_.validation_timeout_sec,
             cfg_.validation_confirm_score_threshold,
             cfg_.validation_confirm_margin_threshold);
  }

 private:
  void LoadConfig() {
#define LOAD(ns, name, var) ns.param(name, var, var)
    LOAD(pnh_, "input_topic", cfg_.input_topic);
    LOAD(pnh_, "frame_id", cfg_.frame_id);
    LOAD(pnh_, "task1/validation_request_topic", cfg_.validation_request_topic);
    LOAD(pnh_, "task1/validation_result_topic", cfg_.validation_result_topic);

    int rf = static_cast<int>(cfg_.reflectivity_threshold);
    pnh_.param("task1/reflectivity_threshold", rf, rf);
    cfg_.reflectivity_threshold = static_cast<uint8_t>(std::max(0, std::min(255, rf)));

    LOAD(pnh_, "task1/min_distance_m", cfg_.min_distance_m);
    LOAD(pnh_, "task1/max_distance_m", cfg_.max_distance_m);
    LOAD(pnh_, "task1/validation_accumulation_sec", cfg_.validation_accumulation_sec);
    LOAD(pnh_, "task1/validation_timeout_sec", cfg_.validation_timeout_sec);
    LOAD(pnh_, "task1/validation_confirm_score_threshold", cfg_.validation_confirm_score_threshold);
    LOAD(pnh_, "task1/validation_confirm_margin_threshold", cfg_.validation_confirm_margin_threshold);
    LOAD(pnh_, "task1/template_path", cfg_.template_path);
    auto load_size_param = [&](const std::string& name, size_t* value) {
      int tmp = static_cast<int>(*value);
      pnh_.param<int>(name, tmp, tmp);
      *value = static_cast<size_t>(std::max(0, tmp));
    };
    load_size_param("task1/template_grid_cols", &cfg_.template_grid_cols);
    load_size_param("task1/template_grid_rows", &cfg_.template_grid_rows);
    load_size_param("task1/template_mid_grid_cols", &cfg_.template_mid_grid_cols);
    load_size_param("task1/template_mid_grid_rows", &cfg_.template_mid_grid_rows);
    load_size_param("task1/template_coarse_grid_cols", &cfg_.template_coarse_grid_cols);
    load_size_param("task1/template_coarse_grid_rows", &cfg_.template_coarse_grid_rows);
    load_size_param("task1/min_template_points", &cfg_.min_template_points);
    load_size_param("task1/target_template_match_points", &cfg_.target_template_match_points);
    load_size_param("task1/far_template_match_points", &cfg_.far_template_match_points);
    LOAD(pnh_, "task1/far_template_distance_m", cfg_.far_template_distance_m);
    LOAD(pnh_, "task1/min_template_score", cfg_.min_template_score);
    LOAD(pnh_, "task1/far_min_template_score", cfg_.far_min_template_score);
    LOAD(pnh_, "task1/good_template_score", cfg_.good_template_score);
    LOAD(pnh_, "task1/template_plane_tolerance_m", cfg_.template_plane_tolerance_m);
    LOAD(pnh_, "task1/confirm_score_threshold", cfg_.confirm_score_threshold);
    LOAD(pnh_, "task1/confirm_margin_threshold", cfg_.confirm_margin_threshold);
#undef LOAD

    cfg_.validation_accumulation_sec = std::max(0.5, cfg_.validation_accumulation_sec);
    cfg_.validation_timeout_sec = std::max(cfg_.validation_accumulation_sec + 0.5,
                                           cfg_.validation_timeout_sec);
  }

  // ---- Main point cloud callback (accumulates when a validation job is active) ----

  void PointCloudCallback(const livox_ros_driver2::CustomMsg::ConstPtr& msg) {
    if (!job_active_) return;

    // Timeout check
    if ((msg->header.stamp - job_start_).toSec() > cfg_.validation_timeout_sec) {
      PublishSimple(false, 0.0, 0.0, "timeout");
      ResetJob();
      return;
    }

    // Filter points inside ROI
    float roi_sq = roi_radius_ * roi_radius_;
    for (const auto& raw : msg->points) {
      PointXYZI p;
      p.x = raw.x; p.y = raw.y; p.z = raw.z;
      p.reflectivity = raw.reflectivity;
      if (!IsReflectiveCandidate(p, cfg_)) continue;
      Eigen::Vector3f pt = ToEigen(p);
      if ((pt - roi_center_).squaredNorm() <= roi_sq)
        accumulated_.push_back({p, msg->header.stamp});
    }

    // Check if enough time has passed
    if ((msg->header.stamp - job_start_).toSec() < cfg_.validation_accumulation_sec)
      return;

    // Run validation
    RunValidation(msg->header.stamp);
  }

  // ---- Validation request callback ----

  void RequestCallback(
      const livox_reflective_marker::CandidateValidationRequest::ConstPtr& req) {
    ROS_INFO("[trad_validator] received validation request id=%u mode=%s pos=[%.2f,%.2f,%.2f]",
             req->proposal_id,
             req->mode == 0 ? "INIT" : "CORRECTION",
             static_cast<double>(req->center.x),
             static_cast<double>(req->center.y),
             static_cast<double>(req->center.z));

    // Reset previous job
    ResetJob();

    job_active_ = true;
    proposal_id_ = req->proposal_id;
    mode_ = req->mode;
    roi_center_ = Eigen::Vector3f(req->center.x, req->center.y, req->center.z);
    roi_radius_ = req->roi_radius > 0.0f ? req->roi_radius : 0.45f;
    job_start_ = req->header.stamp;

    const auto seed_points =
        ExtractSeedCloudPoints(req->seed_cloud, roi_center_, roi_radius_, cfg_);
    for (const auto& p : seed_points) {
      accumulated_.push_back({p, job_start_});
    }
    if (!seed_points.empty()) {
      ROS_INFO("[trad_validator] seeded validation id=%u with %zu BPU points",
               proposal_id_, seed_points.size());
    }
  }

  // ---- Core validation logic ----

  void RunValidation(const ros::Time& stamp) {
    // Extract high-reflectivity points from accumulated
    std::vector<PointXYZI> high_pts;
    high_pts.reserve(accumulated_.size());
    for (const auto& tp : accumulated_)
      high_pts.push_back(tp.point);

    if (high_pts.size() < cfg_.min_template_points) {
      PublishSimple(false, 0.0, 0.0,
                    "insufficient points: " + std::to_string(high_pts.size()));
      ResetJob();
      return;
    }

    // PCA pose estimation
    BoardPose board;
    if (!EstimatePoseFromHighPoints(high_pts, &board, cfg_.min_template_points)) {
      PublishSimple(false, 0.0, 0.0, "PCA pose estimation failed");
      ResetJob();
      return;
    }

    // Template matching
    double distance = static_cast<double>(roi_center_.norm());
    size_t req_pts = RequiredTemplatePoints(distance, cfg_);
    double req_score = RequiredTemplateScore(distance, cfg_);

    double raw = TemplateScore(high_pts, board, model_, cfg_);
    double calibrated = CalibratedTemplateScore(raw, req_score, cfg_);

    double margin = calibrated - req_score;
    bool accepted = calibrated >= cfg_.validation_confirm_score_threshold &&
                    margin >= cfg_.validation_confirm_margin_threshold &&
                    high_pts.size() >= req_pts;

    std::string reason;
    if (accepted) {
      reason = "accepted";
    } else if (high_pts.size() < req_pts) {
      reason = "rejected (pts=" + std::to_string(high_pts.size()) +
               " < " + std::to_string(req_pts) + ")";
    } else if (margin < cfg_.validation_confirm_margin_threshold) {
      reason = "rejected (margin=" + std::to_string(margin) +
               " < " + std::to_string(cfg_.validation_confirm_margin_threshold) + ")";
    } else {
      reason = "rejected (score=" + std::to_string(calibrated) +
               " < " + std::to_string(cfg_.validation_confirm_score_threshold) + ")";
    }

    ROS_INFO("[trad_validator] validation result: %s | raw=%.3f cal=%.3f margin=%.3f pts=%zu",
             accepted ? "ACCEPTED" : "REJECTED",
             raw, calibrated, margin, high_pts.size());

    PublishResult(accepted, calibrated, margin, reason, board, high_pts, stamp);
    ResetJob();
  }

  void PublishResult(bool accepted, double score, double margin,
                     const std::string& reason,
                     const BoardPose& board = BoardPose{},
                     const std::vector<PointXYZI>& pts = {},
                     const ros::Time& stamp = ros::Time()) {
    livox_reflective_marker::CandidateValidationResult result;
    result.header.stamp = stamp.isZero() ? ros::Time::now() : stamp;
    result.header.frame_id = cfg_.frame_id;
    result.proposal_id = proposal_id_;
    result.mode = mode_;
    result.accepted = accepted;
    result.score = static_cast<float>(score);
    result.margin = static_cast<float>(margin);
    result.reason = reason;
    result.pose.header = result.header;
    result.pose.pose = board.pose;

    // Build ROI cloud
    sensor_msgs::PointCloud2 cloud;
    cloud.header = result.header;
    cloud.height = 1;
    cloud.is_bigendian = false;
    cloud.is_dense = true;
    sensor_msgs::PointField f;
    f.datatype = sensor_msgs::PointField::FLOAT32; f.count = 1;
    f.name = "x"; f.offset = 0; cloud.fields.push_back(f);
    f.name = "y"; f.offset = 4; cloud.fields.push_back(f);
    f.name = "z"; f.offset = 8; cloud.fields.push_back(f);
    f.name = "intensity"; f.datatype = sensor_msgs::PointField::FLOAT32;
    f.offset = 12; cloud.fields.push_back(f);
    cloud.point_step = 16;
    cloud.width = static_cast<uint32_t>(pts.size());
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step);
    float* buf = reinterpret_cast<float*>(cloud.data.data());
    for (size_t i = 0; i < pts.size(); ++i) {
      buf[i * 4 + 0] = pts[i].x;
      buf[i * 4 + 1] = pts[i].y;
      buf[i * 4 + 2] = pts[i].z;
      buf[i * 4 + 3] = static_cast<float>(pts[i].reflectivity);
    }
    result.roi_cloud = cloud;

    validation_pub_.publish(result);
  }

  // Shortcut for simple string-only results (no pose/cloud)
  void PublishSimple(bool accepted, double score, double margin, const std::string& reason) {
    PublishResult(accepted, score, margin, reason, BoardPose{}, {}, ros::Time::now());
  }

  void ResetJob() {
    job_active_ = false;
    proposal_id_ = 0;
    mode_ = 0;
    accumulated_.clear();
    roi_center_ = Eigen::Vector3f::Zero();
    roi_radius_ = 0.45f;
  }

  // ---- Members ----
  ros::NodeHandle nh_, pnh_;
  ValidatorConfig cfg_;
  TemplateModel model_;

  ros::Subscriber sub_;
  ros::Subscriber validation_sub_;
  ros::Publisher validation_pub_;

  bool job_active_ = false;
  uint32_t proposal_id_ = 0;
  uint8_t mode_ = 0;
  Eigen::Vector3f roi_center_ = Eigen::Vector3f::Zero();
  float roi_radius_ = 0.45f;
  ros::Time job_start_;
  std::deque<TimedPoint> accumulated_;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "traditional_validator");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  TraditionalValidator node(nh, pnh);
  ros::spin();
  return 0;
}
