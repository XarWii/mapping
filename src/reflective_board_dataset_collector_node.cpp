#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>

#include <livox_ros_driver2/CustomMsg.h>
#include <livox_ros_driver2/CustomPoint.h>

namespace {

struct PointXYZI {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint8_t reflectivity = 0;
  float distance = 0.0f;
};

struct FramePoints {
  ros::Time stamp;
  uint32_t seq = 0;
  std::vector<PointXYZI> all_points;
  std::vector<PointXYZI> high_points;
};

struct Cluster {
  std::vector<size_t> indices;
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  uint8_t max_reflectivity = 0;
};

struct BoardPose {
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Vector3f x_axis = Eigen::Vector3f::UnitX();
  Eigen::Vector3f y_axis = Eigen::Vector3f::UnitY();
  Eigen::Vector3f z_axis = Eigen::Vector3f::UnitZ();
};

struct CandidateScore {
  size_t candidate_index = 0;
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  float radius = 0.0f;
  double score = 0.0;
  double template_score = 0.0;
  size_t high_points = 0;
  bool valid_pose = false;
  std::string reason = "not scored";
};

struct TemplateModel {
  bool loaded = false;
  std::string path;
  std::vector<Eigen::Vector2f> points_uv;
  std::vector<std::vector<float>> descriptor_variants;
  std::vector<std::vector<float>> mid_descriptor_variants;
  std::vector<std::vector<float>> coarse_descriptor_variants;
};

struct Config {
  std::string input_topic = "/livox/lidar";
  std::string frame_id = "livox_frame";
  int input_queue_size = 8;

  std::string dataset_root = "/home/sunrise/catkin_ws/dataset";
  std::string collection_mode = "candidates";
  std::string label = "auto";
  std::string scene_id = "scene";
  std::string run_id;
  std::string label_source = "external_pose";
  std::string target_pose_topic = "/reflective_board_identifier_node/target_pose";
  double target_pose_timeout_sec = 1.0;
  bool auto_label = true;
  bool save_unconfirmed = false;
  bool publish_visualization = true;

  uint8_t reflectivity_threshold = 250;
  float min_distance_m = 0.1f;
  float max_distance_m = 30.0f;

  std::vector<int> frame_windows = {1, 2, 3, 4};
  float cluster_tolerance_m = 0.25f;
  size_t min_cluster_points = 3;
  size_t max_cluster_points = 5000;
  size_t max_candidates_per_window = 8;
  size_t max_saved_candidates_per_window = 0;

  float roi_padding_m = 0.15f;
  float min_roi_radius_m = 0.30f;
  float max_roi_radius_m = 0.60f;
  size_t max_saved_roi_points = 2048;
  size_t max_saved_high_points = 512;
  int dense_accumulation_frames = 50;
  float dense_target_roi_radius_m = 0.45f;
  size_t dense_max_saved_roi_points = 0;
  size_t dense_max_saved_high_points = 0;

  int label_window_frames = 0;
  float target_center_tolerance_m = 0.20f;
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

  int save_every_n_frames = 1;
  bool save_once = true;
  size_t max_samples_per_session = 0;
  bool auto_shutdown_after_max_samples = false;
  bool verbose = false;
};

bool IsFinite(float v) {
  return std::isfinite(v);
}

PointXYZI ToPoint(const livox_ros_driver2::CustomPoint& raw_point) {
  PointXYZI point;
  point.x = raw_point.x;
  point.y = raw_point.y;
  point.z = raw_point.z;
  point.reflectivity = raw_point.reflectivity;
  point.distance = std::sqrt(point.x * point.x + point.y * point.y +
                             point.z * point.z);
  return point;
}

bool IsValidRangePoint(const PointXYZI& point, const Config& config) {
  return IsFinite(point.x) && IsFinite(point.y) && IsFinite(point.z) &&
         IsFinite(point.distance) &&
         point.distance >= config.min_distance_m &&
         point.distance <= config.max_distance_m;
}

bool IsHighReflectivePoint(const PointXYZI& point, const Config& config) {
  return IsValidRangePoint(point, config) &&
         point.reflectivity >= config.reflectivity_threshold;
}

Eigen::Vector3f ToEigen(const PointXYZI& point) {
  return Eigen::Vector3f(point.x, point.y, point.z);
}

float SquaredDistance(const PointXYZI& a, const PointXYZI& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

float SquaredDistance(const PointXYZI& point, const Eigen::Vector3f& center) {
  const float dx = point.x - center.x();
  const float dy = point.y - center.y();
  const float dz = point.z - center.z();
  return dx * dx + dy * dy + dz * dz;
}

bool DirectoryExists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool EnsureDirectoryExists(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  if (DirectoryExists(path)) {
    return true;
  }

  std::string current;
  size_t start = 0;
  if (path.front() == '/') {
    current = "/";
    start = 1;
  }

  while (start <= path.size()) {
    const size_t end = path.find('/', start);
    const std::string part =
        path.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (!part.empty()) {
      if (!current.empty() && current.back() != '/') {
        current += "/";
      }
      current += part;
      if (!DirectoryExists(current) && ::mkdir(current.c_str(), 0755) != 0 &&
          errno != EEXIST) {
        return false;
      }
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }

  return DirectoryExists(path);
}

std::string JoinPath(const std::string& dir, const std::string& file) {
  if (dir.empty() || dir.back() == '/') {
    return dir + file;
  }
  return dir + "/" + file;
}

std::string YamlQuote(const std::string& value) {
  std::string quoted = "\"";
  for (char ch : value) {
    if (ch == '\\' || ch == '"') {
      quoted += '\\';
    }
    quoted += ch;
  }
  quoted += "\"";
  return quoted;
}

std::string TimestampId(const ros::Time& stamp) {
  std::ostringstream oss;
  oss << stamp.sec << "_" << std::setw(9) << std::setfill('0') << stamp.nsec;
  return oss.str();
}

std::string SanitizeToken(std::string value) {
  if (value.empty()) {
    return "unset";
  }
  for (char& ch : value) {
    const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                    (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
    if (!ok) {
      ch = '_';
    }
  }
  return value;
}

void WriteVector(std::ofstream& out, const Eigen::Vector3f& v) {
  out << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]";
}

void WritePointList(std::ofstream& out,
                    const std::vector<PointXYZI>& points,
                    size_t max_points) {
  const size_t n = points.size();
  const size_t saved = max_points == 0 ? n : std::min(n, max_points);
  if (saved == 0) {
    return;
  }

  const double stride =
      saved == n ? 1.0 : static_cast<double>(n) / static_cast<double>(saved);
  for (size_t i = 0; i < saved; ++i) {
    const size_t idx =
        saved == n ? i : std::min(n - 1, static_cast<size_t>(std::floor(i * stride)));
    const PointXYZI& point = points[idx];
    out << "  - [" << point.x << ", " << point.y << ", " << point.z << "]\n";
  }
}

float PackedRgb(uint8_t r, uint8_t g, uint8_t b) {
  uint32_t rgb = (static_cast<uint32_t>(r) << 16) |
                 (static_cast<uint32_t>(g) << 8) |
                 static_cast<uint32_t>(b);
  float packed = 0.0f;
  std::memcpy(&packed, &rgb, sizeof(packed));
  return packed;
}

sensor_msgs::PointCloud2 BuildPointCloud2Msg(const std::vector<float>& xyz_rgb,
                                             const ros::Time& stamp,
                                             const std::string& frame_id) {
  const size_t n = xyz_rgb.size() / 4;

  sensor_msgs::PointCloud2 msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.height = 1;
  msg.width = static_cast<uint32_t>(n);
  msg.is_bigendian = false;
  msg.is_dense = true;
  msg.point_step = 16;
  msg.row_step = msg.point_step * msg.width;

  sensor_msgs::PointField field;
  field.name = "x";
  field.offset = 0;
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  msg.fields.push_back(field);

  field.name = "y";
  field.offset = 4;
  msg.fields.push_back(field);

  field.name = "z";
  field.offset = 8;
  msg.fields.push_back(field);

  field.name = "rgb";
  field.offset = 12;
  msg.fields.push_back(field);

  msg.data.resize(n * msg.point_step);
  for (size_t i = 0; i < n; ++i) {
    std::memcpy(msg.data.data() + i * msg.point_step,
                xyz_rgb.data() + i * 4,
                16);
  }
  return msg;
}

float Clamp01(float v) {
  return std::max(0.0f, std::min(1.0f, v));
}

std::vector<Eigen::Vector2f> NormalizeUvPoints(
    const std::vector<Eigen::Vector2f>& points) {
  std::vector<Eigen::Vector2f> normalized;
  if (points.size() < 3) {
    return normalized;
  }

  Eigen::Vector2f mean = Eigen::Vector2f::Zero();
  for (const auto& point : points) {
    mean += point;
  }
  mean /= static_cast<float>(points.size());

  Eigen::Matrix2f covariance = Eigen::Matrix2f::Zero();
  for (const auto& point : points) {
    const Eigen::Vector2f d = point - mean;
    covariance += d * d.transpose();
  }
  covariance /= static_cast<float>(points.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return normalized;
  }

  Eigen::Matrix2f axes;
  axes.col(0) = solver.eigenvectors().col(1).normalized();
  axes.col(1) = solver.eigenvectors().col(0).normalized();

  normalized.reserve(points.size());
  float min_x = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  for (const auto& point : points) {
    const Eigen::Vector2f q = axes.transpose() * (point - mean);
    normalized.push_back(q);
    min_x = std::min(min_x, q.x());
    max_x = std::max(max_x, q.x());
    min_y = std::min(min_y, q.y());
    max_y = std::max(max_y, q.y());
  }

  const float span_x = std::max(1e-4f, max_x - min_x);
  const float span_y = std::max(1e-4f, max_y - min_y);
  const float scale = std::max(span_x, span_y);
  for (auto& point : normalized) {
    point /= scale;
  }
  return normalized;
}

std::vector<float> BuildUvDescriptor(const std::vector<Eigen::Vector2f>& points,
                                     size_t cols,
                                     size_t rows,
                                     bool flip_x,
                                     bool flip_y) {
  std::vector<float> grid(cols * rows, 0.0f);
  if (points.empty() || cols == 0 || rows == 0) {
    return grid;
  }

  for (const auto& point : points) {
    float x = flip_x ? -point.x() : point.x();
    float y = flip_y ? -point.y() : point.y();
    x += 0.5f;
    y += 0.5f;
    if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) {
      continue;
    }
    const size_t col =
        std::min(cols - 1, static_cast<size_t>(std::floor(x * cols)));
    const size_t row =
        std::min(rows - 1, static_cast<size_t>(std::floor(y * rows)));
    grid[row * cols + col] += 1.0f;
  }

  float norm = 0.0f;
  for (float& value : grid) {
    value = std::sqrt(value);
    norm += value * value;
  }
  norm = std::sqrt(norm);
  if (norm > 1e-6f) {
    for (float& value : grid) {
      value /= norm;
    }
  }
  return grid;
}

float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) {
    return 0.0f;
  }
  float dot = 0.0f;
  float an = 0.0f;
  float bn = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    an += a[i] * a[i];
    bn += b[i] * b[i];
  }
  if (an < 1e-6f || bn < 1e-6f) {
    return 0.0f;
  }
  return Clamp01(dot / std::sqrt(an * bn));
}

std::vector<std::vector<float>> BuildDescriptorVariants(
    const std::vector<Eigen::Vector2f>& points,
    size_t cols,
    size_t rows) {
  const std::vector<Eigen::Vector2f> normalized = NormalizeUvPoints(points);
  std::vector<std::vector<float>> variants;
  variants.push_back(BuildUvDescriptor(normalized, cols, rows, false, false));
  variants.push_back(BuildUvDescriptor(normalized, cols, rows, true, false));
  variants.push_back(BuildUvDescriptor(normalized, cols, rows, false, true));
  variants.push_back(BuildUvDescriptor(normalized, cols, rows, true, true));
  return variants;
}

bool LoadTemplateModel(const Config& config, TemplateModel* model) {
  if (model == nullptr) {
    return false;
  }
  *model = TemplateModel();
  model->path = config.template_path;
  if (config.template_path.empty()) {
    ROS_WARN("Dataset collector auto_label is enabled, but template_path is empty.");
    return false;
  }

  std::ifstream in(config.template_path);
  if (!in.is_open()) {
    ROS_WARN_STREAM("Dataset collector failed to open template: "
                    << config.template_path);
    return false;
  }

  bool in_points = false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("points_local_uv:") != std::string::npos) {
      in_points = true;
      continue;
    }
    if (!in_points) {
      continue;
    }
    const size_t lb = line.find('[');
    const size_t rb = line.find(']');
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) {
      continue;
    }
    std::string values = line.substr(lb + 1, rb - lb - 1);
    std::replace(values.begin(), values.end(), ',', ' ');
    std::stringstream ss(values);
    float u = 0.0f;
    float v = 0.0f;
    int reflectivity = 0;
    if (ss >> u >> v >> reflectivity) {
      model->points_uv.emplace_back(u, v);
    }
  }

  if (model->points_uv.size() < config.min_template_points) {
    ROS_WARN_STREAM("Dataset collector template has too few points: "
                    << model->points_uv.size() << "/"
                    << config.min_template_points);
    return false;
  }

  model->descriptor_variants =
      BuildDescriptorVariants(model->points_uv,
                              config.template_grid_cols,
                              config.template_grid_rows);
  model->mid_descriptor_variants =
      BuildDescriptorVariants(model->points_uv,
                              config.template_mid_grid_cols,
                              config.template_mid_grid_rows);
  model->coarse_descriptor_variants =
      BuildDescriptorVariants(model->points_uv,
                              config.template_coarse_grid_cols,
                              config.template_coarse_grid_rows);
  model->loaded = true;
  ROS_INFO_STREAM("Dataset collector loaded template: " << config.template_path
                  << " points=" << model->points_uv.size());
  return true;
}

Cluster BuildCluster(const std::vector<PointXYZI>& points,
                     const std::vector<size_t>& indices) {
  Cluster cluster;
  cluster.indices = indices;
  if (indices.empty()) {
    return cluster;
  }

  for (size_t idx : indices) {
    cluster.center += ToEigen(points[idx]);
    cluster.max_reflectivity = std::max(cluster.max_reflectivity,
                                        points[idx].reflectivity);
  }
  cluster.center /= static_cast<float>(indices.size());
  return cluster;
}

std::vector<Cluster> EuclideanCluster(const std::vector<PointXYZI>& points,
                                      float tolerance_m,
                                      size_t min_cluster_points,
                                      size_t max_cluster_points) {
  std::vector<Cluster> clusters;
  if (points.empty()) {
    return clusters;
  }

  const float tolerance_sq = tolerance_m * tolerance_m;
  std::vector<bool> visited(points.size(), false);
  std::vector<size_t> queue;
  queue.reserve(points.size());

  for (size_t i = 0; i < points.size(); ++i) {
    if (visited[i]) {
      continue;
    }

    visited[i] = true;
    queue.clear();
    queue.push_back(i);

    std::vector<size_t> indices;
    for (size_t head = 0; head < queue.size(); ++head) {
      const size_t current = queue[head];
      indices.push_back(current);
      for (size_t j = 0; j < points.size(); ++j) {
        if (!visited[j] &&
            SquaredDistance(points[current], points[j]) <= tolerance_sq) {
          visited[j] = true;
          queue.push_back(j);
        }
      }
    }

    if (indices.size() >= min_cluster_points &&
        (max_cluster_points == 0 || indices.size() <= max_cluster_points)) {
      clusters.push_back(BuildCluster(points, indices));
    }
  }

  std::sort(clusters.begin(), clusters.end(), [](const Cluster& a, const Cluster& b) {
    if (a.indices.size() != b.indices.size()) {
      return a.indices.size() > b.indices.size();
    }
    return a.max_reflectivity > b.max_reflectivity;
  });
  return clusters;
}

std::vector<int> ParseFrameWindows(const ros::NodeHandle& pnh,
                                   const std::string& key,
                                   const std::vector<int>& fallback) {
  XmlRpc::XmlRpcValue values;
  if (!pnh.getParam(key, values)) {
    return fallback;
  }

  std::vector<int> windows;
  if (values.getType() == XmlRpc::XmlRpcValue::TypeArray) {
    for (int i = 0; i < values.size(); ++i) {
      if (values[i].getType() == XmlRpc::XmlRpcValue::TypeInt) {
        windows.push_back(static_cast<int>(values[i]));
      }
    }
  } else if (values.getType() == XmlRpc::XmlRpcValue::TypeString) {
    std::string text = static_cast<std::string>(values);
    std::replace(text.begin(), text.end(), ',', ' ');
    std::stringstream ss(text);
    int window = 0;
    while (ss >> window) {
      windows.push_back(window);
    }
  }

  windows.erase(std::remove_if(windows.begin(), windows.end(),
                               [](int value) { return value <= 0; }),
                windows.end());
  std::sort(windows.begin(), windows.end());
  windows.erase(std::unique(windows.begin(), windows.end()), windows.end());
  return windows.empty() ? fallback : windows;
}

class ReflectiveBoardDatasetCollector {
 public:
  ReflectiveBoardDatasetCollector(ros::NodeHandle nh, ros::NodeHandle pnh)
      : nh_(nh), pnh_(pnh) {
    LoadConfig();
    if (config_.label_source != "external_pose") {
      LoadTemplateModel(config_, &template_model_);
    }
    PrepareSession();

    labeled_cloud_pub_ =
        pnh_.advertise<sensor_msgs::PointCloud2>("labeled_cloud", 1);
    high_labeled_cloud_pub_ =
        pnh_.advertise<sensor_msgs::PointCloud2>("high_labeled_cloud", 1);
    marker_pub_ =
        pnh_.advertise<visualization_msgs::MarkerArray>("markers", 1);

    if (config_.label_source == "external_pose") {
      target_pose_sub_ =
          nh_.subscribe(config_.target_pose_topic, 1,
                        &ReflectiveBoardDatasetCollector::TargetPoseCallback,
                        this);
    }

    sub_ = nh_.subscribe(config_.input_topic,
                         static_cast<uint32_t>(config_.input_queue_size),
                         &ReflectiveBoardDatasetCollector::Callback, this);

    ROS_INFO_STREAM("reflective_board_dataset_collector_node listening on "
                    << config_.input_topic
                    << ", label=" << config_.label
                    << ", scene_id=" << config_.scene_id
                    << ", run_id=" << config_.run_id
                    << ", label_source=" << config_.label_source
                    << ", frame_windows=" << FrameWindowsText()
                    << ", auto_label=" << (config_.auto_label ? "true" : "false")
                    << ", label_window_frames=" << LabelWindow());
  }

 private:
  void LoadConfig() {
    pnh_.param("input_topic", config_.input_topic, config_.input_topic);
    pnh_.param("dataset_collector/input_topic", config_.input_topic,
               config_.input_topic);
    pnh_.param("frame_id", config_.frame_id, config_.frame_id);
    pnh_.param("dataset_collector/frame_id", config_.frame_id,
               config_.frame_id);
    pnh_.param("input_queue_size", config_.input_queue_size,
               config_.input_queue_size);
    pnh_.param("dataset_collector/input_queue_size", config_.input_queue_size,
               config_.input_queue_size);

    pnh_.param("dataset_collector/dataset_root", config_.dataset_root,
               config_.dataset_root);
    pnh_.param("dataset_collector/collection_mode", config_.collection_mode,
               config_.collection_mode);
    pnh_.param("dataset_collector/label", config_.label, config_.label);
    pnh_.param("dataset_collector/scene_id", config_.scene_id, config_.scene_id);
    pnh_.param("dataset_collector/run_id", config_.run_id, config_.run_id);
    pnh_.param("dataset_collector/label_source", config_.label_source,
               config_.label_source);
    pnh_.param("dataset_collector/target_pose_topic",
               config_.target_pose_topic, config_.target_pose_topic);
    pnh_.param("dataset_collector/target_pose_timeout_sec",
               config_.target_pose_timeout_sec,
               config_.target_pose_timeout_sec);
    pnh_.param("dataset_collector/auto_label", config_.auto_label,
               config_.auto_label);
    pnh_.param("dataset_collector/save_unconfirmed", config_.save_unconfirmed,
               config_.save_unconfirmed);
    pnh_.param("dataset_collector/publish_visualization",
               config_.publish_visualization,
               config_.publish_visualization);

    int reflectivity_threshold = config_.reflectivity_threshold;
    pnh_.param("reflectivity_threshold", reflectivity_threshold,
               reflectivity_threshold);
    pnh_.param("dataset_collector/reflectivity_threshold",
               reflectivity_threshold, reflectivity_threshold);
    config_.reflectivity_threshold =
        static_cast<uint8_t>(std::max(0, std::min(255, reflectivity_threshold)));

    pnh_.param("min_distance_m", config_.min_distance_m, config_.min_distance_m);
    pnh_.param("max_distance_m", config_.max_distance_m, config_.max_distance_m);
    pnh_.param("dataset_collector/min_distance_m", config_.min_distance_m,
               config_.min_distance_m);
    pnh_.param("dataset_collector/max_distance_m", config_.max_distance_m,
               config_.max_distance_m);

    config_.frame_windows =
        ParseFrameWindows(pnh_, "dataset_collector/frame_windows",
                          config_.frame_windows);
    pnh_.param("dataset_collector/cluster_tolerance_m",
               config_.cluster_tolerance_m, config_.cluster_tolerance_m);

    int min_cluster_points = static_cast<int>(config_.min_cluster_points);
    int max_cluster_points = static_cast<int>(config_.max_cluster_points);
    int max_candidates_per_window =
        static_cast<int>(config_.max_candidates_per_window);
    int max_saved_candidates_per_window =
        static_cast<int>(config_.max_saved_candidates_per_window);
    pnh_.param("dataset_collector/min_cluster_points", min_cluster_points,
               min_cluster_points);
    pnh_.param("dataset_collector/max_cluster_points", max_cluster_points,
               max_cluster_points);
    pnh_.param("dataset_collector/max_candidates_per_window",
               max_candidates_per_window, max_candidates_per_window);
    pnh_.param("dataset_collector/max_saved_candidates_per_window",
               max_saved_candidates_per_window,
               max_saved_candidates_per_window);
    config_.min_cluster_points =
        static_cast<size_t>(std::max(1, min_cluster_points));
    config_.max_cluster_points =
        static_cast<size_t>(std::max(0, max_cluster_points));
    config_.max_candidates_per_window =
        static_cast<size_t>(std::max(1, max_candidates_per_window));
    config_.max_saved_candidates_per_window =
        static_cast<size_t>(std::max(0, max_saved_candidates_per_window));

    pnh_.param("dataset_collector/roi_padding_m", config_.roi_padding_m,
               config_.roi_padding_m);
    pnh_.param("dataset_collector/min_roi_radius_m", config_.min_roi_radius_m,
               config_.min_roi_radius_m);
    pnh_.param("dataset_collector/max_roi_radius_m", config_.max_roi_radius_m,
               config_.max_roi_radius_m);

    int max_saved_roi_points = static_cast<int>(config_.max_saved_roi_points);
    int max_saved_high_points = static_cast<int>(config_.max_saved_high_points);
    pnh_.param("dataset_collector/max_saved_roi_points", max_saved_roi_points,
               max_saved_roi_points);
    pnh_.param("dataset_collector/max_saved_high_points", max_saved_high_points,
               max_saved_high_points);
    config_.max_saved_roi_points =
        static_cast<size_t>(std::max(0, max_saved_roi_points));
    config_.max_saved_high_points =
        static_cast<size_t>(std::max(0, max_saved_high_points));
    pnh_.param("dataset_collector/dense_accumulation_frames",
               config_.dense_accumulation_frames,
               config_.dense_accumulation_frames);
    pnh_.param("dataset_collector/dense_target_roi_radius_m",
               config_.dense_target_roi_radius_m,
               config_.dense_target_roi_radius_m);
    int dense_max_saved_roi_points =
        static_cast<int>(config_.dense_max_saved_roi_points);
    int dense_max_saved_high_points =
        static_cast<int>(config_.dense_max_saved_high_points);
    pnh_.param("dataset_collector/dense_max_saved_roi_points",
               dense_max_saved_roi_points, dense_max_saved_roi_points);
    pnh_.param("dataset_collector/dense_max_saved_high_points",
               dense_max_saved_high_points, dense_max_saved_high_points);
    config_.dense_accumulation_frames =
        std::max(1, config_.dense_accumulation_frames);
    config_.dense_target_roi_radius_m =
        std::max(0.05f, config_.dense_target_roi_radius_m);
    config_.dense_max_saved_roi_points =
        static_cast<size_t>(std::max(0, dense_max_saved_roi_points));
    config_.dense_max_saved_high_points =
        static_cast<size_t>(std::max(0, dense_max_saved_high_points));

    pnh_.param("dataset_collector/label_window_frames",
               config_.label_window_frames, config_.label_window_frames);
    pnh_.param("dataset_collector/target_center_tolerance_m",
               config_.target_center_tolerance_m,
               config_.target_center_tolerance_m);

    pnh_.param("task1/template_path", config_.template_path,
               config_.template_path);
    pnh_.param("dataset_collector/template_path", config_.template_path,
               config_.template_path);
    int template_grid_cols = static_cast<int>(config_.template_grid_cols);
    int template_grid_rows = static_cast<int>(config_.template_grid_rows);
    int template_mid_grid_cols = static_cast<int>(config_.template_mid_grid_cols);
    int template_mid_grid_rows = static_cast<int>(config_.template_mid_grid_rows);
    int template_coarse_grid_cols =
        static_cast<int>(config_.template_coarse_grid_cols);
    int template_coarse_grid_rows =
        static_cast<int>(config_.template_coarse_grid_rows);
    int min_template_points = static_cast<int>(config_.min_template_points);
    int target_template_match_points =
        static_cast<int>(config_.target_template_match_points);
    int far_template_match_points =
        static_cast<int>(config_.far_template_match_points);
    pnh_.param("task1/template_grid_cols", template_grid_cols,
               template_grid_cols);
    pnh_.param("task1/template_grid_rows", template_grid_rows,
               template_grid_rows);
    pnh_.param("task1/template_mid_grid_cols", template_mid_grid_cols,
               template_mid_grid_cols);
    pnh_.param("task1/template_mid_grid_rows", template_mid_grid_rows,
               template_mid_grid_rows);
    pnh_.param("task1/template_coarse_grid_cols", template_coarse_grid_cols,
               template_coarse_grid_cols);
    pnh_.param("task1/template_coarse_grid_rows", template_coarse_grid_rows,
               template_coarse_grid_rows);
    pnh_.param("task1/min_template_points", min_template_points,
               min_template_points);
    pnh_.param("task1/target_template_match_points",
               target_template_match_points,
               target_template_match_points);
    pnh_.param("task1/far_template_match_points",
               far_template_match_points,
               far_template_match_points);
    pnh_.param("task1/far_template_distance_m",
               config_.far_template_distance_m,
               config_.far_template_distance_m);
    pnh_.param("task1/min_template_score", config_.min_template_score,
               config_.min_template_score);
    pnh_.param("task1/far_min_template_score", config_.far_min_template_score,
               config_.far_min_template_score);
    pnh_.param("task1/good_template_score", config_.good_template_score,
               config_.good_template_score);
    pnh_.param("task1/template_plane_tolerance_m",
               config_.template_plane_tolerance_m,
               config_.template_plane_tolerance_m);
    pnh_.param("task1/confirm_score_threshold", config_.confirm_score_threshold,
               config_.confirm_score_threshold);
    pnh_.param("task1/confirm_margin_threshold", config_.confirm_margin_threshold,
               config_.confirm_margin_threshold);

    pnh_.param("dataset_collector/template_grid_cols", template_grid_cols,
               template_grid_cols);
    pnh_.param("dataset_collector/template_grid_rows", template_grid_rows,
               template_grid_rows);
    pnh_.param("dataset_collector/template_mid_grid_cols", template_mid_grid_cols,
               template_mid_grid_cols);
    pnh_.param("dataset_collector/template_mid_grid_rows", template_mid_grid_rows,
               template_mid_grid_rows);
    pnh_.param("dataset_collector/template_coarse_grid_cols",
               template_coarse_grid_cols, template_coarse_grid_cols);
    pnh_.param("dataset_collector/template_coarse_grid_rows",
               template_coarse_grid_rows, template_coarse_grid_rows);
    pnh_.param("dataset_collector/min_template_points", min_template_points,
               min_template_points);
    pnh_.param("dataset_collector/target_template_match_points",
               target_template_match_points,
               target_template_match_points);
    pnh_.param("dataset_collector/far_template_match_points",
               far_template_match_points,
               far_template_match_points);
    pnh_.param("dataset_collector/far_template_distance_m",
               config_.far_template_distance_m,
               config_.far_template_distance_m);
    pnh_.param("dataset_collector/min_template_score",
               config_.min_template_score, config_.min_template_score);
    pnh_.param("dataset_collector/far_min_template_score",
               config_.far_min_template_score,
               config_.far_min_template_score);
    pnh_.param("dataset_collector/good_template_score",
               config_.good_template_score, config_.good_template_score);
    pnh_.param("dataset_collector/template_plane_tolerance_m",
               config_.template_plane_tolerance_m,
               config_.template_plane_tolerance_m);
    pnh_.param("dataset_collector/confirm_score_threshold",
               config_.confirm_score_threshold,
               config_.confirm_score_threshold);
    pnh_.param("dataset_collector/confirm_margin_threshold",
               config_.confirm_margin_threshold,
               config_.confirm_margin_threshold);

    config_.template_grid_cols =
        static_cast<size_t>(std::max(8, template_grid_cols));
    config_.template_grid_rows =
        static_cast<size_t>(std::max(8, template_grid_rows));
    config_.template_mid_grid_cols =
        static_cast<size_t>(std::max(8, template_mid_grid_cols));
    config_.template_mid_grid_rows =
        static_cast<size_t>(std::max(8, template_mid_grid_rows));
    config_.template_coarse_grid_cols =
        static_cast<size_t>(std::max(8, template_coarse_grid_cols));
    config_.template_coarse_grid_rows =
        static_cast<size_t>(std::max(8, template_coarse_grid_rows));
    config_.min_template_points =
        static_cast<size_t>(std::max(3, min_template_points));
    config_.target_template_match_points =
        static_cast<size_t>(std::max(static_cast<int>(config_.min_template_points),
                                     target_template_match_points));
    config_.far_template_match_points =
        static_cast<size_t>(std::max(static_cast<int>(config_.min_template_points),
                                     far_template_match_points));
    config_.good_template_score =
        std::max(config_.min_template_score + 1e-3,
                 std::min(1.0, config_.good_template_score));
    config_.template_plane_tolerance_m =
        std::max(0.001f, config_.template_plane_tolerance_m);

    pnh_.param("dataset_collector/save_every_n_frames",
               config_.save_every_n_frames, config_.save_every_n_frames);
    config_.save_every_n_frames = std::max(1, config_.save_every_n_frames);
    pnh_.param("dataset_collector/save_once", config_.save_once,
               config_.save_once);

    int max_samples_per_session =
        static_cast<int>(config_.max_samples_per_session);
    pnh_.param("dataset_collector/max_samples_per_session",
               max_samples_per_session, max_samples_per_session);
    config_.max_samples_per_session =
        static_cast<size_t>(std::max(0, max_samples_per_session));
    pnh_.param("dataset_collector/auto_shutdown_after_max_samples",
               config_.auto_shutdown_after_max_samples,
               config_.auto_shutdown_after_max_samples);

    pnh_.param("verbose", config_.verbose, config_.verbose);
    pnh_.param("dataset_collector/verbose", config_.verbose, config_.verbose);

    config_.label = SanitizeToken(config_.label);
    config_.collection_mode = SanitizeToken(config_.collection_mode);
    config_.scene_id = SanitizeToken(config_.scene_id);
    config_.run_id = SanitizeToken(config_.run_id);
    config_.label_source = SanitizeToken(config_.label_source);
    if (config_.label == "auto") {
      config_.auto_label = true;
    }
    config_.target_pose_timeout_sec =
        std::max(0.05, config_.target_pose_timeout_sec);
    config_.max_distance_m =
        std::max(config_.min_distance_m + 0.01f, config_.max_distance_m);
    config_.max_roi_radius_m =
        std::max(config_.min_roi_radius_m, config_.max_roi_radius_m);
    config_.target_center_tolerance_m =
        std::max(0.0f, config_.target_center_tolerance_m);
  }

  void PrepareSession() {
    const ros::Time now = ros::Time::now();
    if (config_.run_id.empty() || config_.run_id == "unset") {
      config_.run_id = config_.scene_id + "_" + TimestampId(now);
    }
    if (!EnsureDirectoryExists(config_.dataset_root)) {
      ROS_FATAL_STREAM("Failed to create dataset root directory: "
                       << config_.dataset_root);
      ros::shutdown();
      return;
    }
  }

  std::string SessionDirForLabel(const std::string& label) {
    const std::string safe_label = SanitizeToken(label);
    const std::string dir = JoinPath(JoinPath(config_.dataset_root, safe_label),
                                     config_.run_id);
    if (!EnsureDirectoryExists(dir)) {
      ROS_WARN_STREAM("Failed to create dataset session directory: " << dir);
      return {};
    }
    if (metadata_written_labels_.insert(safe_label).second) {
      WriteSessionMetadata(safe_label, dir);
    }
    return dir;
  }

  void WriteSessionMetadata(const std::string& label, const std::string& dir) {
    const std::string path = JoinPath(dir, "metadata.yaml");
    std::ofstream out(path);
    if (!out.is_open()) {
      ROS_WARN_STREAM("Failed to write dataset metadata: " << path);
      return;
    }

    out << std::fixed << std::setprecision(6);
    out << "session_version: 1\n";
    out << "collection_mode: " << YamlQuote(config_.collection_mode) << "\n";
    out << "label: " << YamlQuote(label) << "\n";
    out << "requested_label: " << YamlQuote(config_.label) << "\n";
    out << "auto_label: " << (config_.auto_label ? "true" : "false") << "\n";
    out << "label_source: " << YamlQuote(config_.label_source) << "\n";
    out << "target_pose_topic: " << YamlQuote(config_.target_pose_topic) << "\n";
    out << "scene_id: " << YamlQuote(config_.scene_id) << "\n";
    out << "run_id: " << YamlQuote(config_.run_id) << "\n";
    out << "input_topic: " << YamlQuote(config_.input_topic) << "\n";
    out << "frame_id: " << YamlQuote(config_.frame_id) << "\n";
    out << "template_path: " << YamlQuote(config_.template_path) << "\n";
    out << "reflectivity_threshold: "
        << static_cast<int>(config_.reflectivity_threshold) << "\n";
    out << "min_distance_m: " << config_.min_distance_m << "\n";
    out << "max_distance_m: " << config_.max_distance_m << "\n";
    out << "frame_windows: [";
    for (size_t i = 0; i < config_.frame_windows.size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      out << config_.frame_windows[i];
    }
    out << "]\n";
    out << "cluster_tolerance_m: " << config_.cluster_tolerance_m << "\n";
    out << "min_cluster_points: " << config_.min_cluster_points << "\n";
    out << "roi_padding_m: " << config_.roi_padding_m << "\n";
    out << "min_roi_radius_m: " << config_.min_roi_radius_m << "\n";
    out << "max_roi_radius_m: " << config_.max_roi_radius_m << "\n";
    out << "label_window_frames: " << LabelWindow() << "\n";
    out << "confirm_score_threshold: " << config_.confirm_score_threshold << "\n";
    out << "confirm_margin_threshold: " << config_.confirm_margin_threshold << "\n";
    out << "max_candidates_per_window: "
        << config_.max_candidates_per_window << "\n";
    out << "max_saved_candidates_per_window: "
        << MaxSavedCandidatesPerWindow(config_.max_candidates_per_window) << "\n";
    out << "save_once: " << (config_.save_once ? "true" : "false") << "\n";
    out << "dense_accumulation_frames: "
        << config_.dense_accumulation_frames << "\n";
    out << "dense_target_roi_radius_m: "
        << config_.dense_target_roi_radius_m << "\n";
  }

  std::string FrameWindowsText() const {
    std::ostringstream oss;
    for (size_t i = 0; i < config_.frame_windows.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << config_.frame_windows[i];
    }
    return oss.str();
  }

  int MaxWindow() const {
    int max_window =
        *std::max_element(config_.frame_windows.begin(),
                          config_.frame_windows.end());
    if (config_.collection_mode == "dense_target") {
      max_window = std::max(max_window, config_.dense_accumulation_frames);
    }
    return max_window;
  }

  int LabelWindow() const {
    if (config_.label_window_frames > 0) {
      return std::min(config_.label_window_frames, MaxWindow());
    }
    return MaxWindow();
  }

  size_t MaxSavedCandidatesPerWindow(size_t available_candidates) const {
    const size_t limit = config_.max_saved_candidates_per_window == 0
                             ? config_.max_candidates_per_window
                             : config_.max_saved_candidates_per_window;
    return std::min(available_candidates, limit);
  }

  std::vector<size_t> CandidateSaveOrder(
      const std::vector<Cluster>& clusters,
      const std::vector<PointXYZI>& high_points,
      bool have_confirmed_target,
      const CandidateScore& confirmed_target) const {
    const size_t candidates =
        std::min(clusters.size(), config_.max_candidates_per_window);
    const size_t limit = MaxSavedCandidatesPerWindow(candidates);
    std::vector<size_t> target_indices;
    std::vector<size_t> other_indices;
    target_indices.reserve(candidates);
    other_indices.reserve(candidates);

    for (size_t i = 0; i < candidates; ++i) {
      const std::string label =
          LabelForCandidate(clusters[i], high_points, have_confirmed_target,
                            confirmed_target);
      if (label == "target") {
        target_indices.push_back(i);
      } else {
        other_indices.push_back(i);
      }
    }

    std::vector<size_t> order;
    order.reserve(limit);
    for (size_t index : target_indices) {
      if (order.size() >= limit) {
        return order;
      }
      order.push_back(index);
    }
    for (size_t index : other_indices) {
      if (order.size() >= limit) {
        return order;
      }
      order.push_back(index);
    }
    return order;
  }

  bool EstimatePoseFromHighPoints(const std::vector<PointXYZI>& high_points,
                                  BoardPose* pose) const {
    if (pose == nullptr || high_points.size() < config_.min_template_points) {
      return false;
    }

    Eigen::Vector3f origin = Eigen::Vector3f::Zero();
    for (const auto& point : high_points) {
      origin += ToEigen(point);
    }
    origin /= static_cast<float>(high_points.size());

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& point : high_points) {
      const Eigen::Vector3f d = ToEigen(point) - origin;
      covariance += d * d.transpose();
    }
    covariance /= static_cast<float>(high_points.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) {
      return false;
    }

    Eigen::Vector3f z_axis = solver.eigenvectors().col(0).normalized();
    Eigen::Vector3f x_axis = solver.eigenvectors().col(2).normalized();
    x_axis = x_axis - x_axis.dot(z_axis) * z_axis;
    if (x_axis.norm() < 1e-4f) {
      return false;
    }
    x_axis.normalize();

    Eigen::Vector3f y_axis = z_axis.cross(x_axis);
    if (y_axis.norm() < 1e-4f) {
      return false;
    }
    y_axis.normalize();

    const Eigen::Vector3f to_lidar = -origin;
    if (to_lidar.norm() > 1e-4f && z_axis.dot(to_lidar.normalized()) < 0.0f) {
      z_axis = -z_axis;
      y_axis = -y_axis;
    }

    pose->origin = origin;
    pose->x_axis = x_axis;
    pose->y_axis = y_axis;
    pose->z_axis = z_axis;
    return true;
  }

  size_t RequiredTemplatePoints(double distance_m) const {
    if (distance_m >= config_.far_template_distance_m) {
      return config_.far_template_match_points;
    }
    const double t = Clamp01(static_cast<float>(
        distance_m / std::max(1e-3f, config_.far_template_distance_m)));
    return static_cast<size_t>(
        std::round((1.0 - t) * static_cast<double>(config_.target_template_match_points) +
                   t * static_cast<double>(config_.far_template_match_points)));
  }

  double RequiredTemplateScore(double distance_m) const {
    if (distance_m >= config_.far_template_distance_m) {
      return config_.far_min_template_score;
    }
    const double t = Clamp01(static_cast<float>(
        distance_m / std::max(1e-3f, config_.far_template_distance_m)));
    return (1.0 - t) * config_.min_template_score +
           t * config_.far_min_template_score;
  }

  double TemplateScore(const std::vector<PointXYZI>& high_points,
                       const BoardPose& board) const {
    if (!template_model_.loaded || high_points.size() < config_.min_template_points) {
      return 0.0;
    }

    std::vector<Eigen::Vector2f> uv_points;
    uv_points.reserve(high_points.size());
    const float max_plane_offset =
        std::max(0.08f, 2.0f * config_.template_plane_tolerance_m);
    for (const auto& point : high_points) {
      const Eigen::Vector3f rel = ToEigen(point) - board.origin;
      const float w = rel.dot(board.z_axis);
      if (std::fabs(w) > max_plane_offset) {
        continue;
      }
      uv_points.emplace_back(rel.dot(board.x_axis), rel.dot(board.y_axis));
    }

    if (uv_points.size() < config_.min_template_points) {
      return 0.0;
    }

    size_t cols = config_.template_grid_cols;
    size_t rows = config_.template_grid_rows;
    const std::vector<std::vector<float>>* template_variants =
        &template_model_.descriptor_variants;
    if (uv_points.size() < 0.5 * config_.target_template_match_points) {
      cols = config_.template_coarse_grid_cols;
      rows = config_.template_coarse_grid_rows;
      template_variants = &template_model_.coarse_descriptor_variants;
    } else if (uv_points.size() < config_.target_template_match_points) {
      cols = config_.template_mid_grid_cols;
      rows = config_.template_mid_grid_rows;
      template_variants = &template_model_.mid_descriptor_variants;
    }

    const std::vector<std::vector<float>> candidate_variants =
        BuildDescriptorVariants(uv_points, cols, rows);
    double best = 0.0;
    for (const auto& candidate : candidate_variants) {
      for (const auto& templ : *template_variants) {
        best = std::max(best,
                        static_cast<double>(CosineSimilarity(candidate, templ)));
      }
    }
    return best;
  }

  double CalibratedTemplateScore(double template_score,
                                 double min_template_score) const {
    if (template_score < min_template_score) {
      return 0.0;
    }
    const double normalized_template =
        (template_score - min_template_score) /
        std::max(1e-6, config_.good_template_score - min_template_score);
    const double confidence = Clamp01(static_cast<float>(normalized_template));
    return config_.confirm_score_threshold +
           (1.0 - config_.confirm_score_threshold) * confidence;
  }

  CandidateScore ScoreCluster(size_t candidate_index,
                              const Cluster& cluster,
                              const std::vector<PointXYZI>& high_points) const {
    CandidateScore result;
    result.candidate_index = candidate_index;
    result.center = cluster.center;
    result.radius = CandidateRadius(cluster, high_points);
    result.high_points = cluster.indices.size();

    const double distance_m = static_cast<double>(cluster.center.norm());
    const size_t required_points = RequiredTemplatePoints(distance_m);
    const double required_score = RequiredTemplateScore(distance_m);
    if (!template_model_.loaded) {
      result.reason = "template not loaded";
      return result;
    }
    if (cluster.indices.size() < config_.min_template_points) {
      result.reason = "too few high-reflectivity points";
      return result;
    }
    if (cluster.indices.size() < required_points) {
      std::ostringstream oss;
      oss << "waiting for template points: " << cluster.indices.size()
          << "/" << required_points;
      result.reason = oss.str();
      return result;
    }

    std::vector<PointXYZI> cluster_points;
    cluster_points.reserve(cluster.indices.size());
    for (size_t idx : cluster.indices) {
      cluster_points.push_back(high_points[idx]);
    }

    BoardPose pose;
    if (!EstimatePoseFromHighPoints(cluster_points, &pose)) {
      result.reason = "could not estimate pose";
      return result;
    }

    result.template_score = TemplateScore(cluster_points, pose);
    if (result.template_score < required_score) {
      std::ostringstream oss;
      oss << "template score too low: " << result.template_score
          << "/" << required_score;
      result.reason = oss.str();
      return result;
    }

    result.valid_pose = true;
    result.score = CalibratedTemplateScore(result.template_score, required_score);
    result.score = std::min(1.0, result.score);
    result.reason = "ok(template)";
    return result;
  }

  bool ConfirmTargetFromLabelWindow(CandidateScore* confirmed_score) const {
    if (confirmed_score == nullptr) {
      return false;
    }
    *confirmed_score = CandidateScore();
    if (!config_.auto_label) {
      return false;
    }

    const int window = LabelWindow();
    if (static_cast<int>(frames_.size()) < window) {
      return false;
    }

    std::vector<PointXYZI> all_points;
    std::vector<PointXYZI> high_points;
    std::vector<FramePoints> source_frames;
    AccumulateLastFrames(window, &all_points, &high_points, &source_frames);
    if (high_points.size() < config_.min_cluster_points) {
      return false;
    }

    std::vector<Cluster> clusters =
        EuclideanCluster(high_points, config_.cluster_tolerance_m,
                         config_.min_cluster_points,
                         config_.max_cluster_points);
    if (clusters.empty()) {
      return false;
    }

    std::vector<CandidateScore> scores;
    const size_t candidates =
        std::min(clusters.size(), config_.max_candidates_per_window);
    scores.reserve(candidates);
    for (size_t i = 0; i < candidates; ++i) {
      scores.push_back(ScoreCluster(i, clusters[i], high_points));
    }
    std::sort(scores.begin(), scores.end(),
              [](const CandidateScore& a, const CandidateScore& b) {
                return a.score > b.score;
              });

    const CandidateScore& best = scores.front();
    const double second_score = scores.size() >= 2 ? scores[1].score : 0.0;
    const double margin = best.score - second_score;
    ROS_INFO_STREAM_THROTTLE(
        1.0,
        "Dataset auto-label scores window=" << window
        << " candidates=" << scores.size()
        << " best=" << best.score
        << " second=" << second_score
        << " margin=" << margin
        << " best_template_score=" << best.template_score
        << " best_reason=\"" << best.reason << "\"");

    if (best.valid_pose &&
        best.score >= config_.confirm_score_threshold &&
        margin >= config_.confirm_margin_threshold) {
      *confirmed_score = best;
      return true;
    }
    return false;
  }

  bool ConfirmTargetFromExternalPose(const ros::Time& stamp,
                                     CandidateScore* confirmed_score) const {
    if (confirmed_score == nullptr || !have_target_pose_) {
      return false;
    }
    const double age = std::fabs((stamp - latest_target_pose_stamp_).toSec());
    if (age > config_.target_pose_timeout_sec) {
      return false;
    }

    *confirmed_score = CandidateScore();
    confirmed_score->center = latest_target_center_;
    confirmed_score->radius = 0.0f;
    confirmed_score->score = 1.0;
    confirmed_score->template_score = 1.0;
    confirmed_score->valid_pose = true;
    confirmed_score->reason = "external_target_pose";
    return true;
  }

  bool ConfirmTarget(const ros::Time& stamp,
                     CandidateScore* confirmed_score) const {
    if (config_.label_source == "external_pose") {
      return ConfirmTargetFromExternalPose(stamp, confirmed_score);
    }
    return ConfirmTargetFromLabelWindow(confirmed_score);
  }

  void Callback(const livox_ros_driver2::CustomMsg::ConstPtr& msg) {
    if (config_.max_samples_per_session > 0 &&
        samples_saved_ >= config_.max_samples_per_session) {
      if (config_.auto_shutdown_after_max_samples) {
        ros::shutdown();
      }
      return;
    }

    FramePoints frame;
    frame.stamp = msg->header.stamp;
    if (frame.stamp.isZero()) {
      frame.stamp = ros::Time::now();
    }
    frame.seq = msg->header.seq;
    frame.all_points.reserve(msg->points.size());

    for (const auto& raw_point : msg->points) {
      const PointXYZI point = ToPoint(raw_point);
      if (!IsValidRangePoint(point, config_)) {
        continue;
      }
      frame.all_points.push_back(point);
      if (IsHighReflectivePoint(point, config_)) {
        frame.high_points.push_back(point);
      }
    }

    frames_.push_back(frame);
    while (static_cast<int>(frames_.size()) > MaxWindow()) {
      frames_.erase(frames_.begin());
    }

    ++frames_seen_;
    if (frames_seen_ % static_cast<size_t>(config_.save_every_n_frames) != 0) {
      return;
    }

    SaveWindows();
  }

  void TargetPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    latest_target_pose_stamp_ = msg->header.stamp;
    if (latest_target_pose_stamp_.isZero()) {
      latest_target_pose_stamp_ = ros::Time::now();
    }
    latest_target_center_ =
        Eigen::Vector3f(msg->pose.position.x,
                        msg->pose.position.y,
                        msg->pose.position.z);
    have_target_pose_ = true;
  }

  void SaveWindows() {
    CandidateScore confirmed_target;
    const ros::Time stamp =
        frames_.empty() ? ros::Time::now() : frames_.back().stamp;
    const bool have_confirmed_target =
        ConfirmTarget(stamp, &confirmed_target);
    PublishLabelWindowVisualization(have_confirmed_target, confirmed_target);
    if (config_.save_once && saved_once_) {
      return;
    }
    if (config_.auto_label && !have_confirmed_target &&
        !config_.save_unconfirmed) {
      ROS_INFO_STREAM_THROTTLE(
          1.0,
          "Dataset collector waiting for target confirmation before saving"
          << " (label_source=" << config_.label_source
          << ", target_pose_topic=" << config_.target_pose_topic << ")");
      return;
    }
    if (config_.collection_mode == "dense_target") {
      SaveDenseTargetCloud(have_confirmed_target, confirmed_target);
      return;
    }

    size_t saved_this_frame = 0;
    for (int window : config_.frame_windows) {
      if (static_cast<int>(frames_.size()) < window) {
        continue;
      }

      std::vector<PointXYZI> all_points;
      std::vector<PointXYZI> high_points;
      std::vector<FramePoints> source_frames;
      AccumulateLastFrames(window, &all_points, &high_points, &source_frames);
      if (high_points.size() < config_.min_cluster_points) {
        continue;
      }

      std::vector<Cluster> clusters =
          EuclideanCluster(high_points, config_.cluster_tolerance_m,
                           config_.min_cluster_points,
                           config_.max_cluster_points);
      const std::vector<size_t> save_order =
          CandidateSaveOrder(clusters, high_points, have_confirmed_target,
                             confirmed_target);
      for (size_t i : save_order) {
        if (config_.max_samples_per_session > 0 &&
            samples_saved_ >= config_.max_samples_per_session) {
          break;
        }
        const std::string sample_label =
            LabelForCandidate(clusters[i], high_points, have_confirmed_target,
                              confirmed_target);
        if (sample_label == "unconfirmed" && !config_.save_unconfirmed) {
          continue;
        }
        if (SaveCandidateSample(window, i, sample_label, clusters[i], all_points,
                                high_points, source_frames,
                                have_confirmed_target, confirmed_target)) {
          ++saved_this_frame;
          ++samples_saved_;
        }
      }
    }

    ROS_INFO_STREAM_THROTTLE(
        1.0,
        "Dataset collector frames=" << frames_seen_
        << " buffered=" << frames_.size()
        << " saved_total=" << samples_saved_
        << " saved_last_tick=" << saved_this_frame
        << " run_id=" << config_.run_id);

    if (config_.max_samples_per_session > 0 &&
        samples_saved_ >= config_.max_samples_per_session &&
        config_.auto_shutdown_after_max_samples) {
      ROS_INFO_STREAM("Dataset collector reached max_samples_per_session="
                      << config_.max_samples_per_session << ", shutting down.");
      ros::shutdown();
    }
    if (config_.save_once && saved_this_frame > 0) {
      saved_once_ = true;
      ROS_INFO_STREAM("Dataset collector save_once complete: saved="
                      << saved_this_frame << " sample(s), run_id="
                      << config_.run_id);
      if (config_.auto_shutdown_after_max_samples) {
        ros::shutdown();
      }
    }
  }

  void AccumulateLastFrames(int window,
                            std::vector<PointXYZI>* all_points,
                            std::vector<PointXYZI>* high_points,
                            std::vector<FramePoints>* source_frames) const {
    const size_t start = frames_.size() - static_cast<size_t>(window);
    for (size_t i = start; i < frames_.size(); ++i) {
      const FramePoints& frame = frames_[i];
      all_points->insert(all_points->end(), frame.all_points.begin(),
                         frame.all_points.end());
      high_points->insert(high_points->end(), frame.high_points.begin(),
                          frame.high_points.end());
      source_frames->push_back(frame);
    }
  }

  float CandidateRadius(const Cluster& cluster,
                        const std::vector<PointXYZI>& high_points) const {
    float radius = 0.0f;
    for (size_t idx : cluster.indices) {
      radius = std::max(radius,
                        std::sqrt(SquaredDistance(high_points[idx],
                                                  cluster.center)));
    }
    radius += config_.roi_padding_m;
    return std::max(config_.min_roi_radius_m,
                    std::min(config_.max_roi_radius_m, radius));
  }

  std::vector<PointXYZI> RoiPoints(const std::vector<PointXYZI>& all_points,
                                   const Eigen::Vector3f& center,
                                   float radius) const {
    std::vector<PointXYZI> roi;
    const float radius_sq = radius * radius;
    for (const auto& point : all_points) {
      if (SquaredDistance(point, center) <= radius_sq) {
        roi.push_back(point);
      }
    }
    return roi;
  }

  std::vector<PointXYZI> ClusterHighPoints(
      const Cluster& cluster,
      const std::vector<PointXYZI>& high_points) const {
    std::vector<PointXYZI> points;
    points.reserve(cluster.indices.size());
    for (size_t idx : cluster.indices) {
      points.push_back(high_points[idx]);
    }
    return points;
  }

  bool SaveDenseTargetCloud(bool have_confirmed_target,
                            const CandidateScore& confirmed_target) {
    if (!have_confirmed_target) {
      return false;
    }
    const int window = config_.dense_accumulation_frames;
    if (static_cast<int>(frames_.size()) < window) {
      ROS_INFO_STREAM_THROTTLE(
          1.0,
          "Dense target collector waiting for accumulation frames: "
          << frames_.size() << "/" << window);
      return false;
    }

    std::vector<PointXYZI> all_points;
    std::vector<PointXYZI> high_points;
    std::vector<FramePoints> source_frames;
    AccumulateLastFrames(window, &all_points, &high_points, &source_frames);
    if (source_frames.empty()) {
      return false;
    }

    const std::vector<PointXYZI> roi_points =
        RoiPoints(all_points, confirmed_target.center,
                  config_.dense_target_roi_radius_m);
    const std::vector<PointXYZI> roi_high_points =
        RoiPoints(high_points, confirmed_target.center,
                  config_.dense_target_roi_radius_m);
    if (roi_points.empty()) {
      ROS_WARN_STREAM_THROTTLE(
          1.0,
          "Dense target collector found no ROI points around target center.");
      return false;
    }

    const std::string session_dir = SessionDirForLabel("dense_target");
    if (session_dir.empty()) {
      return false;
    }

    const ros::Time stamp = source_frames.back().stamp;
    std::ostringstream name;
    name << "dense_target_" << stamp.sec << "_"
         << std::setw(9) << std::setfill('0') << stamp.nsec
         << "_f" << std::setw(3) << std::setfill('0') << window
         << ".yaml";
    const std::string path = JoinPath(session_dir, name.str());
    std::ofstream out(path);
    if (!out.is_open()) {
      ROS_WARN_STREAM("Failed to write dense target cloud: " << path);
      return false;
    }

    out << std::fixed << std::setprecision(6);
    out << "sample_version: 1\n";
    out << "collection_mode: \"dense_target\"\n";
    out << "label: \"target\"\n";
    out << "label_source: " << YamlQuote(config_.label_source) << "\n";
    out << "scene_id: " << YamlQuote(config_.scene_id) << "\n";
    out << "run_id: " << YamlQuote(config_.run_id) << "\n";
    out << "input_topic: " << YamlQuote(config_.input_topic) << "\n";
    out << "frame_id: " << YamlQuote(config_.frame_id) << "\n";
    out << "stamp_sec: " << stamp.sec << "\n";
    out << "stamp_nsec: " << stamp.nsec << "\n";
    out << "accumulation_frames: " << window << "\n";
    out << "roi_radius_m: " << config_.dense_target_roi_radius_m << "\n";
    out << "target_center_xyz: ";
    WriteVector(out, confirmed_target.center);
    out << "\n";
    out << "target_pose_topic: " << YamlQuote(config_.target_pose_topic) << "\n";
    out << "reflectivity_threshold: "
        << static_cast<int>(config_.reflectivity_threshold) << "\n";
    out << "roi_point_count: " << roi_points.size() << "\n";
    out << "high_point_count: " << roi_high_points.size() << "\n";
    out << "saved_roi_point_count: "
        << (config_.dense_max_saved_roi_points == 0
                ? roi_points.size()
                : std::min(roi_points.size(), config_.dense_max_saved_roi_points))
        << "\n";
    out << "saved_high_point_count: "
        << (config_.dense_max_saved_high_points == 0
                ? roi_high_points.size()
                : std::min(roi_high_points.size(), config_.dense_max_saved_high_points))
        << "\n";
    out << "source_frames:\n";
    for (const auto& frame : source_frames) {
      out << "  - seq: " << frame.seq << "\n";
      out << "    stamp_sec: " << frame.stamp.sec << "\n";
      out << "    stamp_nsec: " << frame.stamp.nsec << "\n";
      out << "    all_points: " << frame.all_points.size() << "\n";
      out << "    high_points: " << frame.high_points.size() << "\n";
    }
    out << "points_xyz:\n";
    WritePointList(out, roi_points, config_.dense_max_saved_roi_points);
    out << "high_points_xyz:\n";
    WritePointList(out, roi_high_points, config_.dense_max_saved_high_points);
    if (!out) {
      ROS_WARN_STREAM("Failed while writing dense target cloud: " << path);
      return false;
    }

    ++samples_saved_;
    saved_once_ = true;
    ROS_INFO_STREAM("Saved dense target cloud to " << path
                    << " frames=" << window
                    << " roi_points=" << roi_points.size()
                    << " high_points=" << roi_high_points.size());
    if (config_.auto_shutdown_after_max_samples) {
      ros::shutdown();
    }
    return true;
  }

  std::string LabelForCandidate(const Cluster& cluster,
                                const std::vector<PointXYZI>& high_points,
                                bool have_confirmed_target,
                                const CandidateScore& confirmed_target) const {
    if (!config_.auto_label) {
      return config_.label;
    }
    if (!have_confirmed_target) {
      return "unconfirmed";
    }

    const float radius = CandidateRadius(cluster, high_points);
    const float target_radius =
        std::max(radius, confirmed_target.radius) + config_.target_center_tolerance_m;
    const float distance =
        std::sqrt((cluster.center - confirmed_target.center).squaredNorm());
    if (distance <= target_radius) {
      return "target";
    }
    return "interference";
  }

  float ColorForLabel(const std::string& label, bool high_points) const {
    if (label == "target") {
      return high_points ? PackedRgb(80, 255, 100) : PackedRgb(30, 190, 70);
    }
    if (label == "interference") {
      return high_points ? PackedRgb(255, 80, 80) : PackedRgb(200, 45, 45);
    }
    return high_points ? PackedRgb(255, 230, 60) : PackedRgb(210, 165, 40);
  }

  void AppendColoredPoints(const std::vector<PointXYZI>& points,
                           float rgb,
                           std::vector<float>* xyz_rgb) const {
    if (xyz_rgb == nullptr) {
      return;
    }
    xyz_rgb->reserve(xyz_rgb->size() + points.size() * 4);
    for (const auto& point : points) {
      xyz_rgb->push_back(point.x);
      xyz_rgb->push_back(point.y);
      xyz_rgb->push_back(point.z);
      xyz_rgb->push_back(rgb);
    }
  }

  visualization_msgs::Marker MakeTextMarker(const Cluster& cluster,
                                            const std::string& label,
                                            size_t marker_id,
                                            size_t candidate_index,
                                            const ros::Time& stamp) const {
    visualization_msgs::Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = config_.frame_id;
    marker.ns = "dataset_labels";
    marker.id = static_cast<int>(marker_id);
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = cluster.center.x();
    marker.pose.position.y = cluster.center.y();
    marker.pose.position.z = cluster.center.z() + 0.25;
    marker.pose.orientation.w = 1.0;
    marker.scale.z = 0.18;
    marker.color.a = 1.0;
    if (label == "target") {
      marker.color.r = 0.2;
      marker.color.g = 1.0;
      marker.color.b = 0.25;
    } else if (label == "interference") {
      marker.color.r = 1.0;
      marker.color.g = 0.2;
      marker.color.b = 0.2;
    } else {
      marker.color.r = 1.0;
      marker.color.g = 0.85;
      marker.color.b = 0.1;
    }
    std::ostringstream text;
    text << label << " c" << candidate_index;
    marker.text = text.str();
    marker.lifetime = ros::Duration(0.5);
    return marker;
  }

  visualization_msgs::Marker MakeSphereMarker(const Cluster& cluster,
                                              const std::string& label,
                                              size_t candidate_index,
                                              float radius,
                                              const ros::Time& stamp) const {
    visualization_msgs::Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = config_.frame_id;
    marker.ns = "dataset_roi";
    marker.id = static_cast<int>(candidate_index);
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = cluster.center.x();
    marker.pose.position.y = cluster.center.y();
    marker.pose.position.z = cluster.center.z();
    marker.pose.orientation.w = 1.0;
    marker.scale.x = std::max(0.05f, radius * 2.0f);
    marker.scale.y = marker.scale.x;
    marker.scale.z = marker.scale.x;
    marker.color.a = 0.18;
    if (label == "target") {
      marker.color.r = 0.1;
      marker.color.g = 1.0;
      marker.color.b = 0.2;
    } else if (label == "interference") {
      marker.color.r = 1.0;
      marker.color.g = 0.1;
      marker.color.b = 0.1;
    } else {
      marker.color.r = 1.0;
      marker.color.g = 0.75;
      marker.color.b = 0.1;
    }
    marker.lifetime = ros::Duration(0.5);
    return marker;
  }

  void PublishLabelWindowVisualization(bool have_confirmed_target,
                                       const CandidateScore& confirmed_target) {
    if (!config_.publish_visualization ||
        labeled_cloud_pub_.getNumSubscribers() + high_labeled_cloud_pub_.getNumSubscribers() +
                marker_pub_.getNumSubscribers() ==
            0) {
      return;
    }

    const int window = LabelWindow();
    if (static_cast<int>(frames_.size()) < window) {
      return;
    }

    std::vector<PointXYZI> all_points;
    std::vector<PointXYZI> high_points;
    std::vector<FramePoints> source_frames;
    AccumulateLastFrames(window, &all_points, &high_points, &source_frames);
    if (high_points.size() < config_.min_cluster_points || source_frames.empty()) {
      return;
    }

    std::vector<Cluster> clusters =
        EuclideanCluster(high_points, config_.cluster_tolerance_m,
                         config_.min_cluster_points,
                         config_.max_cluster_points);
    const size_t candidates =
        std::min(clusters.size(), config_.max_candidates_per_window);
    const ros::Time stamp = source_frames.back().stamp;

    visualization_msgs::MarkerArray markers;
    visualization_msgs::Marker clear;
    clear.header.stamp = stamp;
    clear.header.frame_id = config_.frame_id;
    clear.action = visualization_msgs::Marker::DELETEALL;
    markers.markers.push_back(clear);

    std::vector<float> roi_xyz_rgb;
    std::vector<float> high_xyz_rgb;
    for (size_t i = 0; i < candidates; ++i) {
      const float radius = CandidateRadius(clusters[i], high_points);
      const std::string label =
          LabelForCandidate(clusters[i], high_points, have_confirmed_target,
                            confirmed_target);
      AppendColoredPoints(RoiPoints(all_points, clusters[i].center, radius),
                          ColorForLabel(label, false), &roi_xyz_rgb);
      AppendColoredPoints(ClusterHighPoints(clusters[i], high_points),
                          ColorForLabel(label, true), &high_xyz_rgb);
      markers.markers.push_back(
          MakeSphereMarker(clusters[i], label, i, radius, stamp));
      markers.markers.push_back(MakeTextMarker(clusters[i], label, i + 1000, i, stamp));
    }

    labeled_cloud_pub_.publish(
        BuildPointCloud2Msg(roi_xyz_rgb, stamp, config_.frame_id));
    high_labeled_cloud_pub_.publish(
        BuildPointCloud2Msg(high_xyz_rgb, stamp, config_.frame_id));
    marker_pub_.publish(markers);
  }

  bool SaveCandidateSample(int window,
                           size_t candidate_index,
                           const std::string& sample_label,
                           const Cluster& cluster,
                           const std::vector<PointXYZI>& all_points,
                           const std::vector<PointXYZI>& high_points,
                           const std::vector<FramePoints>& source_frames,
                           bool have_confirmed_target,
                           const CandidateScore& confirmed_target) {
    const float radius = CandidateRadius(cluster, high_points);
    const std::vector<PointXYZI> roi_points =
        RoiPoints(all_points, cluster.center, radius);
    const std::vector<PointXYZI> cluster_points =
        ClusterHighPoints(cluster, high_points);
    if (roi_points.empty() || cluster_points.empty()) {
      return false;
    }

    const ros::Time stamp = source_frames.back().stamp;
    std::ostringstream name;
    name << "sample_" << std::setw(7) << std::setfill('0') << sample_index_
         << "_w" << std::setw(2) << std::setfill('0') << window
         << "_c" << std::setw(2) << std::setfill('0') << candidate_index
         << "_" << sample_label << ".yaml";
    ++sample_index_;

    const std::string session_dir = SessionDirForLabel(sample_label);
    if (session_dir.empty()) {
      return false;
    }
    const std::string path = JoinPath(session_dir, name.str());
    std::ofstream out(path);
    if (!out.is_open()) {
      ROS_WARN_STREAM("Failed to write dataset sample: " << path);
      return false;
    }

    out << std::fixed << std::setprecision(6);
    out << "sample_version: 1\n";
    out << "label: " << YamlQuote(sample_label) << "\n";
    out << "requested_label: " << YamlQuote(config_.label) << "\n";
    out << "auto_label: " << (config_.auto_label ? "true" : "false") << "\n";
    out << "label_source: " << YamlQuote(config_.label_source) << "\n";
    out << "scene_id: " << YamlQuote(config_.scene_id) << "\n";
    out << "run_id: " << YamlQuote(config_.run_id) << "\n";
    out << "input_topic: " << YamlQuote(config_.input_topic) << "\n";
    out << "frame_id: " << YamlQuote(config_.frame_id) << "\n";
    out << "stamp_sec: " << stamp.sec << "\n";
    out << "stamp_nsec: " << stamp.nsec << "\n";
    out << "window_frames: " << window << "\n";
    out << "label_window_frames: " << LabelWindow() << "\n";
    out << "candidate_index: " << candidate_index << "\n";
    out << "confirmed_target_available: "
        << (have_confirmed_target ? "true" : "false") << "\n";
    if (have_confirmed_target) {
      out << "confirmed_target:\n";
      out << "  score: " << confirmed_target.score << "\n";
      out << "  template_score: " << confirmed_target.template_score << "\n";
      out << "  high_points: " << confirmed_target.high_points << "\n";
      out << "  center_xyz: ";
      WriteVector(out, confirmed_target.center);
      out << "\n";
      out << "  radius_m: " << confirmed_target.radius << "\n";
    }
    out << "reflectivity_threshold: "
        << static_cast<int>(config_.reflectivity_threshold) << "\n";
    out << "template_path: " << YamlQuote(config_.template_path) << "\n";
    out << "cluster_tolerance_m: " << config_.cluster_tolerance_m << "\n";
    out << "candidate_center_xyz: ";
    WriteVector(out, cluster.center);
    out << "\n";
    out << "roi_radius_m: " << radius << "\n";
    out << "roi_point_count: " << roi_points.size() << "\n";
    out << "high_point_count: " << cluster_points.size() << "\n";
    out << "saved_roi_point_count: "
        << (config_.max_saved_roi_points == 0
                ? roi_points.size()
                : std::min(roi_points.size(), config_.max_saved_roi_points))
        << "\n";
    out << "saved_high_point_count: "
        << (config_.max_saved_high_points == 0
                ? cluster_points.size()
                : std::min(cluster_points.size(), config_.max_saved_high_points))
        << "\n";
    out << "source_frames:\n";
    for (const auto& frame : source_frames) {
      out << "  - seq: " << frame.seq << "\n";
      out << "    stamp_sec: " << frame.stamp.sec << "\n";
      out << "    stamp_nsec: " << frame.stamp.nsec << "\n";
      out << "    all_points: " << frame.all_points.size() << "\n";
      out << "    high_points: " << frame.high_points.size() << "\n";
    }
    out << "points_xyz:\n";
    WritePointList(out, roi_points, config_.max_saved_roi_points);
    out << "high_points_xyz:\n";
    WritePointList(out, cluster_points, config_.max_saved_high_points);

    if (config_.verbose) {
      ROS_INFO_STREAM("Saved dataset sample " << path
                      << " roi_points=" << roi_points.size()
                      << " high_points=" << cluster_points.size()
                      << " window=" << window);
    }
    return static_cast<bool>(out);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  Config config_;
  ros::Subscriber sub_;
  ros::Subscriber target_pose_sub_;
  ros::Publisher labeled_cloud_pub_;
  ros::Publisher high_labeled_cloud_pub_;
  ros::Publisher marker_pub_;

  TemplateModel template_model_;
  std::set<std::string> metadata_written_labels_;
  Eigen::Vector3f latest_target_center_ = Eigen::Vector3f::Zero();
  ros::Time latest_target_pose_stamp_;
  bool have_target_pose_ = false;
  std::vector<FramePoints> frames_;
  size_t frames_seen_ = 0;
  size_t sample_index_ = 0;
  size_t samples_saved_ = 0;
  bool saved_once_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "reflective_board_dataset_collector_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ReflectiveBoardDatasetCollector collector(nh, pnh);
  ros::spin();
  return 0;
}
