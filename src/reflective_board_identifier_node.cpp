#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <std_msgs/Float32.h>
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

struct TimedPoint {
  PointXYZI point;
  ros::Time stamp;
};

struct Cluster {
  std::vector<size_t> indices;
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  uint8_t max_reflectivity = 0;
};

struct Candidate {
  int id = 0;
  Eigen::Vector3f seed_center = Eigen::Vector3f::Zero();
  float roi_radius = 0.35f;
  ros::Time created_at;
  std::deque<TimedPoint> all_points;
  std::deque<TimedPoint> high_points;
};

struct BoardPose {
  geometry_msgs::Pose pose;
  Eigen::Vector3f x_axis = Eigen::Vector3f::UnitX();
  Eigen::Vector3f y_axis = Eigen::Vector3f::UnitY();
  Eigen::Vector3f z_axis = Eigen::Vector3f::UnitZ();
};

struct CandidateScore {
  int candidate_id = -1;
  double score = 0.0;
  double template_score = 0.0;
  size_t high_points = 0;
  size_t all_points = 0;
  bool valid_pose = false;
  std::string reason = "not scored";
  BoardPose board;
  std::vector<PointXYZI> template_match_points;
};

struct Config {
  std::string input_topic = "/livox/lidar";
  std::string frame_id = "livox_frame";
  int input_queue_size = 8;

  uint8_t reflectivity_threshold = 250;
  float min_distance_m = 0.1f;
  float max_distance_m = 30.0f;

  double discovery_window_sec = 1.0;
  double min_roi_accumulation_sec = 2.0;
  double score_update_sec = 1.0;
  double max_accumulation_sec = 10.0;

  float discovery_cluster_tolerance_m = 0.25f;
  size_t min_discovery_cluster_points = 3;
  size_t max_discovery_cluster_points = 2000;
  size_t max_candidates = 6;
  size_t max_discovery_high_points = 5000;

  float roi_padding_m = 0.15f;
  float min_roi_radius_m = 0.30f;
  float max_roi_radius_m = 0.60f;
  size_t max_roi_points_per_candidate = 50000;
  size_t max_high_points_per_candidate = 3000;

  double confirm_score_threshold = 0.75;
  double confirm_margin_threshold = 0.30;

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

  bool publish_raw_cloud = true;
  bool publish_candidate_cloud = true;
  bool publish_high_reflective_cloud = true;
  bool publish_template_match_cloud = true;
  bool publish_markers = true;
  size_t max_raw_cloud_points = 200000;
  bool record_detection_result = true;
  std::string result_output_dir = "/tmp/reflective_board_results";
  bool record_candidate_templates_after_confirm = false;
  double candidate_template_accumulation_sec = 5.0;
  std::string candidate_template_output_dir = "/tmp/reflective_board_candidate_templates";
  size_t candidate_template_min_points = 3;
  size_t candidate_template_max_saved_points = 5000;

  bool verbose = false;
};

struct TemplateModel {
  bool loaded = false;
  std::string path;
  std::vector<Eigen::Vector2f> points_uv;
  std::vector<std::vector<float>> descriptor_variants;
  std::vector<std::vector<float>> mid_descriptor_variants;
  std::vector<std::vector<float>> coarse_descriptor_variants;
};

struct ResourceUsageSnapshot {
  float cpu_percent = 0.0f;
  float cpu_time_sec = 0.0f;
  int64_t memory_rss_kb = 0;
  int64_t memory_vm_kb = 0;
};

struct ResourceUsageTracker {
  unsigned long long prev_utime = 0;
  unsigned long long prev_stime = 0;
  unsigned long long prev_total_jiffies = 0;
  bool have_prev = false;
  ResourceUsageSnapshot latest;
};

struct PlaneTemplate {
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Vector3f x_axis = Eigen::Vector3f::UnitX();
  Eigen::Vector3f y_axis = Eigen::Vector3f::UnitY();
  Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
  float plane_rms_m = 0.0f;
  float min_u = 0.0f;
  float max_u = 0.0f;
  float min_v = 0.0f;
  float max_v = 0.0f;
  std::vector<PointXYZI> points_local_uv;
};

enum class State {
  kDiscover,
  kAccumulate,
  kConfirmed,
};

const char* StateName(State state) {
  switch (state) {
    case State::kDiscover:
      return "DISCOVER";
    case State::kAccumulate:
      return "ACCUMULATE";
    case State::kConfirmed:
      return "CONFIRMED";
  }
  return "UNKNOWN";
}

using Rgb8 = std::array<uint8_t, 3>;

constexpr Rgb8 kTargetGreen = {60, 255, 90};
constexpr std::array<Rgb8, 8> kCandidatePalette = {{
    {255, 64, 64},   {70, 150, 255},  {255, 210, 64},  {245, 90, 210},
    {64, 210, 210},  {255, 145, 64},  {180, 120, 255}, {230, 230, 230},
}};

bool IsFinite(float v) {
  return std::isfinite(v);
}

bool IsValidPoint(const PointXYZI& p) {
  return IsFinite(p.x) && IsFinite(p.y) && IsFinite(p.z) && IsFinite(p.distance);
}

PointXYZI ToPoint(const livox_ros_driver2::CustomPoint& raw_point) {
  PointXYZI point;
  point.x = raw_point.x;
  point.y = raw_point.y;
  point.z = raw_point.z;
  point.reflectivity = raw_point.reflectivity;
  point.distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
  return point;
}

Eigen::Vector3f ToEigen(const PointXYZI& p) {
  return Eigen::Vector3f(p.x, p.y, p.z);
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

bool IsReflectiveCandidate(const PointXYZI& point, const Config& config) {
  return point.reflectivity >= config.reflectivity_threshold &&
         point.distance >= config.min_distance_m &&
         point.distance <= config.max_distance_m;
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

std::string ReadFirstLine(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    return {};
  }
  std::string line;
  std::getline(f, line);
  return line;
}

bool ReadSelfProcStat(unsigned long long* utime,
                      unsigned long long* stime) {
  if (utime == nullptr || stime == nullptr) {
    return false;
  }

  const std::string line = ReadFirstLine("/proc/self/stat");
  const size_t close_paren = line.rfind(')');
  if (close_paren == std::string::npos) {
    return false;
  }

  std::istringstream iss(line.substr(close_paren + 2));
  std::string token;
  int field = 0;
  while (iss >> token) {
    if (field == 11) {
      *utime = std::stoull(token);
    } else if (field == 12) {
      *stime = std::stoull(token);
      return true;
    }
    ++field;
  }
  return false;
}

bool ReadSelfProcStatus(int64_t* rss_kb, int64_t* vm_kb) {
  if (rss_kb == nullptr || vm_kb == nullptr) {
    return false;
  }
  std::ifstream f("/proc/self/status");
  if (!f.is_open()) {
    return false;
  }

  *rss_kb = 0;
  *vm_kb = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 6, "VmRSS:") == 0 ||
        line.compare(0, 7, "VmSize:") == 0) {
      const bool is_rss = line.compare(0, 6, "VmRSS:") == 0;
      std::istringstream iss(line.substr(is_rss ? 6 : 7));
      int64_t value = 0;
      if (iss >> value) {
        if (is_rss) {
          *rss_kb = value;
        } else {
          *vm_kb = value;
        }
      }
    }
  }
  return true;
}

unsigned long long ReadSystemJiffies() {
  const std::string line = ReadFirstLine("/proc/stat");
  if (line.empty()) {
    return 0;
  }

  std::istringstream iss(line);
  std::string cpu_label;
  iss >> cpu_label;
  unsigned long long total = 0;
  unsigned long long value = 0;
  while (iss >> value) {
    total += value;
  }
  return total;
}

void UpdateResourceUsage(ResourceUsageTracker* tracker) {
  if (tracker == nullptr) {
    return;
  }

  unsigned long long utime = 0;
  unsigned long long stime = 0;
  if (ReadSelfProcStat(&utime, &stime)) {
    const unsigned long long total_jiffies = ReadSystemJiffies();
    if (tracker->have_prev && total_jiffies > tracker->prev_total_jiffies) {
      const unsigned long long delta_proc =
          (utime - tracker->prev_utime) + (stime - tracker->prev_stime);
      const unsigned long long delta_sys =
          total_jiffies - tracker->prev_total_jiffies;
      const long online_cores =
          std::max<long>(1, sysconf(_SC_NPROCESSORS_ONLN));
      tracker->latest.cpu_percent =
          (static_cast<float>(delta_proc) / static_cast<float>(delta_sys)) *
          100.0f * static_cast<float>(online_cores);
      tracker->latest.cpu_percent =
          std::max(0.0f, std::min(tracker->latest.cpu_percent, 10000.0f));
    }

    const long ticks_per_sec = std::max<long>(1, sysconf(_SC_CLK_TCK));
    tracker->latest.cpu_time_sec =
        static_cast<float>(utime + stime) / static_cast<float>(ticks_per_sec);
    tracker->prev_utime = utime;
    tracker->prev_stime = stime;
    tracker->prev_total_jiffies = total_jiffies;
    tracker->have_prev = true;
  }

  ReadSelfProcStatus(&tracker->latest.memory_rss_kb,
                     &tracker->latest.memory_vm_kb);
}

void WriteVector(std::ofstream& out, const Eigen::Vector3f& v) {
  out << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]";
}

bool EstimatePlaneTemplateFromPoints(const std::vector<PointXYZI>& points,
                                     size_t max_saved_points,
                                     PlaneTemplate* out) {
  if (out == nullptr || points.size() < 3) {
    return false;
  }

  PlaneTemplate result;
  for (const auto& point : points) {
    result.origin += ToEigen(point);
  }
  result.origin /= static_cast<float>(points.size());

  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (const auto& point : points) {
    const Eigen::Vector3f d = ToEigen(point) - result.origin;
    covariance += d * d.transpose();
  }
  covariance /= static_cast<float>(points.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  result.normal = solver.eigenvectors().col(0).normalized();
  result.x_axis = solver.eigenvectors().col(2).normalized();
  result.x_axis =
      result.x_axis - result.x_axis.dot(result.normal) * result.normal;
  if (result.x_axis.norm() < 1e-4f) {
    return false;
  }
  result.x_axis.normalize();

  result.y_axis = result.normal.cross(result.x_axis);
  if (result.y_axis.norm() < 1e-4f) {
    return false;
  }
  result.y_axis.normalize();

  const Eigen::Vector3f to_lidar = -result.origin;
  if (to_lidar.norm() > 1e-4f &&
      result.normal.dot(to_lidar.normalized()) < 0.0f) {
    result.normal = -result.normal;
    result.y_axis = -result.y_axis;
  }

  std::vector<size_t> saved_indices;
  const size_t saved_count =
      max_saved_points == 0 ? points.size() : std::min(max_saved_points, points.size());
  saved_indices.reserve(saved_count);
  if (saved_count == points.size()) {
    for (size_t i = 0; i < points.size(); ++i) {
      saved_indices.push_back(i);
    }
  } else {
    const double stride =
        static_cast<double>(points.size()) / static_cast<double>(saved_count);
    for (size_t i = 0; i < saved_count; ++i) {
      saved_indices.push_back(
          std::min(points.size() - 1, static_cast<size_t>(std::floor(i * stride))));
    }
  }

  float squared_plane_error = 0.0f;
  result.min_u = std::numeric_limits<float>::infinity();
  result.max_u = -std::numeric_limits<float>::infinity();
  result.min_v = std::numeric_limits<float>::infinity();
  result.max_v = -std::numeric_limits<float>::infinity();
  result.points_local_uv.reserve(saved_indices.size());
  for (size_t idx : saved_indices) {
    const PointXYZI& point = points[idx];
    const Eigen::Vector3f rel = ToEigen(point) - result.origin;
    const float u = rel.dot(result.x_axis);
    const float v = rel.dot(result.y_axis);
    const float w = rel.dot(result.normal);
    squared_plane_error += w * w;

    PointXYZI projected;
    projected.x = u;
    projected.y = v;
    projected.z = w;
    projected.reflectivity = point.reflectivity;
    projected.distance = point.distance;
    result.points_local_uv.push_back(projected);

    result.min_u = std::min(result.min_u, u);
    result.max_u = std::max(result.max_u, u);
    result.min_v = std::min(result.min_v, v);
    result.max_v = std::max(result.max_v, v);
  }

  result.plane_rms_m =
      std::sqrt(squared_plane_error / static_cast<float>(saved_indices.size()));
  *out = result;
  return true;
}

PlaneTemplate BuildFallbackTemplate(const std::vector<PointXYZI>& points,
                                    const Eigen::Vector3f& fallback_origin,
                                    size_t max_saved_points) {
  PlaneTemplate result;
  result.origin = fallback_origin;
  if (!points.empty()) {
    result.origin = Eigen::Vector3f::Zero();
    for (const auto& point : points) {
      result.origin += ToEigen(point);
    }
    result.origin /= static_cast<float>(points.size());
  }
  result.x_axis = Eigen::Vector3f::UnitX();
  result.y_axis = Eigen::Vector3f::UnitY();
  result.normal = Eigen::Vector3f::UnitZ();
  result.plane_rms_m = 0.0f;
  result.min_u = 0.0f;
  result.max_u = 0.0f;
  result.min_v = 0.0f;
  result.max_v = 0.0f;

  if (points.empty()) {
    return result;
  }

  const size_t saved_count =
      max_saved_points == 0 ? points.size() : std::min(max_saved_points, points.size());
  result.points_local_uv.reserve(saved_count);
  result.min_u = std::numeric_limits<float>::infinity();
  result.max_u = -std::numeric_limits<float>::infinity();
  result.min_v = std::numeric_limits<float>::infinity();
  result.max_v = -std::numeric_limits<float>::infinity();

  const double stride =
      saved_count == points.size()
          ? 1.0
          : static_cast<double>(points.size()) / static_cast<double>(saved_count);
  for (size_t i = 0; i < saved_count; ++i) {
    const size_t idx =
        saved_count == points.size()
            ? i
            : std::min(points.size() - 1, static_cast<size_t>(std::floor(i * stride)));
    const PointXYZI& point = points[idx];
    const Eigen::Vector3f rel = ToEigen(point) - result.origin;

    PointXYZI projected;
    projected.x = rel.x();
    projected.y = rel.y();
    projected.z = rel.z();
    projected.reflectivity = point.reflectivity;
    projected.distance = point.distance;
    result.points_local_uv.push_back(projected);

    result.min_u = std::min(result.min_u, projected.x);
    result.max_u = std::max(result.max_u, projected.x);
    result.min_v = std::min(result.min_v, projected.y);
    result.max_v = std::max(result.max_v, projected.y);
  }

  return result;
}

bool SavePlaneTemplate(const PlaneTemplate& tpl,
                       const Config& config,
                       const std::string& path) {
  std::ofstream out(path);
  if (!out.is_open()) {
    ROS_ERROR_STREAM("Failed to open candidate template output file: " << path);
    return false;
  }

  out << std::fixed << std::setprecision(6);
  out << "template_version: 1\n";
  out << "type: reflective_board_plane_points\n";
  out << "frame_id: " << YamlQuote(config.frame_id) << "\n";
  out << "created_from_topic: " << YamlQuote(config.input_topic) << "\n";
  out << "reflectivity_threshold: "
      << static_cast<int>(config.reflectivity_threshold) << "\n";
  out << "min_accumulation_sec: "
      << config.candidate_template_accumulation_sec << "\n";
  out << "max_recording_sec: "
      << config.candidate_template_accumulation_sec << "\n";
  out << "target_template_points: " << config.target_template_match_points << "\n";
  out << "cluster_tolerance_m: " << config.discovery_cluster_tolerance_m << "\n";
  out << "plane:\n";
  out << "  origin_xyz: ";
  WriteVector(out, tpl.origin);
  out << "\n";
  out << "  x_axis_xyz: ";
  WriteVector(out, tpl.x_axis);
  out << "\n";
  out << "  y_axis_xyz: ";
  WriteVector(out, tpl.y_axis);
  out << "\n";
  out << "  normal_xyz: ";
  WriteVector(out, tpl.normal);
  out << "\n";
  out << "  rms_m: " << tpl.plane_rms_m << "\n";
  out << "local_bounds:\n";
  out << "  min_u_m: " << tpl.min_u << "\n";
  out << "  max_u_m: " << tpl.max_u << "\n";
  out << "  min_v_m: " << tpl.min_v << "\n";
  out << "  max_v_m: " << tpl.max_v << "\n";
  out << "points_local_uv:\n";
  for (const auto& point : tpl.points_local_uv) {
    out << "  - [" << point.x << ", " << point.y << ", "
        << static_cast<int>(point.reflectivity) << "]\n";
  }

  return static_cast<bool>(out);
}

float Clamp01(float v) {
  return std::max(0.0f, std::min(1.0f, v));
}

float PackedRgb(uint8_t r, uint8_t g, uint8_t b) {
  uint32_t rgb = (static_cast<uint32_t>(r) << 16) |
                 (static_cast<uint32_t>(g) << 8) |
                 static_cast<uint32_t>(b);
  float packed = 0.0f;
  std::memcpy(&packed, &rgb, sizeof(packed));
  return packed;
}

float PackedRgb(const Rgb8& color) {
  return PackedRgb(color[0], color[1], color[2]);
}

geometry_msgs::Quaternion QuaternionFromAxes(const Eigen::Vector3f& x_axis,
                                             const Eigen::Vector3f& y_axis,
                                             const Eigen::Vector3f& z_axis) {
  Eigen::Matrix3f rot;
  rot.col(0) = x_axis;
  rot.col(1) = y_axis;
  rot.col(2) = z_axis;
  Eigen::Quaternionf q(rot);
  q.normalize();

  geometry_msgs::Quaternion msg;
  msg.x = q.x();
  msg.y = q.y();
  msg.z = q.z();
  msg.w = q.w();
  return msg;
}

template <typename Container>
std::vector<PointXYZI> PointsFromTimed(const Container& timed_points, size_t limit = 0) {
  std::vector<PointXYZI> points;
  const size_t n = timed_points.size();
  if (limit == 0 || n <= limit) {
    points.reserve(n);
    for (const auto& timed : timed_points) {
      points.push_back(timed.point);
    }
    return points;
  }

  points.reserve(limit);
  const double stride = static_cast<double>(n) / static_cast<double>(limit);
  for (size_t i = 0; i < limit; ++i) {
    const size_t idx = std::min(n - 1, static_cast<size_t>(std::floor(i * stride)));
    points.push_back(timed_points[idx].point);
  }
  return points;
}

sensor_msgs::PointCloud2 BuildPointCloud2Msg(const std::vector<float>& xyz_rgb,
                                             const std::vector<float>& intensities,
                                             const ros::Time& stamp,
                                             const std::string& frame_id) {
  const size_t n = xyz_rgb.size() / 4;
  const bool has_intensity = intensities.size() == n;

  sensor_msgs::PointCloud2 msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.height = 1;
  msg.width = static_cast<uint32_t>(n);
  msg.is_bigendian = false;
  msg.is_dense = true;
  msg.point_step = has_intensity ? 20 : 16;
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

  if (has_intensity) {
    field.name = "intensity";
    field.offset = 16;
    msg.fields.push_back(field);
  }

  msg.data.resize(n * msg.point_step);
  for (size_t i = 0; i < n; ++i) {
    uint8_t* dst = msg.data.data() + i * msg.point_step;
    std::memcpy(dst, xyz_rgb.data() + i * 4, 16);
    if (has_intensity) {
      std::memcpy(dst + 16, intensities.data() + i, sizeof(float));
    }
  }
  return msg;
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
    cluster.max_reflectivity = std::max(cluster.max_reflectivity, points[idx].reflectivity);
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
        if (!visited[j] && SquaredDistance(points[current], points[j]) <= tolerance_sq) {
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

std::vector<Eigen::Vector2f> NormalizeUvPoints(
    const std::vector<Eigen::Vector2f>& points) {
  std::vector<Eigen::Vector2f> normalized;
  if (points.size() < 3) {
    return normalized;
  }

  Eigen::Vector2f mean = Eigen::Vector2f::Zero();
  for (const auto& p : points) {
    mean += p;
  }
  mean /= static_cast<float>(points.size());

  Eigen::Matrix2f covariance = Eigen::Matrix2f::Zero();
  for (const auto& p : points) {
    const Eigen::Vector2f d = p - mean;
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
  for (const auto& p : points) {
    const Eigen::Vector2f q = axes.transpose() * (p - mean);
    normalized.push_back(q);
    min_x = std::min(min_x, q.x());
    max_x = std::max(max_x, q.x());
    min_y = std::min(min_y, q.y());
    max_y = std::max(max_y, q.y());
  }

  const float span_x = std::max(1e-4f, max_x - min_x);
  const float span_y = std::max(1e-4f, max_y - min_y);
  const float scale = std::max(span_x, span_y);
  for (auto& p : normalized) {
    p /= scale;
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

  for (const auto& p : points) {
    float x = flip_x ? -p.x() : p.x();
    float y = flip_y ? -p.y() : p.y();
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
  for (float& v : grid) {
    v = std::sqrt(v);
    norm += v * v;
  }
  norm = std::sqrt(norm);
  if (norm > 1e-6f) {
    for (float& v : grid) {
      v /= norm;
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
    return false;
  }

  std::ifstream in(config.template_path);
  if (!in.is_open()) {
    ROS_WARN_STREAM("Template matching enabled, but failed to open template: "
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
    ROS_WARN_STREAM("Template has too few points: " << model->points_uv.size()
                    << "/" << config.min_template_points);
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
  ROS_INFO_STREAM("Loaded reflective board template: " << config.template_path
                  << " points=" << model->points_uv.size()
                  << " grid=" << config.template_grid_cols
                  << "x" << config.template_grid_rows
                  << " mid=" << config.template_mid_grid_cols
                  << "x" << config.template_mid_grid_rows
                  << " coarse=" << config.template_coarse_grid_cols
                  << "x" << config.template_coarse_grid_rows);
  return true;
}

class ReflectiveBoardIdentifier {
 public:
  ReflectiveBoardIdentifier(ros::NodeHandle nh, ros::NodeHandle pnh)
      : nh_(nh), pnh_(pnh) {
    LoadConfig();
    LoadTemplateModel(config_, &template_model_);
    UpdateResourceUsage(&resource_usage_);

    raw_cloud_pub_ =
        pnh_.advertise<sensor_msgs::PointCloud2>("raw_cloud", 1);
    candidate_cloud_pub_ =
        pnh_.advertise<sensor_msgs::PointCloud2>("candidate_cloud", 1);
    high_cloud_pub_ =
        pnh_.advertise<sensor_msgs::PointCloud2>("high_reflective_cloud", 1);
    template_match_cloud_pub_ =
        pnh_.advertise<sensor_msgs::PointCloud2>("template_match_cloud", 1);
    marker_pub_ =
        pnh_.advertise<visualization_msgs::MarkerArray>("markers", 1);
    target_pose_pub_ =
        pnh_.advertise<geometry_msgs::PoseStamped>("target_pose", 1);
    best_score_pub_ =
        pnh_.advertise<std_msgs::Float32>("best_score", 1);

    sub_ = nh_.subscribe(config_.input_topic,
                        static_cast<uint32_t>(config_.input_queue_size),
                        &ReflectiveBoardIdentifier::PointCloudCallback, this);
    timer_ = nh_.createTimer(ros::Duration(0.1),
                             &ReflectiveBoardIdentifier::TimerCallback, this);

    ROS_INFO_STREAM("reflective_board_identifier_node listening on "
                    << config_.input_topic
                    << ", discovery_window=" << config_.discovery_window_sec
                    << "s, min_roi_accumulation="
                    << config_.min_roi_accumulation_sec
                    << "s, record_detection_result="
                    << (config_.record_detection_result ? "true" : "false")
                    << ", result_output_dir=" << config_.result_output_dir);
  }

 private:
  void LoadConfig() {
    std::string input_topic = config_.input_topic;
    pnh_.param("input_topic", input_topic, input_topic);
    pnh_.param("task1/input_topic", input_topic, input_topic);
    config_.input_topic = input_topic;

    pnh_.param("frame_id", config_.frame_id, config_.frame_id);
    pnh_.param("task1/frame_id", config_.frame_id, config_.frame_id);
    pnh_.param("input_queue_size", config_.input_queue_size, config_.input_queue_size);
    pnh_.param("task1/input_queue_size", config_.input_queue_size,
               config_.input_queue_size);

    int reflectivity_threshold = config_.reflectivity_threshold;
    pnh_.param("reflectivity_threshold", reflectivity_threshold, reflectivity_threshold);
    pnh_.param("task1/reflectivity_threshold", reflectivity_threshold,
               reflectivity_threshold);
    config_.reflectivity_threshold =
        static_cast<uint8_t>(std::max(0, std::min(255, reflectivity_threshold)));

    pnh_.param("min_distance_m", config_.min_distance_m, config_.min_distance_m);
    pnh_.param("max_distance_m", config_.max_distance_m, config_.max_distance_m);
    pnh_.param("task1/min_distance_m", config_.min_distance_m,
               config_.min_distance_m);
    pnh_.param("task1/max_distance_m", config_.max_distance_m,
               config_.max_distance_m);

    pnh_.param("task1/discovery_window_sec", config_.discovery_window_sec,
               config_.discovery_window_sec);
    pnh_.param("task1/min_roi_accumulation_sec", config_.min_roi_accumulation_sec,
               config_.min_roi_accumulation_sec);
    pnh_.param("task1/score_update_sec", config_.score_update_sec,
               config_.score_update_sec);
    pnh_.param("task1/max_accumulation_sec", config_.max_accumulation_sec,
               config_.max_accumulation_sec);

    pnh_.param("task1/discovery_cluster_tolerance_m",
               config_.discovery_cluster_tolerance_m,
               config_.discovery_cluster_tolerance_m);
    int min_discovery_cluster_points =
        static_cast<int>(config_.min_discovery_cluster_points);
    int max_discovery_cluster_points =
        static_cast<int>(config_.max_discovery_cluster_points);
    int max_candidates = static_cast<int>(config_.max_candidates);
    int max_discovery_high_points =
        static_cast<int>(config_.max_discovery_high_points);
    pnh_.param("task1/min_discovery_cluster_points", min_discovery_cluster_points,
               min_discovery_cluster_points);
    pnh_.param("task1/max_discovery_cluster_points", max_discovery_cluster_points,
               max_discovery_cluster_points);
    pnh_.param("task1/max_candidates", max_candidates, max_candidates);
    pnh_.param("task1/max_discovery_high_points", max_discovery_high_points,
               max_discovery_high_points);
    config_.min_discovery_cluster_points =
        static_cast<size_t>(std::max(1, min_discovery_cluster_points));
    config_.max_discovery_cluster_points =
        static_cast<size_t>(std::max(1, max_discovery_cluster_points));
    config_.max_candidates = static_cast<size_t>(std::max(1, max_candidates));
    config_.max_discovery_high_points =
        static_cast<size_t>(std::max(100, max_discovery_high_points));

    pnh_.param("task1/roi_padding_m", config_.roi_padding_m,
               config_.roi_padding_m);
    pnh_.param("task1/min_roi_radius_m", config_.min_roi_radius_m,
               config_.min_roi_radius_m);
    pnh_.param("task1/max_roi_radius_m", config_.max_roi_radius_m,
               config_.max_roi_radius_m);
    int max_roi_points = static_cast<int>(config_.max_roi_points_per_candidate);
    int max_high_points = static_cast<int>(config_.max_high_points_per_candidate);
    pnh_.param("task1/max_roi_points_per_candidate", max_roi_points,
               max_roi_points);
    pnh_.param("task1/max_high_points_per_candidate", max_high_points,
               max_high_points);
    config_.max_roi_points_per_candidate =
        static_cast<size_t>(std::max(100, max_roi_points));
    config_.max_high_points_per_candidate =
        static_cast<size_t>(std::max(100, max_high_points));

    pnh_.param("task1/confirm_score_threshold", config_.confirm_score_threshold,
               config_.confirm_score_threshold);
    pnh_.param("task1/confirm_margin_threshold", config_.confirm_margin_threshold,
               config_.confirm_margin_threshold);
    pnh_.param("task1/template_path", config_.template_path,
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
    pnh_.param("task1/min_template_score", config_.min_template_score,
               config_.min_template_score);
    pnh_.param("task1/far_min_template_score", config_.far_min_template_score,
               config_.far_min_template_score);
    pnh_.param("task1/good_template_score", config_.good_template_score,
               config_.good_template_score);
    pnh_.param("task1/template_plane_tolerance_m",
               config_.template_plane_tolerance_m,
               config_.template_plane_tolerance_m);
    config_.good_template_score =
        std::max(config_.min_template_score + 1e-3,
                 std::min(1.0, config_.good_template_score));
    config_.template_plane_tolerance_m =
        std::max(0.001f, config_.template_plane_tolerance_m);
    pnh_.param("task1/publish_raw_cloud", config_.publish_raw_cloud,
               config_.publish_raw_cloud);
    pnh_.param("task1/publish_candidate_cloud", config_.publish_candidate_cloud,
               config_.publish_candidate_cloud);
    pnh_.param("task1/publish_high_reflective_cloud",
               config_.publish_high_reflective_cloud,
               config_.publish_high_reflective_cloud);
    pnh_.param("task1/publish_template_match_cloud",
               config_.publish_template_match_cloud,
               config_.publish_template_match_cloud);
    pnh_.param("task1/publish_markers", config_.publish_markers,
               config_.publish_markers);
    int max_raw_cloud_points = static_cast<int>(config_.max_raw_cloud_points);
    pnh_.param("task1/max_raw_cloud_points", max_raw_cloud_points,
               max_raw_cloud_points);
    config_.max_raw_cloud_points =
        static_cast<size_t>(std::max(1000, max_raw_cloud_points));
    pnh_.param("task1/record_detection_result",
               config_.record_detection_result,
               config_.record_detection_result);
    pnh_.param("task1/result_output_dir", config_.result_output_dir,
               config_.result_output_dir);
    pnh_.param("task1/record_candidate_templates_after_confirm",
               config_.record_candidate_templates_after_confirm,
               config_.record_candidate_templates_after_confirm);
    pnh_.param("task1/candidate_template_accumulation_sec",
               config_.candidate_template_accumulation_sec,
               config_.candidate_template_accumulation_sec);
    pnh_.param("task1/candidate_template_output_dir",
               config_.candidate_template_output_dir,
               config_.candidate_template_output_dir);
    int candidate_template_min_points =
        static_cast<int>(config_.candidate_template_min_points);
    int candidate_template_max_saved_points =
        static_cast<int>(config_.candidate_template_max_saved_points);
    pnh_.param("task1/candidate_template_min_points",
               candidate_template_min_points,
               candidate_template_min_points);
    pnh_.param("task1/candidate_template_max_saved_points",
               candidate_template_max_saved_points,
               candidate_template_max_saved_points);
    config_.candidate_template_min_points =
        static_cast<size_t>(std::max(0, candidate_template_min_points));
    config_.candidate_template_max_saved_points =
        static_cast<size_t>(std::max(0, candidate_template_max_saved_points));
    pnh_.param("verbose", config_.verbose, config_.verbose);
    pnh_.param("task1/verbose", config_.verbose, config_.verbose);

    config_.discovery_window_sec = std::max(0.2, config_.discovery_window_sec);
    config_.min_roi_accumulation_sec =
        std::max(0.2, config_.min_roi_accumulation_sec);
    config_.score_update_sec = std::max(0.2, config_.score_update_sec);
    config_.max_accumulation_sec =
        std::max(config_.min_roi_accumulation_sec, config_.max_accumulation_sec);
    config_.candidate_template_accumulation_sec =
        std::max(0.1, config_.candidate_template_accumulation_sec);
    config_.min_roi_radius_m = std::max(0.05f, config_.min_roi_radius_m);
    config_.max_roi_radius_m =
        std::max(config_.min_roi_radius_m, config_.max_roi_radius_m);
  }

  void PointCloudCallback(const livox_ros_driver2::CustomMsg::ConstPtr& msg) {
    ros::Time stamp = msg->header.stamp;
    if (stamp.isZero()) {
      stamp = ros::Time::now();
    }
    last_stamp_ = stamp;

    last_frame_total_points_ = msg->points.size();
    last_frame_valid_points_ = 0;
    last_frame_high_points_ = 0;
    last_frame_roi_hits_ = 0;
    last_raw_points_.clear();
    last_raw_points_.reserve(
        std::min(config_.max_raw_cloud_points, msg->points.size()));

    for (const auto& raw_point : msg->points) {
      const PointXYZI point = ToPoint(raw_point);
      if (!IsValidPoint(point) ||
          point.distance < config_.min_distance_m ||
          point.distance > config_.max_distance_m) {
        continue;
      }

      ++last_frame_valid_points_;
      if (last_raw_points_.size() < config_.max_raw_cloud_points) {
        last_raw_points_.push_back(point);
      }

      if (IsReflectiveCandidate(point, config_)) {
        ++last_frame_high_points_;
        if (discovery_started_at_.isZero()) {
          discovery_started_at_ = stamp;
        }
        discovery_high_points_.push_back({point, stamp});
      }

      if (state_ == State::kAccumulate || CandidateTemplateRecordingActive()) {
        for (auto& candidate : candidates_) {
          if (SquaredDistance(point, candidate.seed_center) <=
              candidate.roi_radius * candidate.roi_radius) {
            ++last_frame_roi_hits_;
            PushCapped({point, stamp}, config_.max_roi_points_per_candidate,
                       &candidate.all_points);
            if (IsReflectiveCandidate(point, config_)) {
              PushCapped({point, stamp}, config_.max_high_points_per_candidate,
                         &candidate.high_points);
            }
          }
        }
      }
    }

    PruneDiscoveryWindow(stamp);
  }

  void TimerCallback(const ros::TimerEvent&) {
    const ros::Time stamp = last_stamp_.isZero() ? ros::Time::now() : last_stamp_;
    UpdateResourceUsage(&resource_usage_);
    PruneDiscoveryWindow(stamp);
    LogStatus(stamp);

    if (state_ == State::kDiscover) {
      TryStartAccumulation(stamp);
      PublishVisualization(stamp);
      return;
    }

    if (state_ == State::kAccumulate) {
      const bool ready_to_score =
          (last_score_stamp_.isZero() ||
           (stamp - last_score_stamp_).toSec() >= config_.score_update_sec);
      if (ready_to_score &&
          !accumulation_started_at_.isZero() &&
          (stamp - accumulation_started_at_).toSec() >=
              config_.min_roi_accumulation_sec) {
        ScoreCandidates(stamp);
        last_score_stamp_ = stamp;
      }

      if (state_ == State::kAccumulate &&
          !accumulation_started_at_.isZero() &&
          (stamp - accumulation_started_at_).toSec() >= config_.max_accumulation_sec) {
        ROS_WARN_STREAM("No unique target after " << config_.max_accumulation_sec
                        << "s; restarting discovery.");
        ResetToDiscover();
      }
    }

    MaybeFinishCandidateTemplateRecording(stamp);
    PublishVisualization(stamp);
    PublishBestScore();
    PublishTargetPose(stamp);
  }

  void LogStatus(const ros::Time& stamp) const {
    const double discovery_age =
        discovery_high_points_.empty()
            ? 0.0
            : std::max(0.0, (stamp - discovery_high_points_.front().stamp).toSec());
    ROS_INFO_STREAM_THROTTLE(
        1.0,
        "Task1 state=" << StateName(state_)
        << " frame(total/valid/high)=" << last_frame_total_points_
        << "/" << last_frame_valid_points_
        << "/" << last_frame_high_points_
        << " discovery_high=" << discovery_high_points_.size()
        << " age=" << discovery_age << "s"
        << " candidates=" << candidates_.size()
        << " roi_hits=" << last_frame_roi_hits_
        << " threshold=" << static_cast<int>(config_.reflectivity_threshold)
        << " confirmed_id=" << confirmed_candidate_id_
        << " confirmed_score=" << confirmed_score_.score);
  }

  void PruneDiscoveryWindow(const ros::Time& stamp) {
    const ros::Duration window(config_.discovery_window_sec);
    while (!discovery_high_points_.empty() &&
           stamp - discovery_high_points_.front().stamp > window) {
      discovery_high_points_.pop_front();
    }
    while (discovery_high_points_.size() > config_.max_discovery_high_points) {
      discovery_high_points_.pop_front();
    }
    if (discovery_high_points_.empty()) {
      discovery_started_at_ = ros::Time(0);
    }
  }

  void TryStartAccumulation(const ros::Time& stamp) {
    if (discovery_high_points_.empty()) {
      return;
    }
    const double covered =
        discovery_started_at_.isZero()
            ? 0.0
            : (stamp - discovery_started_at_).toSec();
    if (covered < config_.discovery_window_sec) {
      return;
    }

    const std::vector<PointXYZI> high_points =
        PointsFromTimed(discovery_high_points_, config_.max_discovery_high_points);
    std::vector<Cluster> clusters = EuclideanCluster(
        high_points,
        config_.discovery_cluster_tolerance_m,
        config_.min_discovery_cluster_points,
        config_.max_discovery_cluster_points);
    if (clusters.empty()) {
      ROS_WARN_STREAM_THROTTLE(
          1.0,
          "Task1 discovery has " << high_points.size()
          << " high-reflectivity point(s), but no cluster passed filters"
          << " (tolerance=" << config_.discovery_cluster_tolerance_m
          << ", min_points=" << config_.min_discovery_cluster_points
          << ", max_points=" << config_.max_discovery_cluster_points << ").");
      return;
    }

    candidates_.clear();
    const size_t keep = std::min(config_.max_candidates, clusters.size());
    for (size_t i = 0; i < keep; ++i) {
      Candidate candidate;
      candidate.id = next_candidate_id_++;
      candidate.seed_center = clusters[i].center;
      candidate.created_at = stamp;

      float radius = config_.min_roi_radius_m;
      for (size_t idx : clusters[i].indices) {
        radius = std::max(radius,
                          std::sqrt(SquaredDistance(high_points[idx],
                                                    candidate.seed_center)) +
                              config_.roi_padding_m);
      }
      candidate.roi_radius =
          std::max(config_.min_roi_radius_m,
                   std::min(config_.max_roi_radius_m, radius));

      for (size_t idx : clusters[i].indices) {
        PushCapped({high_points[idx], stamp}, config_.max_high_points_per_candidate,
                   &candidate.high_points);
      }
      ROS_INFO_STREAM("Task1 candidate id=" << candidate.id
                      << " seed=(" << candidate.seed_center.x()
                      << ", " << candidate.seed_center.y()
                      << ", " << candidate.seed_center.z()
                      << ") roi_radius=" << candidate.roi_radius
                      << " initial_high_points=" << candidate.high_points.size()
                      << " discovery_cluster_points="
                      << clusters[i].indices.size());
      candidates_.push_back(candidate);
    }

    state_ = State::kAccumulate;
    accumulation_started_at_ = stamp;
    UpdateResourceUsage(&resource_usage_);
    accumulation_start_cpu_time_sec_ = resource_usage_.latest.cpu_time_sec;
    discovery_started_at_ = ros::Time(0);
    last_score_stamp_ = ros::Time(0);
    latest_scores_.clear();
    best_score_ = CandidateScore();
    confirmed_ = false;

    ROS_INFO_STREAM("Task1 discovery found " << candidates_.size()
                    << " candidate ROI(s) from " << clusters.size()
                    << " cluster(s); accumulating local full clouds.");
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

    pose->pose.position.x = origin.x();
    pose->pose.position.y = origin.y();
    pose->pose.position.z = origin.z();
    pose->pose.orientation = QuaternionFromAxes(x_axis, y_axis, z_axis);
    pose->x_axis = x_axis;
    pose->y_axis = y_axis;
    pose->z_axis = z_axis;
    return true;
  }

  double CandidateDistance(const Candidate& candidate) const {
    return static_cast<double>(candidate.seed_center.norm());
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

    const Eigen::Vector3f origin(board.pose.position.x,
                                 board.pose.position.y,
                                 board.pose.position.z);
    std::vector<Eigen::Vector2f> uv_points;
    uv_points.reserve(high_points.size());
    const float max_plane_offset =
        std::max(0.08f, 2.0f * config_.template_plane_tolerance_m);
    for (const auto& point : high_points) {
      const Eigen::Vector3f rel = ToEigen(point) - origin;
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
        std::max(1e-6,
                 config_.good_template_score - min_template_score);
    const double confidence = Clamp01(static_cast<float>(normalized_template));
    return config_.confirm_score_threshold +
           (1.0 - config_.confirm_score_threshold) * confidence;
  }

  std::vector<PointXYZI> SelectTemplateMatchPoints(
      const std::vector<PointXYZI>& high_points,
      const BoardPose& board) const {
    std::vector<PointXYZI> selected;
    if (high_points.empty()) {
      return selected;
    }

    const Eigen::Vector3f origin(board.pose.position.x,
                                 board.pose.position.y,
                                 board.pose.position.z);
    const float max_plane_offset =
        std::max(0.08f, 2.0f * config_.template_plane_tolerance_m);
    selected.reserve(high_points.size());
    for (const auto& point : high_points) {
      const Eigen::Vector3f rel = ToEigen(point) - origin;
      const float w = rel.dot(board.z_axis);
      if (std::fabs(w) <= max_plane_offset) {
        selected.push_back(point);
      }
    }
    return selected;
  }

  Rgb8 CandidateDisplayColor(size_t candidate_index, int candidate_id) const {
    if (confirmed_ && candidate_id == confirmed_candidate_id_) {
      return kTargetGreen;
    }
    return kCandidatePalette[candidate_index % kCandidatePalette.size()];
  }

  bool TryTemplateScore(const std::vector<PointXYZI>& high_points,
                        double distance_m,
                        bool allow_waiting,
                        CandidateScore* result) const {
    const size_t required_points = RequiredTemplatePoints(distance_m);
    const double required_score = RequiredTemplateScore(distance_m);
    if (result == nullptr || !template_model_.loaded ||
        high_points.size() < config_.min_template_points) {
      return false;
    }
    if (allow_waiting && high_points.size() < required_points) {
      std::ostringstream oss;
      oss << "waiting for template points: " << high_points.size()
          << "/" << required_points
          << " distance=" << distance_m << "m";
      result->reason = oss.str();
      return false;
    }

    BoardPose pose;
    if (!EstimatePoseFromHighPoints(high_points, &pose)) {
      result->reason = "could not estimate template pose from high-reflectivity points";
      return false;
    }

    const double template_score = TemplateScore(high_points, pose);
    result->template_score = template_score;
    if (template_score < required_score) {
      std::ostringstream oss;
      oss << "template score too low: " << template_score
          << "/" << required_score
          << " distance=" << distance_m << "m";
      result->reason = oss.str();
      return false;
    }

    result->valid_pose = true;
    result->reason = "ok(template)";
    result->board = pose;
    result->template_match_points = SelectTemplateMatchPoints(high_points, pose);
    result->score = CalibratedTemplateScore(template_score, required_score);
    result->score = std::min(1.0, result->score);
    return true;
  }

  CandidateScore ComputeScore(const Candidate& candidate) const {
    CandidateScore result;
    result.candidate_id = candidate.id;
    result.high_points = candidate.high_points.size();
    result.all_points = candidate.all_points.size();

    const std::vector<PointXYZI> high_points =
        PointsFromTimed(candidate.high_points, config_.max_high_points_per_candidate);
    const double candidate_distance = CandidateDistance(candidate);
    const bool near_max_accumulation =
        !accumulation_started_at_.isZero() &&
        (last_stamp_ - accumulation_started_at_).toSec() >=
            0.8 * config_.max_accumulation_sec;

    if (TryTemplateScore(high_points, candidate_distance,
                         !near_max_accumulation, &result)) {
      return result;
    }

    if (result.reason == "not scored") {
      std::ostringstream oss;
      oss << "template matching failed"
          << " (high_points=" << result.high_points
          << ", min_template_points=" << config_.min_template_points
          << ", required_template_points="
          << RequiredTemplatePoints(candidate_distance)
          << ", distance=" << candidate_distance
          << ", template_loaded="
          << (template_model_.loaded ? "true" : "false") << ")";
      result.reason = oss.str();
    }
    return result;
  }

  void ScoreCandidates(const ros::Time& stamp) {
    if (confirmed_) {
      return;
    }

    latest_scores_.clear();
    latest_scores_.reserve(candidates_.size());
    last_template_match_points_.clear();
    for (const auto& candidate : candidates_) {
      latest_scores_.push_back(ComputeScore(candidate));
    }

    std::sort(latest_scores_.begin(), latest_scores_.end(),
              [](const CandidateScore& a, const CandidateScore& b) {
                return a.score > b.score;
              });

    if (latest_scores_.empty()) {
      ROS_WARN_STREAM("Task1 score update requested, but there are no candidates.");
      return;
    }

    best_score_ = latest_scores_.front();
    last_template_match_points_ = best_score_.template_match_points;
    const double second_score =
        latest_scores_.size() >= 2 ? latest_scores_[1].score : 0.0;
    const double margin = best_score_.score - second_score;

    ROS_INFO_STREAM("Task1 candidate scores: count=" << latest_scores_.size()
                    << " mode=template"
                    << " best=" << best_score_.score
                    << " second=" << second_score
                    << " margin=" << margin
                    << " confirm_threshold="
                    << config_.confirm_score_threshold
                    << " margin_threshold="
                    << config_.confirm_margin_threshold);
    for (size_t rank = 0; rank < latest_scores_.size(); ++rank) {
      const auto& score = latest_scores_[rank];
      ROS_INFO_STREAM(
          "  rank=" << (rank + 1)
          << " id=" << score.candidate_id
          << " score=" << score.score
          << " valid=" << (score.valid_pose ? "true" : "false")
          << " reason=\"" << score.reason << "\""
          << " template_score=" << score.template_score
          << " points(high/all)=" << score.high_points
          << "/" << score.all_points);
    }

    ROS_INFO_STREAM(
        "Task1 scores: best id=" << best_score_.candidate_id
        << " score=" << best_score_.score
        << " second=" << second_score
        << " margin=" << margin
        << " valid_pose=" << (best_score_.valid_pose ? "true" : "false")
        << " template_score=" << best_score_.template_score
        << " points(high/all)=" << best_score_.high_points
        << "/" << best_score_.all_points);

    if (best_score_.valid_pose &&
        best_score_.score >= config_.confirm_score_threshold &&
        margin >= config_.confirm_margin_threshold) {
      confirmed_ = true;
      confirmed_candidate_id_ = best_score_.candidate_id;
      state_ = State::kConfirmed;
      confirmed_score_ = best_score_;
      last_template_match_points_ = confirmed_score_.template_match_points;
      SaveDetectionResult(stamp, second_score, margin);
      StartCandidateTemplateRecording(stamp);
      ROS_INFO_STREAM("Task1 target confirmed at t=" << stamp.toSec()
                      << " candidate_id=" << confirmed_score_.candidate_id
                      << " score=" << confirmed_score_.score
                      << " margin=" << margin);
    }
  }

  bool CandidateTemplateRecordingActive() const {
    return config_.record_candidate_templates_after_confirm &&
           candidate_template_recording_ &&
           !candidate_template_started_at_.isZero();
  }

  void StartCandidateTemplateRecording(const ros::Time& stamp) {
    if (!config_.record_candidate_templates_after_confirm || candidates_.empty()) {
      return;
    }

    candidate_template_recording_ = true;
    candidate_template_saved_ = false;
    candidate_template_started_at_ = stamp;
    ++candidate_template_batch_count_;
    for (auto& candidate : candidates_) {
      candidate.created_at = stamp;
      candidate.all_points.clear();
      candidate.high_points.clear();
    }

    ROS_INFO_STREAM("Task1 candidate template recording started: candidates="
                    << candidates_.size()
                    << " duration="
                    << config_.candidate_template_accumulation_sec
                    << "s output_dir="
                    << config_.candidate_template_output_dir);
  }

  void MaybeFinishCandidateTemplateRecording(const ros::Time& stamp) {
    if (!CandidateTemplateRecordingActive()) {
      return;
    }
    const double elapsed = (stamp - candidate_template_started_at_).toSec();
    if (elapsed < config_.candidate_template_accumulation_sec) {
      return;
    }
    SaveCandidateTemplates(stamp);
    candidate_template_recording_ = false;
    candidate_template_saved_ = true;
    candidate_template_started_at_ = ros::Time(0);
  }

  void SaveCandidateTemplates(const ros::Time& stamp) {
    if (!EnsureDirectoryExists(config_.candidate_template_output_dir)) {
      ROS_WARN_STREAM("Task1 failed to create candidate template directory: "
                      << config_.candidate_template_output_dir);
      return;
    }

    size_t saved = 0;
    size_t interference_index = 0;
    for (const auto& candidate : candidates_) {
      const std::vector<PointXYZI> high_points =
          PointsFromTimed(candidate.high_points, config_.max_high_points_per_candidate);

      PlaneTemplate tpl;
      if (high_points.size() >= 3) {
        if (!EstimatePlaneTemplateFromPoints(
                high_points, config_.candidate_template_max_saved_points, &tpl)) {
          ROS_WARN_STREAM("Task1 candidate template plane estimation failed id="
                          << candidate.id
                          << " high_points=" << high_points.size()
                          << "; saving fallback template.");
          tpl = BuildFallbackTemplate(high_points, candidate.seed_center,
                                      config_.candidate_template_max_saved_points);
        }
      } else {
        tpl = BuildFallbackTemplate(high_points, candidate.seed_center,
                                    config_.candidate_template_max_saved_points);
      }

      std::string template_label;
      if (candidate.id == confirmed_candidate_id_) {
        template_label = "target";
      } else {
        std::ostringstream label;
        label << "interference" << interference_index++;
        template_label = label.str();
      }

      std::ostringstream filename;
      filename << "candidate_template_" << stamp.sec << "_"
               << std::setw(9) << std::setfill('0') << stamp.nsec
               << "_batch" << std::setw(3) << std::setfill('0')
               << candidate_template_batch_count_
               << "_" << template_label
               << ".yaml";
      const std::string path =
          JoinPath(config_.candidate_template_output_dir, filename.str());
      if (SavePlaneTemplate(tpl, config_, path)) {
        ++saved;
        ROS_INFO_STREAM("Task1 saved candidate template id=" << candidate.id
                        << " label=" << template_label
                        << " points=" << high_points.size()
                        << " path=" << path);
      }
    }

    ROS_INFO_STREAM("Task1 candidate template recording finished: saved="
                    << saved << "/" << candidates_.size()
                    << " elapsed="
                    << (stamp - candidate_template_started_at_).toSec()
                    << "s");
  }

  void SaveDetectionResult(const ros::Time& stamp,
                           double second_score,
                           double margin) {
    if (!config_.record_detection_result) {
      return;
    }
    UpdateResourceUsage(&resource_usage_);
    if (!EnsureDirectoryExists(config_.result_output_dir)) {
      ROS_WARN_STREAM("Task1 failed to create result directory: "
                      << config_.result_output_dir);
      return;
    }

    const int sequence = ++detection_result_count_;
    std::ostringstream filename;
    filename << "detection_result_" << stamp.sec << "_"
             << std::setw(9) << std::setfill('0') << stamp.nsec
             << "_" << std::setw(3) << std::setfill('0') << sequence
             << ".yaml";
    const std::string path = JoinPath(config_.result_output_dir, filename.str());

    const Candidate* confirmed_candidate = nullptr;
    for (const auto& candidate : candidates_) {
      if (candidate.id == confirmed_score_.candidate_id) {
        confirmed_candidate = &candidate;
        break;
      }
    }

    const double accumulation_sec =
        accumulation_started_at_.isZero()
            ? 0.0
            : std::max(0.0, (stamp - accumulation_started_at_).toSec());
    const double candidate_accumulation_sec =
        confirmed_candidate == nullptr || confirmed_candidate->created_at.isZero()
            ? accumulation_sec
            : std::max(0.0, (stamp - confirmed_candidate->created_at).toSec());
    const double accumulation_cpu_time_sec =
        std::max(0.0, static_cast<double>(resource_usage_.latest.cpu_time_sec) -
                          static_cast<double>(accumulation_start_cpu_time_sec_));
    const double accumulation_cpu_percent =
        accumulation_sec > 1e-6 ? (accumulation_cpu_time_sec / accumulation_sec) * 100.0
                                : 0.0;

    std::ofstream out(path);
    if (!out) {
      ROS_WARN_STREAM("Task1 failed to open detection result file: " << path);
      return;
    }

    out << std::fixed << std::setprecision(6);
    out << "type: reflective_board_detection_result\n";
    out << "sequence: " << sequence << "\n";
    out << "confirmed_stamp_sec: " << stamp.toSec() << "\n";
    out << "accumulation_started_stamp_sec: "
        << (accumulation_started_at_.isZero() ? 0.0
                                              : accumulation_started_at_.toSec())
        << "\n";
    out << "actual_accumulation_sec: " << accumulation_sec << "\n";
    out << "candidate_accumulation_sec: " << candidate_accumulation_sec << "\n";
    out << "state: " << YamlQuote(StateName(state_)) << "\n";
    out << "candidate_count: " << candidates_.size() << "\n";
    out << "confirmed_candidate_id: " << confirmed_score_.candidate_id << "\n";
    out << "score: " << confirmed_score_.score << "\n";
    out << "second_score: " << second_score << "\n";
    out << "margin: " << margin << "\n";
    out << "template_score: " << confirmed_score_.template_score << "\n";
    out << "reason: " << YamlQuote(confirmed_score_.reason) << "\n";
    out << "used_high_reflective_points: " << confirmed_score_.high_points << "\n";
    out << "roi_total_points: " << confirmed_score_.all_points << "\n";
    out << "template_match_points: "
        << confirmed_score_.template_match_points.size() << "\n";
    out << "reflectivity_threshold: "
        << static_cast<int>(config_.reflectivity_threshold) << "\n";
    out << "template_path: " << YamlQuote(config_.template_path) << "\n";
    out << "resource_usage:\n";
    out << "  cpu_percent_single_core: "
        << resource_usage_.latest.cpu_percent << "\n";
    out << "  cpu_percent_avg_during_accumulation: "
        << accumulation_cpu_percent << "\n";
    out << "  cpu_time_during_accumulation_sec: "
        << accumulation_cpu_time_sec << "\n";
    out << "  cpu_time_sec: " << resource_usage_.latest.cpu_time_sec << "\n";
    out << "  memory_rss_kb: " << resource_usage_.latest.memory_rss_kb << "\n";
    out << "  memory_rss_mb: "
        << static_cast<double>(resource_usage_.latest.memory_rss_kb) / 1024.0
        << "\n";
    out << "  memory_vm_kb: " << resource_usage_.latest.memory_vm_kb << "\n";
    out << "  memory_vm_mb: "
        << static_cast<double>(resource_usage_.latest.memory_vm_kb) / 1024.0
        << "\n";
    if (confirmed_candidate != nullptr) {
      out << "candidate_roi:\n";
      out << "  radius_m: " << confirmed_candidate->roi_radius << "\n";
      out << "  seed_center: ["
          << confirmed_candidate->seed_center.x() << ", "
          << confirmed_candidate->seed_center.y() << ", "
          << confirmed_candidate->seed_center.z() << "]\n";
    }
    out << "pose:\n";
    out << "  frame_id: " << YamlQuote(config_.frame_id) << "\n";
    out << "  position: ["
        << confirmed_score_.board.pose.position.x << ", "
        << confirmed_score_.board.pose.position.y << ", "
        << confirmed_score_.board.pose.position.z << "]\n";
    out << "  orientation_xyzw: ["
        << confirmed_score_.board.pose.orientation.x << ", "
        << confirmed_score_.board.pose.orientation.y << ", "
        << confirmed_score_.board.pose.orientation.z << ", "
        << confirmed_score_.board.pose.orientation.w << "]\n";

    if (!out) {
      ROS_WARN_STREAM("Task1 failed while writing detection result file: " << path);
      return;
    }
    ROS_INFO_STREAM("Task1 saved detection result to " << path);
  }

  void PublishBestScore() {
    std_msgs::Float32 msg;
    msg.data = static_cast<float>(best_score_.score);
    best_score_pub_.publish(msg);
  }

  void PublishTargetPose(const ros::Time& stamp) {
    if (!confirmed_ || !confirmed_score_.valid_pose) {
      return;
    }

    geometry_msgs::PoseStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = config_.frame_id;
    msg.pose = confirmed_score_.board.pose;
    target_pose_pub_.publish(msg);
  }

  void PublishVisualization(const ros::Time& stamp) {
    PublishRawCloud(stamp);
    PublishCandidateClouds(stamp);
    PublishTemplateMatchCloud(stamp);
    PublishMarkers(stamp);
  }

  void PublishRawCloud(const ros::Time& stamp) {
    if (!config_.publish_raw_cloud) {
      return;
    }

    std::vector<float> xyz_rgb;
    std::vector<float> intensities;
    xyz_rgb.reserve(last_raw_points_.size() * 4);
    intensities.reserve(last_raw_points_.size());

    for (const auto& p : last_raw_points_) {
      const float t = Clamp01(static_cast<float>(p.reflectivity) /
                              std::max(1.0f, static_cast<float>(config_.reflectivity_threshold)));
      const uint8_t r = static_cast<uint8_t>(255.0f * t);
      const uint8_t g = static_cast<uint8_t>(180.0f * (1.0f - t));
      const uint8_t b = static_cast<uint8_t>(255.0f * (1.0f - t));
      xyz_rgb.push_back(p.x);
      xyz_rgb.push_back(p.y);
      xyz_rgb.push_back(p.z);
      xyz_rgb.push_back(PackedRgb(r, g, b));
      intensities.push_back(static_cast<float>(p.reflectivity));
    }

    raw_cloud_pub_.publish(
        BuildPointCloud2Msg(xyz_rgb, intensities, stamp, config_.frame_id));
  }

  void PublishCandidateClouds(const ros::Time& stamp) {
    if (!config_.publish_candidate_cloud &&
        !config_.publish_high_reflective_cloud) {
      return;
    }

    std::vector<float> all_xyz_rgb;
    std::vector<float> all_intensity;
    std::vector<float> high_xyz_rgb;
    std::vector<float> high_intensity;

    if (state_ == State::kDiscover) {
      const float rgb = PackedRgb(255, 230, 80);
      for (const auto& timed : discovery_high_points_) {
        const auto& p = timed.point;
        high_xyz_rgb.push_back(p.x);
        high_xyz_rgb.push_back(p.y);
        high_xyz_rgb.push_back(p.z);
        high_xyz_rgb.push_back(rgb);
        high_intensity.push_back(static_cast<float>(p.reflectivity));
      }
    }

    for (size_t c = 0; c < candidates_.size(); ++c) {
      const Rgb8 color = CandidateDisplayColor(c, candidates_[c].id);
      const float rgb = PackedRgb(color);

      for (const auto& timed : candidates_[c].all_points) {
        const auto& p = timed.point;
        all_xyz_rgb.push_back(p.x);
        all_xyz_rgb.push_back(p.y);
        all_xyz_rgb.push_back(p.z);
        all_xyz_rgb.push_back(rgb);
        all_intensity.push_back(static_cast<float>(p.reflectivity));
      }

      for (const auto& timed : candidates_[c].high_points) {
        const auto& p = timed.point;
        high_xyz_rgb.push_back(p.x);
        high_xyz_rgb.push_back(p.y);
        high_xyz_rgb.push_back(p.z);
        high_xyz_rgb.push_back(rgb);
        high_intensity.push_back(static_cast<float>(p.reflectivity));
      }
    }

    if (config_.publish_candidate_cloud) {
      candidate_cloud_pub_.publish(
          BuildPointCloud2Msg(all_xyz_rgb, all_intensity, stamp, config_.frame_id));
    }
    if (config_.publish_high_reflective_cloud) {
      high_cloud_pub_.publish(
          BuildPointCloud2Msg(high_xyz_rgb, high_intensity, stamp, config_.frame_id));
    }
  }

  void PublishTemplateMatchCloud(const ros::Time& stamp) {
    if (!config_.publish_template_match_cloud) {
      return;
    }

    std::vector<float> xyz_rgb;
    std::vector<float> intensities;
    xyz_rgb.reserve(last_template_match_points_.size() * 4);
    intensities.reserve(last_template_match_points_.size());
    const float rgb = confirmed_ ? PackedRgb(kTargetGreen) : PackedRgb(255, 230, 80);
    for (const auto& p : last_template_match_points_) {
      xyz_rgb.push_back(p.x);
      xyz_rgb.push_back(p.y);
      xyz_rgb.push_back(p.z);
      xyz_rgb.push_back(rgb);
      intensities.push_back(static_cast<float>(p.reflectivity));
    }

    template_match_cloud_pub_.publish(
        BuildPointCloud2Msg(xyz_rgb, intensities, stamp, config_.frame_id));
  }

  void PublishMarkers(const ros::Time& stamp) {
    if (!config_.publish_markers) {
      return;
    }

    visualization_msgs::MarkerArray array;

    visualization_msgs::Marker clear;
    clear.header.stamp = stamp;
    clear.header.frame_id = config_.frame_id;
    clear.ns = "task1_reflective_board";
    clear.action = visualization_msgs::Marker::DELETEALL;
    array.markers.push_back(clear);

    for (size_t i = 0; i < candidates_.size(); ++i) {
      const auto& candidate = candidates_[i];
      const Rgb8 color = CandidateDisplayColor(i, candidate.id);
      const bool is_confirmed_target =
          confirmed_ && candidate.id == confirmed_candidate_id_;

      visualization_msgs::Marker roi;
      roi.header.stamp = stamp;
      roi.header.frame_id = config_.frame_id;
      roi.ns = "task1_roi";
      roi.id = candidate.id;
      roi.type = visualization_msgs::Marker::SPHERE;
      roi.action = visualization_msgs::Marker::ADD;
      roi.pose.position.x = candidate.seed_center.x();
      roi.pose.position.y = candidate.seed_center.y();
      roi.pose.position.z = candidate.seed_center.z();
      roi.pose.orientation.w = 1.0;
      roi.scale.x = 2.0 * candidate.roi_radius;
      roi.scale.y = 2.0 * candidate.roi_radius;
      roi.scale.z = 2.0 * candidate.roi_radius;
      roi.color.r = color[0] / 255.0f;
      roi.color.g = color[1] / 255.0f;
      roi.color.b = color[2] / 255.0f;
      roi.color.a = is_confirmed_target ? 0.32f : 0.12f;
      array.markers.push_back(roi);

      const CandidateScore* score = FindScore(candidate.id);
      visualization_msgs::Marker text;
      text.header.stamp = stamp;
      text.header.frame_id = config_.frame_id;
      text.ns = "task1_scores";
      text.id = candidate.id;
      text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::Marker::ADD;
      text.pose.position.x = candidate.seed_center.x();
      text.pose.position.y = candidate.seed_center.y();
      text.pose.position.z = candidate.seed_center.z() + candidate.roi_radius + 0.08f;
      text.pose.orientation.w = 1.0;
      text.scale.z = 0.08;
      text.color.r = 1.0f;
      text.color.g = 1.0f;
      text.color.b = 1.0f;
      text.color.a = 0.95f;
      std::ostringstream oss;
      oss.precision(2);
      oss << "C" << candidate.id;
      if (score != nullptr) {
        oss << " score=" << std::fixed << score->score
            << " tpl=" << score->template_score;
      }
      text.text = oss.str();
      array.markers.push_back(text);
    }

    if (confirmed_ && confirmed_score_.valid_pose) {
      visualization_msgs::Marker axis;
      axis.header.stamp = stamp;
      axis.header.frame_id = config_.frame_id;
      axis.ns = "task1_confirmed";
      axis.id = 1;
      axis.type = visualization_msgs::Marker::ARROW;
      axis.action = visualization_msgs::Marker::ADD;
      axis.pose = confirmed_score_.board.pose;
      axis.scale.x = 0.20;
      axis.scale.y = 0.018;
      axis.scale.z = 0.018;
      axis.color.r = kTargetGreen[0] / 255.0f;
      axis.color.g = kTargetGreen[1] / 255.0f;
      axis.color.b = kTargetGreen[2] / 255.0f;
      axis.color.a = 1.0f;
      array.markers.push_back(axis);
    }

    marker_pub_.publish(array);
  }

  const CandidateScore* FindScore(int candidate_id) const {
    for (const auto& score : latest_scores_) {
      if (score.candidate_id == candidate_id) {
        return &score;
      }
    }
    return nullptr;
  }

  void ResetToDiscover() {
    state_ = State::kDiscover;
    candidates_.clear();
    latest_scores_.clear();
    best_score_ = CandidateScore();
    confirmed_score_ = CandidateScore();
    confirmed_ = false;
    confirmed_candidate_id_ = -1;
    discovery_started_at_ = ros::Time(0);
    accumulation_started_at_ = ros::Time(0);
    accumulation_start_cpu_time_sec_ = 0.0f;
    last_score_stamp_ = ros::Time(0);
    candidate_template_recording_ = false;
    candidate_template_saved_ = false;
    candidate_template_started_at_ = ros::Time(0);
  }

  void PushCapped(const TimedPoint& point,
                  size_t max_points,
                  std::deque<TimedPoint>* points) const {
    if (points == nullptr) {
      return;
    }
    points->push_back(point);
    while (points->size() > max_points) {
      points->pop_front();
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  Config config_;
  TemplateModel template_model_;
  ResourceUsageTracker resource_usage_;

  ros::Subscriber sub_;
  ros::Timer timer_;
  ros::Publisher raw_cloud_pub_;
  ros::Publisher candidate_cloud_pub_;
  ros::Publisher high_cloud_pub_;
  ros::Publisher template_match_cloud_pub_;
  ros::Publisher marker_pub_;
  ros::Publisher target_pose_pub_;
  ros::Publisher best_score_pub_;

  State state_ = State::kDiscover;
  std::deque<TimedPoint> discovery_high_points_;
  std::vector<Candidate> candidates_;
  std::vector<CandidateScore> latest_scores_;
  CandidateScore best_score_;
  CandidateScore confirmed_score_;
  bool confirmed_ = false;
  int confirmed_candidate_id_ = -1;

  std::vector<PointXYZI> last_raw_points_;
  ros::Time last_stamp_;
  size_t last_frame_total_points_ = 0;
  size_t last_frame_valid_points_ = 0;
  size_t last_frame_high_points_ = 0;
  size_t last_frame_roi_hits_ = 0;
  std::vector<PointXYZI> last_template_match_points_;

  int next_candidate_id_ = 0;
  int detection_result_count_ = 0;
  int candidate_template_batch_count_ = 0;
  float accumulation_start_cpu_time_sec_ = 0.0f;
  bool candidate_template_recording_ = false;
  bool candidate_template_saved_ = false;
  ros::Time discovery_started_at_;
  ros::Time accumulation_started_at_;
  ros::Time candidate_template_started_at_;
  ros::Time last_score_stamp_;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "reflective_board_identifier_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  ReflectiveBoardIdentifier identifier(nh, pnh);
  ros::spin();
  return 0;
}
