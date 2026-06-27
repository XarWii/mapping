#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <ros/ros.h>

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

struct Cluster {
  std::vector<size_t> indices;
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
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
  std::vector<PointXYZI> points;
  std::vector<PointXYZI> points_xyz;
};

struct Config {
  std::string input_topic = "/livox/lidar";
  std::string frame_id = "livox_frame";
  std::string output_path = "/tmp/target_board_template.yaml";
  int input_queue_size = 8;

  uint8_t reflectivity_threshold = 250;
  float min_distance_m = 0.1f;
  float max_distance_m = 30.0f;

  double min_accumulation_sec = 5.0;
  double max_recording_sec = 30.0;
  size_t target_template_points = 300;
  float cluster_tolerance_m = 0.35f;
  size_t min_cluster_points = 20;
  size_t max_accumulated_points = 20000;
  size_t max_saved_points = 5000;
  bool auto_shutdown_after_save = false;
  bool save_preview_image = true;
  std::string preview_image_path;
  int preview_image_size_px = 640;
  int preview_image_padding_px = 48;
  int preview_point_radius_px = 2;
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

bool IsValidPoint(const PointXYZI& point, const Config& config) {
  return IsFinite(point.x) && IsFinite(point.y) && IsFinite(point.z) &&
         point.distance >= config.min_distance_m &&
         point.distance <= config.max_distance_m &&
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

std::vector<Cluster> EuclideanCluster(const std::vector<PointXYZI>& points,
                                      float tolerance_m,
                                      size_t min_cluster_points) {
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

    Cluster cluster;
    for (size_t head = 0; head < queue.size(); ++head) {
      const size_t current = queue[head];
      cluster.indices.push_back(current);
      for (size_t j = 0; j < points.size(); ++j) {
        if (!visited[j] &&
            SquaredDistance(points[current], points[j]) <= tolerance_sq) {
          visited[j] = true;
          queue.push_back(j);
        }
      }
    }

    if (cluster.indices.size() >= min_cluster_points) {
      for (size_t idx : cluster.indices) {
        cluster.center += ToEigen(points[idx]);
      }
      cluster.center /= static_cast<float>(cluster.indices.size());
      clusters.push_back(cluster);
    }
  }

  std::sort(clusters.begin(), clusters.end(),
            [](const Cluster& a, const Cluster& b) {
              return a.indices.size() > b.indices.size();
            });
  return clusters;
}

bool EstimatePlaneTemplate(const std::vector<PointXYZI>& points,
                           const std::vector<size_t>& indices,
                           const Config& config,
                           PlaneTemplate* out) {
  if (out == nullptr || indices.size() < 3) {
    return false;
  }

  PlaneTemplate result;
  for (size_t idx : indices) {
    result.origin += ToEigen(points[idx]);
  }
  result.origin /= static_cast<float>(indices.size());

  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (size_t idx : indices) {
    const Eigen::Vector3f d = ToEigen(points[idx]) - result.origin;
    covariance += d * d.transpose();
  }
  covariance /= static_cast<float>(indices.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  result.normal = solver.eigenvectors().col(0).normalized();
  result.x_axis = solver.eigenvectors().col(2).normalized();
  result.x_axis =
      (result.x_axis - result.x_axis.dot(result.normal) * result.normal).normalized();
  result.y_axis = result.normal.cross(result.x_axis).normalized();

  const Eigen::Vector3f to_lidar = -result.origin;
  if (to_lidar.norm() > 1e-4f &&
      result.normal.dot(to_lidar.normalized()) < 0.0f) {
    result.normal = -result.normal;
    result.y_axis = -result.y_axis;
  }

  float squared_plane_error = 0.0f;
  result.min_u = std::numeric_limits<float>::infinity();
  result.max_u = -std::numeric_limits<float>::infinity();
  result.min_v = std::numeric_limits<float>::infinity();
  result.max_v = -std::numeric_limits<float>::infinity();

  std::vector<size_t> saved_indices = indices;
  if (saved_indices.size() > config.max_saved_points) {
    std::vector<size_t> sampled;
    sampled.reserve(config.max_saved_points);
    const double stride =
        static_cast<double>(saved_indices.size()) /
        static_cast<double>(config.max_saved_points);
    for (size_t i = 0; i < config.max_saved_points; ++i) {
      const size_t src =
          std::min(saved_indices.size() - 1,
                   static_cast<size_t>(std::floor(i * stride)));
      sampled.push_back(saved_indices[src]);
    }
    saved_indices.swap(sampled);
  }

  result.points.reserve(saved_indices.size());
  result.points_xyz.reserve(saved_indices.size());
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
    result.points.push_back(projected);
    result.points_xyz.push_back(point);

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

void WriteVector(std::ofstream& out, const Eigen::Vector3f& v) {
  out << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]";
}

std::string DerivePreviewImagePath(const std::string& template_path) {
  const size_t slash = template_path.find_last_of("/\\");
  const size_t dot = template_path.find_last_of('.');
  const bool has_extension =
      dot != std::string::npos && (slash == std::string::npos || dot > slash);
  const std::string base =
      has_extension ? template_path.substr(0, dot) : template_path;
  return base + "_uv.bmp";
}

void WriteLe16(std::ofstream& out, uint16_t value) {
  out.put(static_cast<char>(value & 0xff));
  out.put(static_cast<char>((value >> 8) & 0xff));
}

void WriteLe32(std::ofstream& out, uint32_t value) {
  out.put(static_cast<char>(value & 0xff));
  out.put(static_cast<char>((value >> 8) & 0xff));
  out.put(static_cast<char>((value >> 16) & 0xff));
  out.put(static_cast<char>((value >> 24) & 0xff));
}

void SetPixelBlack(std::vector<uint8_t>* image, int width, int height,
                   int x, int y) {
  if (image == nullptr || x < 0 || x >= width || y < 0 || y >= height) {
    return;
  }
  const size_t offset = static_cast<size_t>(y * width + x) * 3;
  (*image)[offset + 0] = 0;
  (*image)[offset + 1] = 0;
  (*image)[offset + 2] = 0;
}

bool SaveUvPreviewImage(const PlaneTemplate& tpl,
                        const Config& config,
                        const std::string& path) {
  if (tpl.points.empty()) {
    ROS_WARN_STREAM("Skipping template preview image: no local UV points.");
    return false;
  }

  const int size = std::max(128, config.preview_image_size_px);
  const int padding =
      std::max(0, std::min(config.preview_image_padding_px, size / 3));
  const int radius = std::max(1, config.preview_point_radius_px);
  const int drawable = std::max(1, size - 2 * padding);
  const float span_u = std::max(1e-6f, tpl.max_u - tpl.min_u);
  const float span_v = std::max(1e-6f, tpl.max_v - tpl.min_v);
  const float scale =
      std::min(static_cast<float>(drawable) / span_u,
               static_cast<float>(drawable) / span_v);

  std::vector<uint8_t> image(static_cast<size_t>(size) * size * 3, 255);
  for (const auto& point : tpl.points) {
    const int x = static_cast<int>(
        std::round((point.x - tpl.min_u) * scale + padding));
    const int y = static_cast<int>(
        std::round((tpl.max_v - point.y) * scale + padding));
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        if (dx * dx + dy * dy <= radius * radius) {
          SetPixelBlack(&image, size, size, x + dx, y + dy);
        }
      }
    }
  }

  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    ROS_WARN_STREAM("Failed to open template preview image output file: "
                    << path);
    return false;
  }

  const uint32_t row_bytes = static_cast<uint32_t>(size) * 3;
  const uint32_t row_stride = (row_bytes + 3u) & ~3u;
  const uint32_t pixel_bytes = row_stride * static_cast<uint32_t>(size);
  const uint32_t file_size = 14u + 40u + pixel_bytes;

  out.put('B');
  out.put('M');
  WriteLe32(out, file_size);
  WriteLe16(out, 0);
  WriteLe16(out, 0);
  WriteLe32(out, 14u + 40u);

  WriteLe32(out, 40u);
  WriteLe32(out, static_cast<uint32_t>(size));
  WriteLe32(out, static_cast<uint32_t>(size));
  WriteLe16(out, 1);
  WriteLe16(out, 24);
  WriteLe32(out, 0);
  WriteLe32(out, pixel_bytes);
  WriteLe32(out, 2835);
  WriteLe32(out, 2835);
  WriteLe32(out, 0);
  WriteLe32(out, 0);

  std::vector<uint8_t> padding_bytes(row_stride - row_bytes, 0);
  for (int y = size - 1; y >= 0; --y) {
    const uint8_t* row =
        image.data() + static_cast<size_t>(y) * size * 3;
    for (int x = 0; x < size; ++x) {
      const uint8_t r = row[static_cast<size_t>(x) * 3 + 0];
      const uint8_t g = row[static_cast<size_t>(x) * 3 + 1];
      const uint8_t b = row[static_cast<size_t>(x) * 3 + 2];
      out.put(static_cast<char>(b));
      out.put(static_cast<char>(g));
      out.put(static_cast<char>(r));
    }
    if (!padding_bytes.empty()) {
      out.write(reinterpret_cast<const char*>(padding_bytes.data()),
                static_cast<std::streamsize>(padding_bytes.size()));
    }
  }

  return static_cast<bool>(out);
}

bool SaveTemplate(const PlaneTemplate& tpl,
                  const Config& config,
                  const std::string& path) {
  std::ofstream out(path);
  if (!out.is_open()) {
    ROS_ERROR_STREAM("Failed to open template output file: " << path);
    return false;
  }

  out << std::fixed << std::setprecision(6);
  out << "template_version: 1\n";
  out << "type: reflective_board_plane_points\n";
  out << "frame_id: \"" << config.frame_id << "\"\n";
  out << "created_from_topic: \"" << config.input_topic << "\"\n";
  out << "reflectivity_threshold: "
      << static_cast<int>(config.reflectivity_threshold) << "\n";
  out << "min_accumulation_sec: " << config.min_accumulation_sec << "\n";
  out << "max_recording_sec: " << config.max_recording_sec << "\n";
  out << "target_template_points: " << config.target_template_points << "\n";
  out << "cluster_tolerance_m: " << config.cluster_tolerance_m << "\n";
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
  out << "saved_point_count: " << tpl.points.size() << "\n";
  out << "points_xyz:\n";
  for (const auto& point : tpl.points_xyz) {
    out << "  - [" << point.x << ", " << point.y << ", " << point.z << ", "
        << static_cast<int>(point.reflectivity) << "]\n";
  }
  out << "points_local_uvw:\n";
  for (const auto& point : tpl.points) {
    out << "  - [" << point.x << ", " << point.y << ", " << point.z << ", "
        << static_cast<int>(point.reflectivity) << "]\n";
  }
  out << "points_local_uv:\n";
  for (const auto& point : tpl.points) {
    out << "  - [" << point.x << ", " << point.y << ", "
        << static_cast<int>(point.reflectivity) << "]\n";
  }
  return true;
}

class ReflectiveBoardTemplateRecorder {
 public:
  ReflectiveBoardTemplateRecorder(ros::NodeHandle nh, ros::NodeHandle pnh)
      : nh_(nh), pnh_(pnh) {
    LoadConfig();
    sub_ = nh_.subscribe(config_.input_topic,
                        static_cast<uint32_t>(config_.input_queue_size),
                        &ReflectiveBoardTemplateRecorder::Callback, this);
    timer_ = nh_.createTimer(ros::Duration(0.25),
                             &ReflectiveBoardTemplateRecorder::TimerCallback, this);
    ROS_INFO_STREAM("reflective_board_template_recorder_node listening on "
                    << config_.input_topic
                    << ", output=" << config_.output_path
                    << ", min_accumulation=" << config_.min_accumulation_sec
                    << "s, target_points="
                    << config_.target_template_points);
  }

 private:
  void LoadConfig() {
    pnh_.param("input_topic", config_.input_topic, config_.input_topic);
    pnh_.param("template_recorder/input_topic", config_.input_topic,
               config_.input_topic);
    pnh_.param("frame_id", config_.frame_id, config_.frame_id);
    pnh_.param("template_recorder/frame_id", config_.frame_id, config_.frame_id);
    pnh_.param("template_recorder/output_path", config_.output_path,
               config_.output_path);
    pnh_.param("input_queue_size", config_.input_queue_size,
               config_.input_queue_size);
    pnh_.param("template_recorder/input_queue_size", config_.input_queue_size,
               config_.input_queue_size);

    int reflectivity_threshold = config_.reflectivity_threshold;
    pnh_.param("reflectivity_threshold", reflectivity_threshold,
               reflectivity_threshold);
    pnh_.param("template_recorder/reflectivity_threshold", reflectivity_threshold,
               reflectivity_threshold);
    config_.reflectivity_threshold =
        static_cast<uint8_t>(std::max(0, std::min(255, reflectivity_threshold)));

    pnh_.param("min_distance_m", config_.min_distance_m, config_.min_distance_m);
    pnh_.param("max_distance_m", config_.max_distance_m, config_.max_distance_m);
    pnh_.param("template_recorder/min_distance_m", config_.min_distance_m,
               config_.min_distance_m);
    pnh_.param("template_recorder/max_distance_m", config_.max_distance_m,
               config_.max_distance_m);
    pnh_.param("template_recorder/accumulation_sec",
               config_.min_accumulation_sec,
               config_.min_accumulation_sec);
    pnh_.param("template_recorder/min_accumulation_sec",
               config_.min_accumulation_sec,
               config_.min_accumulation_sec);
    pnh_.param("template_recorder/max_recording_sec",
               config_.max_recording_sec,
               config_.max_recording_sec);
    pnh_.param("template_recorder/cluster_tolerance_m",
               config_.cluster_tolerance_m, config_.cluster_tolerance_m);

    int min_cluster_points = static_cast<int>(config_.min_cluster_points);
    int target_template_points =
        static_cast<int>(config_.target_template_points);
    int max_accumulated_points = static_cast<int>(config_.max_accumulated_points);
    int max_saved_points = static_cast<int>(config_.max_saved_points);
    int preview_image_size_px = config_.preview_image_size_px;
    int preview_image_padding_px = config_.preview_image_padding_px;
    int preview_point_radius_px = config_.preview_point_radius_px;
    pnh_.param("template_recorder/target_template_points",
               target_template_points,
               target_template_points);
    pnh_.param("template_recorder/min_cluster_points", min_cluster_points,
               min_cluster_points);
    pnh_.param("template_recorder/max_accumulated_points", max_accumulated_points,
               max_accumulated_points);
    pnh_.param("template_recorder/max_saved_points", max_saved_points,
               max_saved_points);
    pnh_.param("template_recorder/save_preview_image",
               config_.save_preview_image,
               config_.save_preview_image);
    pnh_.param("template_recorder/preview_image_path",
               config_.preview_image_path,
               config_.preview_image_path);
    pnh_.param("template_recorder/preview_image_size_px",
               preview_image_size_px,
               preview_image_size_px);
    pnh_.param("template_recorder/preview_image_padding_px",
               preview_image_padding_px,
               preview_image_padding_px);
    pnh_.param("template_recorder/preview_point_radius_px",
               preview_point_radius_px,
               preview_point_radius_px);
    config_.min_cluster_points =
        static_cast<size_t>(std::max(3, min_cluster_points));
    config_.target_template_points =
        static_cast<size_t>(std::max(min_cluster_points, target_template_points));
    config_.max_accumulated_points =
        static_cast<size_t>(std::max(100, max_accumulated_points));
    config_.max_saved_points =
        static_cast<size_t>(std::max(100, max_saved_points));
    config_.preview_image_size_px = std::max(128, preview_image_size_px);
    config_.preview_image_padding_px =
        std::max(0, std::min(preview_image_padding_px,
                             config_.preview_image_size_px / 3));
    config_.preview_point_radius_px =
        std::max(1, std::min(20, preview_point_radius_px));
    pnh_.param("template_recorder/auto_shutdown_after_save",
               config_.auto_shutdown_after_save,
               config_.auto_shutdown_after_save);
    config_.min_accumulation_sec = std::max(0.1, config_.min_accumulation_sec);
    config_.max_recording_sec =
        std::max(config_.min_accumulation_sec, config_.max_recording_sec);
  }

  void Callback(const livox_ros_driver2::CustomMsg::ConstPtr& msg) {
    if (saved_) {
      return;
    }

    ros::Time stamp = msg->header.stamp;
    if (stamp.isZero()) {
      stamp = ros::Time::now();
    }

    size_t added = 0;
    for (const auto& raw_point : msg->points) {
      const PointXYZI point = ToPoint(raw_point);
      if (!IsValidPoint(point, config_)) {
        continue;
      }
      if (started_at_.isZero()) {
        started_at_ = stamp;
      }
      if (points_.size() < config_.max_accumulated_points) {
        points_.push_back(point);
        ++added;
      }
    }
    last_added_points_ = added;
    last_stamp_ = stamp;
  }

  void TimerCallback(const ros::TimerEvent&) {
    if (saved_) {
      return;
    }

    const ros::Time now = last_stamp_.isZero() ? ros::Time::now() : last_stamp_;
    const double elapsed =
        started_at_.isZero() ? 0.0 : std::max(0.0, (now - started_at_).toSec());
    const size_t largest_cluster_points = EstimateLargestClusterPoints();
    ROS_INFO_STREAM_THROTTLE(
        1.0,
        "Template recorder accumulating high-reflectivity points: total="
        << points_.size() << " last_frame_added=" << last_added_points_
        << " largest_cluster=" << largest_cluster_points
        << "/" << config_.target_template_points
        << " elapsed=" << elapsed
        << "s min=" << config_.min_accumulation_sec
        << "s max=" << config_.max_recording_sec << "s");

    if (started_at_.isZero() || elapsed < config_.min_accumulation_sec) {
      return;
    }
    if (largest_cluster_points < config_.target_template_points &&
        elapsed < config_.max_recording_sec &&
        points_.size() < config_.max_accumulated_points) {
      return;
    }
    ProcessAndSave();
  }

  size_t EstimateLargestClusterPoints() const {
    if (points_.size() < config_.min_cluster_points) {
      return 0;
    }
    const std::vector<Cluster> clusters =
        EuclideanCluster(points_, config_.cluster_tolerance_m,
                         config_.min_cluster_points);
    return clusters.empty() ? 0 : clusters.front().indices.size();
  }

  void ProcessAndSave() {
    if (points_.size() < config_.min_cluster_points) {
      ROS_ERROR_STREAM("Not enough high-reflectivity points to record template: "
                       << points_.size() << "/" << config_.min_cluster_points);
      saved_ = true;
      return;
    }

    std::vector<Cluster> clusters =
        EuclideanCluster(points_, config_.cluster_tolerance_m,
                         config_.min_cluster_points);
    if (clusters.empty()) {
      ROS_ERROR_STREAM("No reflective cluster passed template recorder filters.");
      saved_ = true;
      return;
    }

    const Cluster& target = clusters.front();
    PlaneTemplate tpl;
    if (!EstimatePlaneTemplate(points_, target.indices, config_, &tpl)) {
      ROS_ERROR_STREAM("Failed to estimate target board plane.");
      saved_ = true;
      return;
    }

    if (SaveTemplate(tpl, config_, config_.output_path)) {
      ROS_INFO_STREAM("Saved reflective board template to " << config_.output_path
                      << " using " << target.indices.size()
                      << " high-reflectivity point(s), plane_rms="
                      << tpl.plane_rms_m << "m, bounds u=["
                      << tpl.min_u << ", " << tpl.max_u << "] v=["
                      << tpl.min_v << ", " << tpl.max_v << "]");
      if (config_.save_preview_image) {
        const std::string preview_path =
            config_.preview_image_path.empty()
                ? DerivePreviewImagePath(config_.output_path)
                : config_.preview_image_path;
        if (SaveUvPreviewImage(tpl, config_, preview_path)) {
          ROS_INFO_STREAM("Saved reflective board UV preview image to "
                          << preview_path);
        }
      }
    }
    saved_ = true;
    if (config_.auto_shutdown_after_save) {
      ros::shutdown();
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  Config config_;
  ros::Subscriber sub_;
  ros::Timer timer_;

  std::vector<PointXYZI> points_;
  ros::Time started_at_;
  ros::Time last_stamp_;
  size_t last_added_points_ = 0;
  bool saved_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "reflective_board_template_recorder_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ReflectiveBoardTemplateRecorder recorder(nh, pnh);
  ros::spin();
  return 0;
}
