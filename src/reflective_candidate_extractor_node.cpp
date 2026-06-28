#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <geometry_msgs/Point.h>
#include <livox_reflective_marker/ReflectiveCandidate.h>
#include <livox_reflective_marker/ReflectiveCandidates.h>
#include <livox_reflective_marker/ReflectiveMapSnapshot.h>
#include <livox_reflective_marker/pointcloud2_codec.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct VoxelKey {
  int32_t x = 0;
  int32_t y = 0;
  int32_t z = 0;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

bool KeyLess(const VoxelKey& a, const VoxelKey& b) {
  if (a.x != b.x) return a.x < b.x;
  if (a.y != b.y) return a.y < b.y;
  return a.z < b.z;
}

uint64_t Mix64(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

struct VoxelKeyHash {
  size_t operator()(const VoxelKey& key) const {
    uint64_t hash = Mix64(static_cast<uint32_t>(key.x));
    hash ^= Mix64(static_cast<uint32_t>(key.y) + 0x9e3779b9U);
    hash ^= Mix64(static_cast<uint32_t>(key.z) + 0x85ebca6bU);
    return static_cast<size_t>(Mix64(hash));
  }
};

struct RawPoint {
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  float intensity = 0.0f;
};

struct FineVoxelRef {
  VoxelKey fine_key;
  VoxelKey mid_key;
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  float intensity = 0.0f;
  uint16_t evidence = 0;
};

struct MidCellEntry {
  VoxelKey key;
  size_t begin = 0;
  size_t end = 0;
  uint32_t evidence_sum = 0;
  uint32_t active_fine_count = 0;
  float max_intensity = 0.0f;
  Eigen::Vector3f min_corner =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
  Eigen::Vector3f max_corner =
      Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
  bool active = false;
};

struct SeedRegion {
  std::vector<uint32_t> mid_indices;
  Eigen::Vector3f min_corner =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
  Eigen::Vector3f max_corner =
      Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
  uint32_t evidence_sum = 0;
  uint32_t active_fine_count = 0;
  bool overflow = false;
};

struct Proposal {
  std::vector<uint32_t> fine_indices;
  Eigen::Vector3f roi_min =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
  Eigen::Vector3f roi_max =
      Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
  Eigen::Vector3f support_min =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
  Eigen::Vector3f support_max =
      Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  uint32_t evidence_sum = 0;
  float max_intensity = 0.0f;
  uint32_t candidate_id = 0;
};

struct CandidateRecord {
  uint32_t id = 0;
  Eigen::Vector3f roi_min = Eigen::Vector3f::Zero();
  Eigen::Vector3f roi_max = Eigen::Vector3f::Zero();
  ros::Time last_seen;
};

struct Stats {
  uint32_t inactive_mid_cells = 0;
  uint32_t seed_regions = 0;
  uint32_t overflow_seed_count = 0;
  uint32_t roi_too_small = 0;
  uint32_t roi_low_evidence = 0;
  uint32_t roi_too_large = 0;
  uint32_t deduped = 0;
  uint32_t overlap_deduped = 0;
};

struct Config {
  std::string map_snapshot_topic = "/reflective/map_snapshot";
  std::string map_frame = "reflective_odom";
  int input_queue_size = 1;
  double extract_snapshot_period_sec = 0.2;
  int max_snapshot_fine_refs = 32768;
  int max_candidates = 64;
  float min_intensity = 160.0f;

  Eigen::Vector3f grid_origin_map = Eigen::Vector3f::Zero();
  float mid_cell_size_m = 0.05f;
  uint16_t evidence_aggregation_cap = 8;
  uint32_t mid_active_evidence_threshold = 2;
  uint32_t mid_active_min_fine_voxels = 1;
  int d_bridge_cells = 2;
  float roi_completion_margin_m = 0.12f;
  uint32_t min_map_support_voxels = 4;
  uint32_t min_map_support_evidence = 4;
  float max_candidate_span_m = 0.85f;
  double keep_time_s = -1.0;
  float candidate_association_iou_threshold = 0.20f;
  float roi_dedup_iou_threshold = 0.85f;
  float roi_dedup_point_overlap_threshold = 0.70f;

  std::string candidates_topic = "/reflective/candidates";
  std::string candidate_cloud_topic = "/reflective/candidate_cloud";
  std::string markers_topic = "/reflective/candidate_markers";
};

float AabbVolume(const Eigen::Vector3f& min_corner,
                 const Eigen::Vector3f& max_corner) {
  const Eigen::Vector3f size = (max_corner - min_corner).cwiseMax(0.0f);
  return size.x() * size.y() * size.z();
}

float AabbIou(const Eigen::Vector3f& a_min, const Eigen::Vector3f& a_max,
              const Eigen::Vector3f& b_min, const Eigen::Vector3f& b_max) {
  const Eigen::Vector3f inter_min = a_min.cwiseMax(b_min);
  const Eigen::Vector3f inter_max = a_max.cwiseMin(b_max);
  const float inter = AabbVolume(inter_min, inter_max);
  if (inter <= 0.0f) return 0.0f;
  const float total = AabbVolume(a_min, a_max) + AabbVolume(b_min, b_max) - inter;
  return total > 0.0f ? inter / total : 0.0f;
}

float SortedIndexOverlapRatio(const std::vector<uint32_t>& a,
                              const std::vector<uint32_t>& b) {
  if (a.empty() || b.empty()) return 0.0f;
  size_t i = 0;
  size_t j = 0;
  size_t intersection = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] == b[j]) {
      ++intersection;
      ++i;
      ++j;
    } else if (a[i] < b[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  const size_t smaller = std::min(a.size(), b.size());
  return smaller > 0 ? static_cast<float>(intersection) /
                           static_cast<float>(smaller)
                     : 0.0f;
}

VoxelKey QuantizeFloor(const Eigen::Vector3f& point,
                       const Eigen::Vector3f& origin,
                       float size) {
  const Eigen::Vector3f scaled = (point - origin) / size;
  return VoxelKey{static_cast<int32_t>(std::floor(scaled.x())),
                  static_cast<int32_t>(std::floor(scaled.y())),
                  static_cast<int32_t>(std::floor(scaled.z()))};
}

Eigen::Vector3f Span(const Eigen::Vector3f& min_corner,
                     const Eigen::Vector3f& max_corner) {
  return (max_corner - min_corner).cwiseMax(0.0f);
}

bool PointInAabb(const Eigen::Vector3f& point,
                 const Eigen::Vector3f& min_corner,
                 const Eigen::Vector3f& max_corner) {
  return (point.array() >= min_corner.array()).all() &&
         (point.array() <= max_corner.array()).all();
}

geometry_msgs::Point ToPointMsg(const Eigen::Vector3f& point) {
  geometry_msgs::Point msg;
  msg.x = point.x();
  msg.y = point.y();
  msg.z = point.z();
  return msg;
}

sensor_msgs::PointCloud2 MakeRawCloud(
    const std_msgs::Header& header, const std::vector<RawPoint>& points,
    const Eigen::Matrix3f& output_from_map,
    const Eigen::Vector3f& output_translation) {
  sensor_msgs::PointCloud2 cloud;
  cloud.header = header;
  cloud.height = 1;
  cloud.width = static_cast<uint32_t>(points.size());
  cloud.is_bigendian = false;
  cloud.is_dense = true;
  cloud.point_step = 16;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(cloud.row_step);

  sensor_msgs::PointField field;
  field.count = 1;
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.name = "x";
  field.offset = 0;
  cloud.fields.push_back(field);
  field.name = "y";
  field.offset = 4;
  cloud.fields.push_back(field);
  field.name = "z";
  field.offset = 8;
  cloud.fields.push_back(field);
  field.name = "intensity";
  field.offset = 12;
  cloud.fields.push_back(field);

  for (size_t i = 0; i < points.size(); ++i) {
    const Eigen::Vector3f position =
        output_from_map * points[i].position + output_translation;
    uint8_t* destination = cloud.data.data() + i * cloud.point_step;
    const float px = position.x();
    const float py = position.y();
    const float pz = position.z();
    std::memcpy(destination, &px, sizeof(float));
    std::memcpy(destination + 4, &py, sizeof(float));
    std::memcpy(destination + 8, &pz, sizeof(float));
    std::memcpy(destination + 12, &points[i].intensity, sizeof(float));
  }
  return cloud;
}

class ReflectiveCandidateExtractor {
 public:
  ReflectiveCandidateExtractor() : private_nh_("~"), config_(LoadConfig()) {
    ValidateConfig();
    map_subscriber_ = nh_.subscribe(config_.map_snapshot_topic,
                                    config_.input_queue_size,
                                    &ReflectiveCandidateExtractor::HandleMap, this,
                                    ros::TransportHints().tcpNoDelay());
    candidates_publisher_ = nh_.advertise<livox_reflective_marker::ReflectiveCandidates>(
        config_.candidates_topic, 1);
    candidate_cloud_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(
        config_.candidate_cloud_topic, 1);
    markers_publisher_ = nh_.advertise<visualization_msgs::MarkerArray>(
        config_.markers_topic, 1);

    ROS_INFO("Reflective candidate extractor phase1: snapshot=%s mid=%.3fm "
             "bridge=%d roi_margin=%.3fm",
             config_.map_snapshot_topic.c_str(),
             static_cast<double>(config_.mid_cell_size_m),
             config_.d_bridge_cells,
             static_cast<double>(config_.roi_completion_margin_m));
  }

 private:
  Config LoadConfig() {
    Config config;
    private_nh_.param("map_snapshot_topic", config.map_snapshot_topic,
                      config.map_snapshot_topic);
    private_nh_.param("map_frame", config.map_frame, config.map_frame);
    private_nh_.param("input_queue_size", config.input_queue_size,
                      config.input_queue_size);
    private_nh_.param("extract_snapshot_period_sec",
                      config.extract_snapshot_period_sec,
                      config.extract_snapshot_period_sec);
    private_nh_.param("max_snapshot_fine_refs", config.max_snapshot_fine_refs,
                      config.max_snapshot_fine_refs);
    private_nh_.param("max_map_points", config.max_snapshot_fine_refs,
                      config.max_snapshot_fine_refs);
    private_nh_.param("max_candidates", config.max_candidates,
                      config.max_candidates);
    private_nh_.param("min_intensity", config.min_intensity,
                      config.min_intensity);
    std::vector<double> origin;
    if (private_nh_.getParam("grid_origin_map", origin) && origin.size() == 3) {
      config.grid_origin_map = Eigen::Vector3f(
          static_cast<float>(origin[0]), static_cast<float>(origin[1]),
          static_cast<float>(origin[2]));
    }
    private_nh_.param("mid_cell_size_m", config.mid_cell_size_m,
                      config.mid_cell_size_m);
    int evidence_cap = config.evidence_aggregation_cap;
    private_nh_.param("evidence_aggregation_cap", evidence_cap, evidence_cap);
    config.evidence_aggregation_cap =
        static_cast<uint16_t>(std::max(1, evidence_cap));
    int mid_evidence = static_cast<int>(config.mid_active_evidence_threshold);
    private_nh_.param("mid_active_evidence_threshold", mid_evidence,
                      mid_evidence);
    config.mid_active_evidence_threshold =
        static_cast<uint32_t>(std::max(1, mid_evidence));
    int mid_fine = static_cast<int>(config.mid_active_min_fine_voxels);
    private_nh_.param("mid_active_min_fine_voxels", mid_fine, mid_fine);
    config.mid_active_min_fine_voxels =
        static_cast<uint32_t>(std::max(1, mid_fine));
    private_nh_.param("d_bridge_cells", config.d_bridge_cells,
                      config.d_bridge_cells);
    private_nh_.param("roi_completion_margin_m",
                      config.roi_completion_margin_m,
                      config.roi_completion_margin_m);
    private_nh_.param("roi_completion_radius_m",
                      config.roi_completion_margin_m,
                      config.roi_completion_margin_m);
    int min_voxels = static_cast<int>(config.min_map_support_voxels);
    private_nh_.param("min_map_support_voxels", min_voxels, min_voxels);
    private_nh_.param("min_unique_voxels", min_voxels, min_voxels);
    config.min_map_support_voxels =
        static_cast<uint32_t>(std::max(1, min_voxels));
    int min_evidence = static_cast<int>(config.min_map_support_evidence);
    private_nh_.param("min_map_support_evidence", min_evidence, min_evidence);
    private_nh_.param("min_component_evidence", min_evidence, min_evidence);
    config.min_map_support_evidence =
        static_cast<uint32_t>(std::max(1, min_evidence));
    private_nh_.param("max_candidate_span_m", config.max_candidate_span_m,
                      config.max_candidate_span_m);
    private_nh_.param("max_3d_diameter_m", config.max_candidate_span_m,
                      config.max_candidate_span_m);
    private_nh_.param("keep_time_s", config.keep_time_s, config.keep_time_s);
    private_nh_.param("candidate_association_iou_threshold",
                      config.candidate_association_iou_threshold,
                      config.candidate_association_iou_threshold);
    private_nh_.param("roi_dedup_iou_threshold", config.roi_dedup_iou_threshold,
                      config.roi_dedup_iou_threshold);
    private_nh_.param("roi_dedup_point_overlap_threshold",
                      config.roi_dedup_point_overlap_threshold,
                      config.roi_dedup_point_overlap_threshold);
    private_nh_.param("candidates_topic", config.candidates_topic,
                      config.candidates_topic);
    private_nh_.param("candidate_cloud_topic", config.candidate_cloud_topic,
                      config.candidate_cloud_topic);
    private_nh_.param("markers_topic", config.markers_topic,
                      config.markers_topic);
    return config;
  }

  void ValidateConfig() const {
    if (config_.input_queue_size <= 0 || config_.max_snapshot_fine_refs <= 0 ||
        config_.max_candidates <= 0 || config_.mid_cell_size_m <= 0.0f ||
        config_.evidence_aggregation_cap == 0 ||
        config_.mid_active_evidence_threshold == 0 ||
        config_.mid_active_min_fine_voxels == 0 ||
        config_.d_bridge_cells < 1 ||
        config_.roi_completion_margin_m < 0.0f ||
        config_.min_map_support_voxels == 0 ||
        config_.min_map_support_evidence == 0 ||
        config_.max_candidate_span_m <= 0.0f ||
        (config_.keep_time_s != -1.0 && config_.keep_time_s <= 0.0) ||
        config_.candidate_association_iou_threshold < 0.0f ||
        config_.candidate_association_iou_threshold > 1.0f ||
        config_.roi_dedup_iou_threshold < 0.0f ||
        config_.roi_dedup_iou_threshold > 1.0f ||
        config_.roi_dedup_point_overlap_threshold < 0.0f ||
        config_.roi_dedup_point_overlap_threshold > 1.0f ||
        config_.extract_snapshot_period_sec < 0.0) {
      throw std::runtime_error("invalid reflective_candidate_extractor parameters");
    }
  }

  bool DecodeMap(const livox_reflective_marker::ReflectiveMapSnapshot& snapshot,
                 std::vector<FineVoxelRef>* refs) const {
    const sensor_msgs::PointCloud2& cloud = snapshot.cloud;
    if (cloud.is_bigendian ||
        !livox_reflective_marker::pointcloud2::HasCompleteRows(cloud)) {
      ROS_WARN_THROTTLE(2.0, "candidate extractor received malformed map cloud");
      return false;
    }
    if (!std::isfinite(snapshot.voxel_size_m) || snapshot.voxel_size_m <= 0.0f) {
      ROS_WARN_THROTTLE(2.0, "candidate extractor received invalid voxel size");
      return false;
    }
    const auto* x = livox_reflective_marker::pointcloud2::FindField(cloud, "x");
    const auto* y = livox_reflective_marker::pointcloud2::FindField(cloud, "y");
    const auto* z = livox_reflective_marker::pointcloud2::FindField(cloud, "z");
    const auto* intensity =
        livox_reflective_marker::pointcloud2::FindField(cloud, "intensity");
    const auto* evidence =
        livox_reflective_marker::pointcloud2::FindField(cloud, "evidence");
    if (!x || !y || !z || !intensity || !evidence) {
      ROS_WARN_THROTTLE(2.0,
                        "candidate extractor needs x/y/z/intensity/evidence fields");
      return false;
    }
    const size_t raw_count = static_cast<size_t>(cloud.width) * cloud.height;
    if (raw_count > static_cast<size_t>(config_.max_snapshot_fine_refs)) {
      ROS_ERROR_THROTTLE(2.0,
                         "candidate extractor snapshot_overflow raw=%zu cap=%d",
                         raw_count, config_.max_snapshot_fine_refs);
      return false;
    }

    refs->clear();
    refs->reserve(raw_count);
    for (uint32_t row = 0; row < cloud.height; ++row) {
      const size_t row_start = static_cast<size_t>(row) * cloud.row_step;
      for (uint32_t col = 0; col < cloud.width; ++col) {
        const uint8_t* source = cloud.data.data() + row_start +
                                static_cast<size_t>(col) * cloud.point_step;
        float px = 0.0f;
        float py = 0.0f;
        float pz = 0.0f;
        float pi = 0.0f;
        float pe = 0.0f;
        if (!livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *x, &px) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *y, &py) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *z, &pz) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *intensity, &pi) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *evidence, &pe)) {
          continue;
        }
        if (pi < config_.min_intensity || pe <= 0.0f) continue;
        FineVoxelRef ref;
        ref.position = Eigen::Vector3f(px, py, pz);
        ref.intensity = pi;
        ref.evidence = static_cast<uint16_t>(std::min(65535.0f, std::floor(pe)));
        ref.fine_key = QuantizeFloor(ref.position, config_.grid_origin_map,
                                     snapshot.voxel_size_m);
        ref.mid_key = QuantizeFloor(ref.position, config_.grid_origin_map,
                                    config_.mid_cell_size_m);
        refs->push_back(ref);
      }
    }
    return true;
  }

  std::vector<MidCellEntry> BuildMidCells(std::vector<FineVoxelRef>* refs,
                                          Stats* stats) const {
    std::sort(refs->begin(), refs->end(),
              [](const FineVoxelRef& a, const FineVoxelRef& b) {
                return KeyLess(a.mid_key, b.mid_key);
              });

    std::vector<MidCellEntry> cells;
    size_t begin = 0;
    while (begin < refs->size()) {
      size_t end = begin + 1;
      while (end < refs->size() &&
             (*refs)[end].mid_key == (*refs)[begin].mid_key) {
        ++end;
      }
      MidCellEntry cell;
      cell.key = (*refs)[begin].mid_key;
      cell.begin = begin;
      cell.end = end;
      for (size_t i = begin; i < end; ++i) {
        const FineVoxelRef& ref = (*refs)[i];
        cell.evidence_sum +=
            std::min<uint16_t>(ref.evidence, config_.evidence_aggregation_cap);
        ++cell.active_fine_count;
        cell.max_intensity = std::max(cell.max_intensity, ref.intensity);
        cell.min_corner = cell.min_corner.cwiseMin(ref.position);
        cell.max_corner = cell.max_corner.cwiseMax(ref.position);
      }
      cell.active = cell.evidence_sum >= config_.mid_active_evidence_threshold &&
                    cell.active_fine_count >= config_.mid_active_min_fine_voxels;
      if (!cell.active && stats) ++stats->inactive_mid_cells;
      cells.push_back(cell);
      begin = end;
    }
    return cells;
  }

  std::vector<SeedRegion> BuildSeedRegions(
      const std::vector<MidCellEntry>& cells,
      const std::unordered_map<VoxelKey, uint32_t, VoxelKeyHash>& cell_index,
      Stats* stats) const {
    std::vector<uint8_t> visited(cells.size(), 0);
    std::vector<uint32_t> queue;
    std::vector<SeedRegion> seeds;
    for (uint32_t seed = 0; seed < cells.size(); ++seed) {
      if (visited[seed] || !cells[seed].active) continue;
      SeedRegion region;
      queue.clear();
      queue.push_back(seed);
      visited[seed] = 1;
      size_t head = 0;
      while (head < queue.size()) {
        const uint32_t index = queue[head++];
        const MidCellEntry& cell = cells[index];
        region.mid_indices.push_back(index);
        region.min_corner = region.min_corner.cwiseMin(cell.min_corner);
        region.max_corner = region.max_corner.cwiseMax(cell.max_corner);
        region.evidence_sum += cell.evidence_sum;
        region.active_fine_count += cell.active_fine_count;
        for (int dx = -config_.d_bridge_cells; dx <= config_.d_bridge_cells; ++dx) {
          for (int dy = -config_.d_bridge_cells; dy <= config_.d_bridge_cells; ++dy) {
            for (int dz = -config_.d_bridge_cells; dz <= config_.d_bridge_cells; ++dz) {
              if (dx == 0 && dy == 0 && dz == 0) continue;
              const VoxelKey neighbor{cell.key.x + dx, cell.key.y + dy,
                                      cell.key.z + dz};
              const auto found = cell_index.find(neighbor);
              if (found == cell_index.end()) continue;
              const uint32_t next = found->second;
              if (visited[next] || !cells[next].active) continue;
              visited[next] = 1;
              queue.push_back(next);
            }
          }
        }
      }
      const Eigen::Vector3f span = Span(region.min_corner, region.max_corner);
      region.overflow = span.maxCoeff() > config_.max_candidate_span_m;
      if (region.overflow && stats) ++stats->overflow_seed_count;
      seeds.push_back(std::move(region));
    }
    if (stats) stats->seed_regions = static_cast<uint32_t>(seeds.size());
    return seeds;
  }

  Proposal CompleteRoi(
      const SeedRegion& seed, const std::vector<FineVoxelRef>& refs,
      const std::vector<MidCellEntry>& cells,
      const std::unordered_map<VoxelKey, uint32_t, VoxelKeyHash>& cell_index) const {
    Proposal proposal;
    proposal.roi_min =
        seed.min_corner - Eigen::Vector3f::Constant(config_.roi_completion_margin_m);
    proposal.roi_max =
        seed.max_corner + Eigen::Vector3f::Constant(config_.roi_completion_margin_m);

    const VoxelKey roi_min_key = QuantizeFloor(proposal.roi_min, config_.grid_origin_map,
                                               config_.mid_cell_size_m);
    const VoxelKey roi_max_key = QuantizeFloor(proposal.roi_max, config_.grid_origin_map,
                                               config_.mid_cell_size_m);
    std::unordered_set<uint32_t> used;
    for (int32_t x = roi_min_key.x; x <= roi_max_key.x; ++x) {
      for (int32_t y = roi_min_key.y; y <= roi_max_key.y; ++y) {
        for (int32_t z = roi_min_key.z; z <= roi_max_key.z; ++z) {
          const auto found = cell_index.find(VoxelKey{x, y, z});
          if (found == cell_index.end()) continue;
          const MidCellEntry& cell = cells[found->second];
          for (size_t i = cell.begin; i < cell.end; ++i) {
            if (!PointInAabb(refs[i].position, proposal.roi_min, proposal.roi_max)) {
              continue;
            }
            const uint32_t ref_index = static_cast<uint32_t>(i);
            if (!used.insert(ref_index).second) continue;
            proposal.fine_indices.push_back(ref_index);
            proposal.support_min = proposal.support_min.cwiseMin(refs[i].position);
            proposal.support_max = proposal.support_max.cwiseMax(refs[i].position);
            proposal.center += refs[i].position;
            proposal.evidence_sum += refs[i].evidence;
            proposal.max_intensity = std::max(proposal.max_intensity, refs[i].intensity);
          }
        }
      }
    }
    if (!proposal.fine_indices.empty()) {
      proposal.center /= static_cast<float>(proposal.fine_indices.size());
      std::sort(proposal.fine_indices.begin(), proposal.fine_indices.end());
    }
    return proposal;
  }

  bool ProposalReady(const Proposal& proposal, Stats* stats) const {
    if (proposal.fine_indices.size() < config_.min_map_support_voxels) {
      if (stats) ++stats->roi_too_small;
      return false;
    }
    if (proposal.evidence_sum < config_.min_map_support_evidence) {
      if (stats) ++stats->roi_low_evidence;
      return false;
    }
    if (Span(proposal.support_min, proposal.support_max).maxCoeff() >
        config_.max_candidate_span_m) {
      if (stats) ++stats->roi_too_large;
      return false;
    }
    return true;
  }

  void DedupProposals(std::vector<Proposal>* proposals, Stats* stats) const {
    std::sort(proposals->begin(), proposals->end(),
              [](const Proposal& a, const Proposal& b) {
                if (a.evidence_sum != b.evidence_sum) {
                  return a.evidence_sum > b.evidence_sum;
                }
                return a.fine_indices.size() > b.fine_indices.size();
              });
    std::vector<Proposal> kept;
    kept.reserve(proposals->size());
    for (auto& proposal : *proposals) {
      bool duplicate = false;
      for (const auto& existing : kept) {
        if (AabbIou(proposal.roi_min, proposal.roi_max,
                   existing.roi_min, existing.roi_max) >=
            config_.roi_dedup_iou_threshold) {
          duplicate = true;
          break;
        }
        if (SortedIndexOverlapRatio(proposal.fine_indices,
                                    existing.fine_indices) >=
            config_.roi_dedup_point_overlap_threshold) {
          duplicate = true;
          if (stats) ++stats->overlap_deduped;
          break;
        }
      }
      if (duplicate) {
        if (stats) ++stats->deduped;
        continue;
      }
      kept.push_back(std::move(proposal));
    }
    *proposals = std::move(kept);
  }

  void AssignCandidateIds(std::vector<Proposal>* proposals,
                          const ros::Time& stamp) {
    PruneRecords(stamp);
    std::unordered_set<uint32_t> assigned_records;
    for (auto& proposal : *proposals) {
      int best_index = -1;
      float best_iou = config_.candidate_association_iou_threshold;
      for (size_t i = 0; i < records_.size(); ++i) {
        if (assigned_records.count(records_[i].id) != 0) continue;
        const float iou = AabbIou(proposal.roi_min, proposal.roi_max,
                                  records_[i].roi_min, records_[i].roi_max);
        if (iou > best_iou) {
          best_iou = iou;
          best_index = static_cast<int>(i);
        }
      }
      if (best_index >= 0) {
        CandidateRecord& record = records_[best_index];
        proposal.candidate_id = record.id;
        record.roi_min = proposal.roi_min;
        record.roi_max = proposal.roi_max;
        record.last_seen = stamp;
        assigned_records.insert(record.id);
      } else {
        CandidateRecord record;
        record.id = next_candidate_id_++;
        record.roi_min = proposal.roi_min;
        record.roi_max = proposal.roi_max;
        record.last_seen = stamp;
        proposal.candidate_id = record.id;
        assigned_records.insert(record.id);
        records_.push_back(record);
      }
    }
  }

  void PruneRecords(const ros::Time& stamp) {
    if (config_.keep_time_s == -1.0) return;
    records_.erase(std::remove_if(records_.begin(), records_.end(),
                                  [&](const CandidateRecord& record) {
                                    return (stamp - record.last_seen).toSec() >
                                           config_.keep_time_s;
                                  }),
                   records_.end());
  }

  void ResetEpochIfNeeded(uint32_t map_epoch) {
    if (!have_epoch_ || map_epoch != map_epoch_) {
      records_.clear();
      next_candidate_id_ = 1;
      map_epoch_ = map_epoch;
      have_epoch_ = true;
      ROS_INFO("candidate extractor entered map epoch %u", map_epoch_);
    }
  }

  static void SetMarkerColor(uint32_t id, std_msgs::ColorRGBA* color) {
    static constexpr float kColors[][3] = {
        {0.95f, 0.33f, 0.24f}, {0.18f, 0.72f, 0.42f},
        {0.20f, 0.55f, 0.94f}, {0.96f, 0.73f, 0.18f},
        {0.82f, 0.28f, 0.72f}, {0.13f, 0.72f, 0.75f}};
    const auto& rgb = kColors[id % (sizeof(kColors) / sizeof(kColors[0]))];
    color->r = rgb[0];
    color->g = rgb[1];
    color->b = rgb[2];
    color->a = 0.70f;
  }

  livox_reflective_marker::ReflectiveCandidate MakeCandidate(
      const Proposal& proposal, const std_msgs::Header& map_header,
      const std_msgs::Header& sensor_header,
      const geometry_msgs::Pose& lidar_pose_in_map,
      const Eigen::Matrix3f& sensor_from_map,
      const Eigen::Vector3f& sensor_translation,
      const std::vector<FineVoxelRef>& refs) const {
    std::vector<RawPoint> points;
    points.reserve(proposal.fine_indices.size());
    for (uint32_t index : proposal.fine_indices) {
      points.push_back(RawPoint{refs[index].position, refs[index].intensity});
    }

    livox_reflective_marker::ReflectiveCandidate candidate;
    candidate.candidate_id = proposal.candidate_id;
    candidate.center_map = ToPointMsg(proposal.center);
    candidate.center_sensor =
        ToPointMsg(sensor_from_map * proposal.center + sensor_translation);
	    candidate.lidar_pose_in_map = lidar_pose_in_map;
	    const Eigen::Vector3f bbox_size = Span(proposal.support_min,
	                                           proposal.support_max);
	    candidate.bbox_size.x = bbox_size.x();
	    candidate.bbox_size.y = bbox_size.y();
	    candidate.bbox_size.z = bbox_size.z();
	    candidate.support_min_map = ToPointMsg(proposal.support_min);
	    candidate.support_max_map = ToPointMsg(proposal.support_max);
	    candidate.roi_min_map = ToPointMsg(proposal.roi_min);
	    candidate.roi_max_map = ToPointMsg(proposal.roi_max);
	    candidate.voxel_count = static_cast<uint32_t>(proposal.fine_indices.size());
	    candidate.evidence_count = proposal.evidence_sum;
	    candidate.support_revision = 1;
	    candidate.max_intensity = proposal.max_intensity;
    candidate.cloud_map = MakeRawCloud(map_header, points,
                                       Eigen::Matrix3f::Identity(),
                                       Eigen::Vector3f::Zero());
    candidate.cloud_sensor = MakeRawCloud(sensor_header, points,
                                          sensor_from_map, sensor_translation);
    return candidate;
  }

  sensor_msgs::PointCloud2 BuildDebugCloud(
      const std_msgs::Header& header,
      const livox_reflective_marker::ReflectiveCandidates& candidates) const {
    size_t total = 0;
    for (const auto& candidate : candidates.candidates) {
      total += static_cast<size_t>(candidate.cloud_map.width) *
               candidate.cloud_map.height;
    }

    sensor_msgs::PointCloud2 cloud;
    cloud.header = header;
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(total);
    cloud.is_bigendian = false;
    cloud.is_dense = true;
    cloud.point_step = 20;
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step);
    sensor_msgs::PointField field;
    field.count = 1;
    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.name = "x";
    field.offset = 0;
    cloud.fields.push_back(field);
    field.name = "y";
    field.offset = 4;
    cloud.fields.push_back(field);
    field.name = "z";
    field.offset = 8;
    cloud.fields.push_back(field);
    field.name = "intensity";
    field.offset = 12;
    cloud.fields.push_back(field);
    field.name = "candidate_id";
    field.offset = 16;
    cloud.fields.push_back(field);

    size_t output = 0;
    for (const auto& candidate : candidates.candidates) {
      const auto* x = livox_reflective_marker::pointcloud2::FindField(candidate.cloud_map, "x");
      const auto* y = livox_reflective_marker::pointcloud2::FindField(candidate.cloud_map, "y");
      const auto* z = livox_reflective_marker::pointcloud2::FindField(candidate.cloud_map, "z");
      const auto* intensity =
          livox_reflective_marker::pointcloud2::FindField(candidate.cloud_map, "intensity");
      if (!x || !y || !z || !intensity) continue;
      const float candidate_id = static_cast<float>(candidate.candidate_id);
      for (uint32_t row = 0; row < candidate.cloud_map.height; ++row) {
        const size_t row_start = static_cast<size_t>(row) * candidate.cloud_map.row_step;
        for (uint32_t col = 0; col < candidate.cloud_map.width; ++col) {
          const uint8_t* source = candidate.cloud_map.data.data() + row_start +
                                  static_cast<size_t>(col) *
                                      candidate.cloud_map.point_step;
          float px = 0.0f;
          float py = 0.0f;
          float pz = 0.0f;
          float pi = 0.0f;
          if (!livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *x, &px) ||
              !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *y, &py) ||
              !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *z, &pz) ||
              !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *intensity, &pi)) {
            continue;
          }
          uint8_t* destination = cloud.data.data() + output * cloud.point_step;
          std::memcpy(destination, &px, sizeof(float));
          std::memcpy(destination + 4, &py, sizeof(float));
          std::memcpy(destination + 8, &pz, sizeof(float));
          std::memcpy(destination + 12, &pi, sizeof(float));
          std::memcpy(destination + 16, &candidate_id, sizeof(float));
          ++output;
        }
      }
    }
    cloud.width = static_cast<uint32_t>(output);
    cloud.row_step = cloud.width * cloud.point_step;
    cloud.data.resize(cloud.row_step);
    return cloud;
  }

  void PublishMarkers(
      const std_msgs::Header& header,
      const livox_reflective_marker::ReflectiveCandidates& candidates,
      const std::vector<SeedRegion>& overflow_regions) {
    visualization_msgs::MarkerArray markers;
    visualization_msgs::Marker clear;
    clear.action = visualization_msgs::Marker::DELETEALL;
    markers.markers.push_back(clear);
    for (const auto& candidate : candidates.candidates) {
      visualization_msgs::Marker marker;
      marker.header = header;
      marker.ns = "reflective_candidate_bbox";
      marker.id = static_cast<int32_t>(candidate.candidate_id);
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position = candidate.center_map;
      marker.pose.orientation.w = 1.0;
      marker.scale.x = std::max(0.03, candidate.bbox_size.x);
      marker.scale.y = std::max(0.03, candidate.bbox_size.y);
      marker.scale.z = std::max(0.03, candidate.bbox_size.z);
      SetMarkerColor(candidate.candidate_id, &marker.color);
      markers.markers.push_back(marker);
    }
    int32_t overflow_id = 1;
    for (const auto& region : overflow_regions) {
      visualization_msgs::Marker marker;
      marker.header = header;
      marker.ns = "reflective_overflow_seed";
      marker.id = overflow_id++;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position = ToPointMsg((region.min_corner + region.max_corner) * 0.5f);
      marker.pose.orientation.w = 1.0;
      const Eigen::Vector3f size = Span(region.min_corner, region.max_corner);
      marker.scale.x = std::max(0.03f, size.x());
      marker.scale.y = std::max(0.03f, size.y());
      marker.scale.z = std::max(0.03f, size.z());
      marker.color.r = 1.0f;
      marker.color.g = 0.05f;
      marker.color.b = 0.05f;
      marker.color.a = 0.22f;
      markers.markers.push_back(marker);
    }
    markers_publisher_.publish(markers);
  }

  void HandleMap(const livox_reflective_marker::ReflectiveMapSnapshot::ConstPtr& snapshot) {
    const ros::Time stamp = snapshot->header.stamp.isZero() ? ros::Time::now()
                                                             : snapshot->header.stamp;
    if (!last_processed_snapshot_stamp_.isZero() &&
        config_.extract_snapshot_period_sec > 0.0 &&
        (stamp - last_processed_snapshot_stamp_).toSec() <
            config_.extract_snapshot_period_sec) {
      ++coalesced_snapshot_count_;
      return;
    }
    last_processed_snapshot_stamp_ = stamp;

    if (!snapshot->header.frame_id.empty() &&
        snapshot->header.frame_id != config_.map_frame) {
      ROS_WARN_THROTTLE(2.0, "map snapshot frame is %s, expected %s",
                        snapshot->header.frame_id.c_str(), config_.map_frame.c_str());
      return;
    }
    const Eigen::Quaterniond q_map_lidar(
        snapshot->lidar_pose_in_map.orientation.w,
        snapshot->lidar_pose_in_map.orientation.x,
        snapshot->lidar_pose_in_map.orientation.y,
        snapshot->lidar_pose_in_map.orientation.z);
    if (snapshot->lidar_frame.empty() || q_map_lidar.norm() < 1e-8) {
      ROS_WARN_THROTTLE(2.0,
                        "candidate extractor received snapshot without valid LiDAR pose");
      return;
    }

    ResetEpochIfNeeded(snapshot->map_epoch);

    std::vector<FineVoxelRef> refs;
    if (!DecodeMap(*snapshot, &refs)) return;

    Stats stats;
    std::vector<MidCellEntry> cells = BuildMidCells(&refs, &stats);
    std::unordered_map<VoxelKey, uint32_t, VoxelKeyHash> cell_index;
    cell_index.reserve(cells.size() * 2 + 1);
    for (uint32_t i = 0; i < cells.size(); ++i) {
      cell_index.emplace(cells[i].key, i);
    }
    std::vector<SeedRegion> seeds = BuildSeedRegions(cells, cell_index, &stats);

    std::vector<Proposal> proposals;
    std::vector<SeedRegion> overflow_regions;
    proposals.reserve(seeds.size());
    for (const auto& seed : seeds) {
      if (seed.overflow) {
        overflow_regions.push_back(seed);
        continue;
      }
      Proposal proposal = CompleteRoi(seed, refs, cells, cell_index);
      if (!ProposalReady(proposal, &stats)) continue;
      proposals.push_back(std::move(proposal));
    }
    DedupProposals(&proposals, &stats);
    if (proposals.size() > static_cast<size_t>(config_.max_candidates)) {
      proposals.resize(config_.max_candidates);
    }
    AssignCandidateIds(&proposals, stamp);

    const Eigen::Matrix3f sensor_from_map =
        q_map_lidar.normalized().toRotationMatrix().transpose().cast<float>();
    const Eigen::Vector3f map_lidar_position(
        static_cast<float>(snapshot->lidar_pose_in_map.position.x),
        static_cast<float>(snapshot->lidar_pose_in_map.position.y),
        static_cast<float>(snapshot->lidar_pose_in_map.position.z));
    const Eigen::Vector3f sensor_translation = -sensor_from_map * map_lidar_position;

    livox_reflective_marker::ReflectiveCandidates output;
	    output.header = snapshot->header;
	    output.header.stamp = stamp;
	    output.map_epoch = map_epoch_;
	    output.candidate_snapshot_id = ++candidate_snapshot_id_;
	    std_msgs::Header sensor_header = output.header;
    sensor_header.frame_id = snapshot->lidar_frame;
    for (const auto& proposal : proposals) {
      output.candidates.push_back(MakeCandidate(
          proposal, output.header, sensor_header, snapshot->lidar_pose_in_map,
          sensor_from_map, sensor_translation, refs));
    }

    candidates_publisher_.publish(output);
    candidate_cloud_publisher_.publish(BuildDebugCloud(output.header, output));
    PublishMarkers(output.header, output, overflow_regions);

    ROS_INFO_THROTTLE(
        2.0,
        "candidate extractor phase1 refs=%zu mid=%zu inactive_mid=%u seeds=%u "
        "overflow=%u proposals=%zu dedup=%u overlap_dedup=%u "
        "small=%u low_ev=%u large=%u "
        "records=%zu coalesced=%u epoch=%u",
        refs.size(), cells.size(), stats.inactive_mid_cells, stats.seed_regions,
        stats.overflow_seed_count, output.candidates.size(), stats.deduped,
        stats.overlap_deduped, stats.roi_too_small, stats.roi_low_evidence,
        stats.roi_too_large,
        records_.size(), coalesced_snapshot_count_, map_epoch_);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  Config config_;
  ros::Subscriber map_subscriber_;
  ros::Publisher candidates_publisher_;
  ros::Publisher candidate_cloud_publisher_;
  ros::Publisher markers_publisher_;
  bool have_epoch_ = false;
	  uint32_t map_epoch_ = 0;
	  uint32_t next_candidate_id_ = 1;
	  uint64_t candidate_snapshot_id_ = 0;
	  std::vector<CandidateRecord> records_;
  ros::Time last_processed_snapshot_stamp_;
  uint32_t coalesced_snapshot_count_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "reflective_candidate_extractor");
  try {
    ReflectiveCandidateExtractor node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("reflective_candidate_extractor initialization failed: %s",
              error.what());
    return 1;
  }
  return 0;
}
