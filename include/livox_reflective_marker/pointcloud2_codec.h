#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

namespace livox_reflective_marker {
namespace pointcloud2 {

inline const sensor_msgs::PointField* FindField(
    const sensor_msgs::PointCloud2& cloud, const std::string& name) {
  for (const auto& field : cloud.fields) {
    if (field.name == name) return &field;
  }
  return nullptr;
}

inline bool ReadFiniteFloat32(const uint8_t* point_data,
                              const sensor_msgs::PointField& field,
                              float* value) {
  if (point_data == nullptr || value == nullptr ||
      field.datatype != sensor_msgs::PointField::FLOAT32) {
    return false;
  }
  std::memcpy(value, point_data + field.offset, sizeof(float));
  return std::isfinite(*value);
}

inline bool HasCompleteRows(const sensor_msgs::PointCloud2& cloud) {
  return cloud.point_step > 0 &&
         cloud.row_step >= static_cast<size_t>(cloud.width) * cloud.point_step &&
         cloud.data.size() >= static_cast<size_t>(cloud.row_step) * cloud.height;
}

}  // namespace pointcloud2
}  // namespace livox_reflective_marker
