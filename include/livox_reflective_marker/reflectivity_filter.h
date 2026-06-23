#pragma once

#include <cstdint>
#include <vector>

#include <livox_ros_driver2/CustomMsg.h>

namespace livox_reflective_marker {

// Keeps complete Livox point records so downstream deskew still has offset_time.
class ReflectivityFilter {
 public:
  explicit ReflectivityFilter(uint8_t threshold) : threshold_(threshold) {}

  bool Accepts(const livox_ros_driver2::CustomPoint& point) const {
    return point.reflectivity >= threshold_;
  }

  void Filter(const livox_ros_driver2::CustomMsg& input,
              livox_ros_driver2::CustomMsg* output) const {
    output->header = input.header;
    output->timebase = input.timebase;
    output->lidar_id = input.lidar_id;
    output->rsvd = input.rsvd;

    std::vector<livox_ros_driver2::CustomPoint>& points = output->points;
    points.clear();
    if (points.capacity() < input.points.size()) {
      points.reserve(input.points.size());
    }

    for (const livox_ros_driver2::CustomPoint& point : input.points) {
      if (Accepts(point)) {
        points.push_back(point);
      }
    }
    output->point_num = static_cast<uint32_t>(points.size());
  }

 private:
  uint8_t threshold_;
};

}  // namespace livox_reflective_marker
