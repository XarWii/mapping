#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <livox_ros_driver2/CustomMsg.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

#include <livox_reflective_marker/reflectivity_filter.h>

namespace {

class ReflectiveIntensityFilterNode {
 public:
  ReflectiveIntensityFilterNode() : private_nh_("~") {
    private_nh_.param<std::string>("input_topic", input_topic_, "/livox/lidar");
    private_nh_.param<std::string>("output_topic", output_topic_,
                                   "/reflective/high_reflect_points");
    private_nh_.param("publish_visualization", publish_visualization_, true);
    private_nh_.param<std::string>("visualization_topic", visualization_topic_,
                                   "/reflective/high_reflect_points_visualization");

    int threshold = 160;
    private_nh_.param("reflectivity_threshold", threshold, threshold);
    const int clamped_threshold = std::max(0, std::min(255, threshold));
    if (clamped_threshold != threshold) {
      ROS_WARN("reflectivity_threshold=%d is outside [0, 255]; using %d",
               threshold, clamped_threshold);
    }
    filter_ = livox_reflective_marker::ReflectivityFilter(
        static_cast<uint8_t>(clamped_threshold));

    int queue_size = 1;
    private_nh_.param("queue_size", queue_size, queue_size);
    queue_size = std::max(1, queue_size);

    if (input_topic_ == output_topic_) {
      ROS_FATAL("input_topic and output_topic must differ to avoid a feedback loop");
      throw std::runtime_error("reflective intensity filter topic loop");
    }

    publisher_ = nh_.advertise<livox_ros_driver2::CustomMsg>(output_topic_,
                                                               queue_size, false);
    if (publish_visualization_) {
      visualization_publisher_ = nh_.advertise<sensor_msgs::PointCloud2>(
          visualization_topic_, queue_size, false);
      InitializeVisualizationLayout();
    }
    subscriber_ = nh_.subscribe(input_topic_, queue_size,
                                &ReflectiveIntensityFilterNode::HandleScan, this,
                                ros::TransportHints().tcpNoDelay());

    ROS_INFO("Reflectivity filter: %s -> %s, threshold >= %d, queue=%d",
             input_topic_.c_str(), output_topic_.c_str(), clamped_threshold,
             queue_size);
    if (publish_visualization_) {
      ROS_INFO("Reflectivity visualization: %s", visualization_topic_.c_str());
    }
  }

 private:
  void InitializeVisualizationLayout() {
    visualization_scan_.height = 1;
    visualization_scan_.is_bigendian = false;
    visualization_scan_.is_dense = true;
    visualization_scan_.point_step = 16;

    sensor_msgs::PointField field;
    field.count = 1;
    field.datatype = sensor_msgs::PointField::FLOAT32;

    field.name = "x";
    field.offset = 0;
    visualization_scan_.fields.push_back(field);
    field.name = "y";
    field.offset = 4;
    visualization_scan_.fields.push_back(field);
    field.name = "z";
    field.offset = 8;
    visualization_scan_.fields.push_back(field);
    field.name = "intensity";
    field.offset = 12;
    visualization_scan_.fields.push_back(field);
  }

  void BuildVisualizationScan() {
    visualization_scan_.header = filtered_scan_.header;
    visualization_scan_.width = filtered_scan_.point_num;
    visualization_scan_.row_step =
        visualization_scan_.point_step * visualization_scan_.width;
    visualization_scan_.data.resize(visualization_scan_.row_step);

    for (size_t i = 0; i < filtered_scan_.points.size(); ++i) {
      const livox_ros_driver2::CustomPoint& point = filtered_scan_.points[i];
      uint8_t* destination = visualization_scan_.data.data() +
                             i * visualization_scan_.point_step;
      const float intensity = static_cast<float>(point.reflectivity);
      std::memcpy(destination, &point.x, sizeof(float));
      std::memcpy(destination + 4, &point.y, sizeof(float));
      std::memcpy(destination + 8, &point.z, sizeof(float));
      std::memcpy(destination + 12, &intensity, sizeof(float));
    }
  }

  void HandleScan(const livox_ros_driver2::CustomMsg::ConstPtr& scan) {
    filter_.Filter(*scan, &filtered_scan_);
    publisher_.publish(filtered_scan_);
    if (publish_visualization_) {
      BuildVisualizationScan();
      visualization_publisher_.publish(visualization_scan_);
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  ros::Publisher visualization_publisher_;
  std::string input_topic_;
  std::string output_topic_;
  std::string visualization_topic_;
  bool publish_visualization_ = true;
  livox_reflective_marker::ReflectivityFilter filter_{0};
  livox_ros_driver2::CustomMsg filtered_scan_;
  sensor_msgs::PointCloud2 visualization_scan_;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "reflective_intensity_filter");
  ReflectiveIntensityFilterNode node;
  ros::spin();
  return 0;
}
