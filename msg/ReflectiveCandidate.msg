# One bounded high-reflectivity region extracted from a rolling map snapshot.
uint32 candidate_id
geometry_msgs/Point center_map
geometry_msgs/Point center_sensor
# LiDAR pose associated with this exact candidate snapshot, expressed in the
# map frame.  It lets downstream geometry choose a consistent board normal.
geometry_msgs/Pose lidar_pose_in_map
geometry_msgs/Vector3 bbox_size
geometry_msgs/Point support_min_map
geometry_msgs/Point support_max_map
geometry_msgs/Point roi_min_map
geometry_msgs/Point roi_max_map
uint32 voxel_count
uint32 evidence_count
uint32 support_revision
float32 max_intensity
# Map cloud has x/y/z/intensity/evidence.  In the dynamic-map route,
# cloud_sensor is retained for compatibility/debug only and is not BPU input.
sensor_msgs/PointCloud2 cloud_map
sensor_msgs/PointCloud2 cloud_sensor
