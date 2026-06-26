# One frame-local high-reflectivity observation source for map-guided BPU input.
Header header
uint32 map_epoch
uint64 scan_id
uint64 observation_source_contract_hash
geometry_msgs/Pose lidar_pose_in_map
geometry_msgs/Point map_aabb_min
geometry_msgs/Point map_aabb_max
bool truncated
sensor_msgs/PointCloud2 cloud_lidar
