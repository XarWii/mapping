# Bounded rolling-map snapshot consumed by candidate extraction.
# cloud is also available separately as PointCloud2 for RViz inspection.
Header header
uint32 map_epoch
float32 voxel_size_m
# T_header.frame_id_lidar at header.stamp.  This makes the snapshot usable by
# recognition without subscribing to any external odometry topic.
string lidar_frame
geometry_msgs/Pose lidar_pose_in_map
sensor_msgs/PointCloud2 cloud
