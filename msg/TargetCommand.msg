# Command from target manager to EKF.
# action: 0=INIT (first handoff), 1=CORRECT (re-acquire after audit)
Header header
uint8  action
geometry_msgs/PoseStamped pose
sensor_msgs/PointCloud2 init_cloud
