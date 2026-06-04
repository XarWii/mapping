# Command from target manager to EKF.
# action: 0=INIT, 1=CORRECT, 2=GATE_UPDATE
Header header
uint8  action
geometry_msgs/PoseStamped pose
sensor_msgs/PointCloud2 init_cloud
