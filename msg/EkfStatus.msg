# EKF feedback to target manager.
# state: 0=TRACKING, 1=LOST
Header header
uint8  state
geometry_msgs/PoseStamped current_pose
