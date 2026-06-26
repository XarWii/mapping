# Geometry output for the dynamic reflective-map route.
Header header
uint32 map_epoch
uint32 candidate_id

# 0=VALID, 1=STALE, 2=LOST, 3=INVALID_GEOMETRY
uint8 status
geometry_msgs/Pose pose
float32 plane_rms_m
uint32 point_count
float32 inplane_eigen_ratio
bool yaw_observable
string reason
