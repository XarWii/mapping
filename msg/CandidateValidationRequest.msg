# Sent by BPU node to request traditional validation of a candidate.
# Mode: 0 = INIT (first confirmation), 1 = CORRECTION (re-acquire)

Header header
uint32 proposal_id
uint8  mode          # 0=INIT, 1=CORRECTION
geometry_msgs/Point center
float32 roi_radius
float32 accumulation_sec
sensor_msgs/PointCloud2 seed_cloud
