# One dynamic-map candidate scored by the BPU.
uint32 candidate_id
uint32 support_revision

# 0=SCORED, 1=RAW_UNAVAILABLE, 2=INSUFFICIENT_WINDOWS,
# 3=INSUFFICIENT_POINTS, 4=NO_MATCHING_CLUSTER, 5=BUDGET_DEFERRED,
# 6=TIMEOUT, 7=STALE_VERSION, 8=WINDOW_EXTRACTION_BUDGET_DEFERRED
uint8 status
bool score_valid
uint32 windows_used

float32 score
geometry_msgs/Point center_map
geometry_msgs/Point center_sensor
bool center_sensor_valid
string center_sensor_frame_id
time center_sensor_stamp
uint32 voxel_count
uint32 evidence_count
