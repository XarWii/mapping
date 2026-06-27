# Decision state for the dynamic reflective-map route only.
Header header
uint32 map_epoch

# 0=SEARCHING, 1=TRACKING, 2=LOST
uint8 state
uint32 candidate_id
geometry_msgs/Point center_map
float32 score
string reason
