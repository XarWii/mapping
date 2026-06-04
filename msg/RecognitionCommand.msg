# Dynamic recognition parameter override sent by target_manager_node.
# Semantics: a negative value means "keep the current setting".
# Non-negative values replace the corresponding runtime parameter in
# siamese_bpu_infer_node until the next command arrives.

Header header

# Accumulation window size (number of LiDAR frames).
int32 max_accumulation_frames

# Minimum interval between successive BpuScores publications (seconds).
float64 publish_interval_sec
