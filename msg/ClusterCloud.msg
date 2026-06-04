# Latest-frame high-reflective points with per-point cluster membership.
# Published by siamese_bpu_infer_node so that target_manager_node can colour
# the cloud according to its own track-state decisions.

Header header
uint32 window_id
sensor_msgs/PointCloud2 cloud              # xyz only — all high-reflective points from the latest frame
int32[] per_point_cluster_id               # parallel to cloud: -1 = orphan, 0..N-1 = valid cluster
geometry_msgs/Point[] cluster_centers      # centroid for each valid cluster (index = cluster id)
