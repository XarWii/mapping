# Sent by traditional validator in response to a validation request.

Header header
uint32 proposal_id
uint8  mode          # mirrored from request
bool   accepted
float32 score
float32 margin
string reason
geometry_msgs/PoseStamped pose
sensor_msgs/PointCloud2 roi_cloud
