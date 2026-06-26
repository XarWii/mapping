# Request one batch BPU evaluation for an exact rolling-map candidate snapshot.
Header header
uint32 map_epoch
uint64 request_id
uint64 candidate_snapshot_id
uint32[] candidate_ids
uint32[] support_revisions

# 1=new candidate, 2=active target disappeared, 3=candidate became scoreable,
# 4=previous request timed out, 5=rejected candidate gained enough points.
uint8 reason
