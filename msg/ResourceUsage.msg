# Resource usage metrics for a monitored ROS node.
# Published periodically by the resource_monitor_node.

Header header

# Monitored node name
string node_name

# --- CPU ---
# CPU usage percentage (0.0 ~ 100.0 * num_cores).
# Computed from /proc/[pid]/stat utime+stime delta over the sampling interval.
float32 cpu_percent

# Total CPU time (user + kernel) in seconds consumed by the process.
float32 cpu_time_sec

# --- Memory ---
# Resident Set Size (physical memory) in kilobytes.
# From /proc/[pid]/status VmRSS.
int64 memory_rss_kb

# Virtual memory size in kilobytes.
# From /proc/[pid]/status VmSize.
int64 memory_vm_kb

# --- Process info ---
# PID of the monitored node (0 if not found).
int32 pid

# Whether the monitored node process is currently alive.
bool alive
