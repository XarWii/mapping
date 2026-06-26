// ---------------------------------------------------------------------------
// resource_monitor_node
//
// Periodically reads /proc/<pid>/stat, /proc/<pid>/status and
// /proc/<pid>/smaps_rollup for a target ROS node and publishes
// livox_reflective_marker/ResourceUsage messages on
//   /resource_monitor_node/resource_usage
//
// The target node PID is first queried through ROS XMLRPC, then falls back to
// scanning /proc for a process whose comm field matches the configured node_name.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include <ros/master.h>
#include <ros/ros.h>
#include <ros/this_node.h>
#include <std_msgs/Header.h>
#include <xmlrpcpp/XmlRpcClient.h>
#include <xmlrpcpp/XmlRpcValue.h>

#include <livox_reflective_marker/ResourceUsage.h>

namespace {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct MonitorConfig {
  std::string target_node_name = "reflective_board_identifier_node";
  std::string target_node_names;
  double publish_rate_hz = 2.0;          // how often to sample & publish
  double cpu_smoothing_alpha = 0.5f;     // EMA alpha for CPU percent (0..1)
  bool aggregate = false;                // sum all target_node_names
  bool ignore_missing_targets = true;
  std::string memory_limit_metric = "pss";

  // Target thresholds for warnings
  double memory_target_mb = 50.0;
  double cpu_target_percent = 50.0;      // target max CPU usage in percent (half a core)
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
struct TargetState {
  explicit TargetState(std::string name_in = {}) : name(std::move(name_in)) {}

  std::string name;

  // Cached PID (re-discovered on failure)
  int pid = -1;

  // Previous CPU ticks for delta computation
  unsigned long long prev_utime = 0;
  unsigned long long prev_stime = 0;
  unsigned long long prev_total_jiffies = 0;  // system-wide jiffies at prev sample
  bool have_prev = false;

  // Smoothed CPU percent
  float cpu_percent_smoothed = 0.0f;
};

struct MonitorState {
  MonitorConfig config;

  std::vector<TargetState> targets;
  TargetState self_target;

  ros::Publisher pub;
  ros::Publisher component_pub;
};

struct SmapsRollup {
  int64_t pss_kb = 0;
  int64_t pss_anon_kb = 0;
  int64_t pss_file_kb = 0;
  int64_t pss_shmem_kb = 0;
  int64_t uss_kb = 0;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Trim whitespace (in-place).
inline void Trim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
      [](unsigned char ch) { return !std::isspace(ch); }));
  s.erase(std::find_if(s.rbegin(), s.rend(),
      [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

std::vector<std::string> SplitTargetNames(const std::string& names) {
  std::vector<std::string> out;
  std::stringstream ss(names);
  std::string item;
  while (std::getline(ss, item, ',')) {
    Trim(item);
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

// Read the first line of a file.  Returns empty string on error.
std::string ReadFirstLine(const std::string& path) {
  std::ifstream f(path);
  if (!f.is_open()) return {};
  std::string line;
  std::getline(f, line);
  return line;
}

std::string NormalizeNodeName(const std::string& node_name) {
  if (node_name.empty() || node_name.front() == '/') {
    return node_name;
  }
  return "/" + node_name;
}

bool ParseHttpUri(const std::string& uri, std::string* host, int* port,
                  std::string* path) {
  if (host == nullptr || port == nullptr || path == nullptr) {
    return false;
  }
  constexpr char kPrefix[] = "http://";
  if (uri.compare(0, sizeof(kPrefix) - 1, kPrefix) != 0) {
    return false;
  }
  const size_t host_start = sizeof(kPrefix) - 1;
  const size_t colon = uri.find(':', host_start);
  if (colon == std::string::npos) {
    return false;
  }
  const size_t path_start = uri.find('/', colon + 1);
  const std::string port_str =
      uri.substr(colon + 1, path_start == std::string::npos
                                ? std::string::npos
                                : path_start - colon - 1);
  try {
    *port = std::stoi(port_str);
  } catch (...) {
    return false;
  }
  *host = uri.substr(host_start, colon - host_start);
  *path = path_start == std::string::npos ? "/" : uri.substr(path_start);
  return !host->empty() && *port > 0;
}

bool LookupPidByRosNode(const std::string& target_node_name, int* pid) {
  if (pid == nullptr) {
    return false;
  }

  XmlRpc::XmlRpcValue lookup_request;
  lookup_request[0] = ros::this_node::getName();
  lookup_request[1] = NormalizeNodeName(target_node_name);

  XmlRpc::XmlRpcValue lookup_response;
  XmlRpc::XmlRpcValue lookup_payload;
  if (!ros::master::execute("lookupNode", lookup_request, lookup_response,
                            lookup_payload, false)) {
    return false;
  }
  if (lookup_payload.getType() != XmlRpc::XmlRpcValue::TypeString) {
    return false;
  }

  std::string host;
  int port = 0;
  std::string path;
  if (!ParseHttpUri(static_cast<std::string>(lookup_payload), &host, &port,
                    &path)) {
    return false;
  }

  XmlRpc::XmlRpcClient client(host.c_str(), port, path.c_str());
  XmlRpc::XmlRpcValue pid_request;
  XmlRpc::XmlRpcValue pid_response;
  pid_request[0] = ros::this_node::getName();
  if (!client.execute("getPid", pid_request, pid_response)) {
    return false;
  }
  if (pid_response.getType() != XmlRpc::XmlRpcValue::TypeArray ||
      pid_response.size() < 3 ||
      static_cast<int>(pid_response[0]) != 1 ||
      pid_response[2].getType() != XmlRpc::XmlRpcValue::TypeInt) {
    return false;
  }

  *pid = static_cast<int>(pid_response[2]);
  return *pid > 0;
}

// ---------------------------------------------------------------------------
// Find PID by process name (comm field in /proc/<pid>/stat).
// The kernel truncates comm to 15 characters (TASK_COMM_LEN),
// so we compare only the first 15 characters of both strings.
// Returns -1 if not found.
// ---------------------------------------------------------------------------
int FindPidByName(const std::string& target_name) {
  // Kernel truncates comm to 15 chars (TASK_COMM_LEN = 16, but 15 usable).
  const std::string target_prefix = target_name.substr(0, 15);

  // Iterate /proc/[0-9]+/stat
  for (int candidate = 0; candidate < 100000; ++candidate) {
    std::ostringstream oss;
    oss << "/proc/" << candidate << "/stat";
    std::string line = ReadFirstLine(oss.str());
    if (line.empty()) continue;

    // Format: pid (comm) state ...
    // comm is in parentheses and may contain spaces/parens itself.
    // Find the last ')' which closes comm.
    auto close_paren = line.rfind(')');
    if (close_paren == std::string::npos) continue;

    // comm is between the first '(' and the last ')'.
    auto open_paren = line.find('(');
    if (open_paren == std::string::npos) continue;

    std::string comm = line.substr(open_paren + 1, close_paren - open_paren - 1);
    // Compare only the first 15 chars (kernel truncation length)
    if (comm.size() >= 15) {
      if (comm.compare(0, 15, target_prefix) == 0) return candidate;
    } else {
      if (comm == target_name) return candidate;
    }
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Read /proc/<pid>/stat fields.
// Returns false on error.
// ---------------------------------------------------------------------------
bool ReadProcStat(int pid,
                  unsigned long long& utime,
                  unsigned long long& stime) {
  std::ostringstream oss;
  oss << "/proc/" << pid << "/stat";
  std::string line = ReadFirstLine(oss.str());
  if (line.empty()) return false;

  // Fields after comm are space-separated.
  // Find the closing ')' of comm, then skip the space after it.
  auto close_paren = line.rfind(')');
  if (close_paren == std::string::npos) return false;

  std::string rest = line.substr(close_paren + 2);  // skip ") "

  std::istringstream iss(rest);
  // Fields after comm (0-indexed from state):
  //   0: state
  //   1: ppid
  //   2: pgrp
  //   3: session
  //   4: tty_nr
  //   5: tpgid
  //   6: flags
  //   7: minflt
  //   8: cminflt
  //   9: majflt
  //  10: cmajflt
  //  11: utime  (clock ticks)
  //  12: stime  (clock ticks)
  std::string token;
  int field = 0;
  while (iss >> token) {
    if (field == 11) {
      utime = std::stoull(token);
    } else if (field == 12) {
      stime = std::stoull(token);
      return true;
    }
    ++field;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Read /proc/<pid>/status for memory info.
// Returns false on error.
// ---------------------------------------------------------------------------
bool ReadProcStatus(int pid,
                    int64_t& vm_rss_kb,
                    int64_t& vm_size_kb) {
  std::ostringstream oss;
  oss << "/proc/" << pid << "/status";
  std::ifstream f(oss.str());
  if (!f.is_open()) return false;

  vm_rss_kb = 0;
  vm_size_kb = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 6, "VmRSS:") == 0) {
      std::string val = line.substr(6);
      Trim(val);
      // Value is like "12345 kB"
      auto space = val.find(' ');
      if (space != std::string::npos) val = val.substr(0, space);
      vm_rss_kb = std::stoll(val);
    } else if (line.compare(0, 7, "VmSize:") == 0) {
      std::string val = line.substr(7);
      Trim(val);
      auto space = val.find(' ');
      if (space != std::string::npos) val = val.substr(0, space);
      vm_size_kb = std::stoll(val);
    }
  }
  return true;
}

bool ReadKbValue(const std::string& line, const std::string& key,
                 int64_t* value) {
  if (value == nullptr || line.compare(0, key.size(), key) != 0) {
    return false;
  }
  std::istringstream input(line.substr(key.size()));
  int64_t parsed = 0;
  std::string units;
  if (!(input >> parsed >> units) || units != "kB") return false;
  *value = parsed;
  return true;
}

// PSS apportions shared mappings among their users. Summing PSS across only
// the configured algorithm nodes avoids charging shared ROS/BPU libraries in
// full to every process, while still retaining each node's fair share.
bool ReadProcSmapsRollup(int pid, SmapsRollup* rollup) {
  if (rollup == nullptr) return false;
  std::ostringstream oss;
  oss << "/proc/" << pid << "/smaps_rollup";
  std::ifstream file(oss.str());
  if (!file.is_open()) return false;

  *rollup = SmapsRollup{};
  bool have_pss = false;
  std::string line;
  while (std::getline(file, line)) {
    if (ReadKbValue(line, "Pss:", &rollup->pss_kb)) {
      have_pss = true;
    } else if (ReadKbValue(line, "Pss_Anon:", &rollup->pss_anon_kb) ||
               ReadKbValue(line, "Pss_File:", &rollup->pss_file_kb) ||
               ReadKbValue(line, "Pss_Shmem:", &rollup->pss_shmem_kb)) {
      continue;
    } else {
      int64_t private_kb = 0;
      if (ReadKbValue(line, "Private_Clean:", &private_kb) ||
          ReadKbValue(line, "Private_Dirty:", &private_kb) ||
          ReadKbValue(line, "Private_Hugetlb:", &private_kb)) {
        rollup->uss_kb += private_kb;
      }
    }
  }
  return have_pss;
}

// ---------------------------------------------------------------------------
// Read system-wide /proc/stat to get total jiffies.
// ---------------------------------------------------------------------------
unsigned long long ReadSystemJiffies() {
  std::string line = ReadFirstLine("/proc/stat");
  if (line.empty()) return 0;

  // First line: "cpu  user nice system idle iowait irq softirq steal ..."
  std::istringstream iss(line);
  std::string cpu_label;
  iss >> cpu_label;  // "cpu"
  unsigned long long total = 0;
  unsigned long long val;
  while (iss >> val) {
    total += val;
  }
  return total;
}

// ---------------------------------------------------------------------------
// Main sampling + publishing logic
// ---------------------------------------------------------------------------
bool SampleTarget(TargetState* target, const MonitorConfig& config,
                  livox_reflective_marker::ResourceUsage* msg) {
  if (target == nullptr || msg == nullptr) return false;

  // Re-discover PID if lost
  if (target->pid < 0) {
    if (!LookupPidByRosNode(target->name, &target->pid)) {
      target->pid = FindPidByName(target->name);
    }
    if (target->pid < 0) {
      msg->header.stamp = ros::Time::now();
      msg->header.frame_id = "resource_monitor";
      msg->node_name = target->name;
      msg->cpu_percent = 0.0f;
      msg->cpu_time_sec = 0.0f;
      msg->memory_rss_kb = 0;
      msg->memory_pss_kb = 0;
      msg->memory_pss_anon_kb = 0;
      msg->memory_pss_file_kb = 0;
      msg->memory_pss_shmem_kb = 0;
      msg->memory_uss_kb = 0;
      msg->memory_pss_available = false;
      msg->memory_vm_kb = 0;
      msg->pid = -1;
      msg->alive = false;
      return false;
    }
    ROS_INFO("resource_monitor: found '%s' with PID %d",
             target->name.c_str(), target->pid);
    target->have_prev = false;
  }

  // Read memory
  int64_t rss_kb = 0, vm_kb = 0;
  if (!ReadProcStatus(target->pid, rss_kb, vm_kb)) {
    ROS_WARN_THROTTLE(5.0,
      "resource_monitor: failed to read /proc/%d/status — process may have died",
      target->pid);
    target->pid = -1;
    return false;
  }
  SmapsRollup rollup;
  const bool pss_available = ReadProcSmapsRollup(target->pid, &rollup);
  if (!pss_available) {
    ROS_WARN_THROTTLE(5.0,
        "resource_monitor: failed to read /proc/%d/smaps_rollup; "
        "aggregate memory falls back to RSS",
        target->pid);
  }

  // Read CPU ticks
  unsigned long long utime = 0, stime = 0;
  if (!ReadProcStat(target->pid, utime, stime)) {
    ROS_WARN_THROTTLE(5.0,
      "resource_monitor: failed to read /proc/%d/stat — process may have died",
      target->pid);
    target->pid = -1;
    return false;
  }

  // Compute CPU percentage
  float cpu_percent = 0.0f;
  if (target->have_prev) {
    unsigned long long delta_proc = (utime - target->prev_utime) +
                                    (stime - target->prev_stime);
    unsigned long long delta_sys =
        ReadSystemJiffies() - target->prev_total_jiffies;

    if (delta_sys > 0) {
      const long online_cores = std::max<long>(1, sysconf(_SC_NPROCESSORS_ONLN));
      // Match top/htop process CPU semantics: 100% means one fully used core.
      // /proc/stat totals all cores, so multiply the ratio by online_cores.
      cpu_percent = (static_cast<float>(delta_proc) /
                     static_cast<float>(delta_sys)) *
                    100.0f * static_cast<float>(online_cores);
      // Clamp to reasonable range
      cpu_percent = std::max(0.0f, std::min(cpu_percent, 10000.0f));
    }
  }

  // EMA smoothing
  const float alpha = static_cast<float>(config.cpu_smoothing_alpha);
  if (target->have_prev) {
    target->cpu_percent_smoothed =
        alpha * cpu_percent + (1.0f - alpha) * target->cpu_percent_smoothed;
  } else {
    target->cpu_percent_smoothed = cpu_percent;
  }

  target->prev_utime = utime;
  target->prev_stime = stime;
  target->prev_total_jiffies = ReadSystemJiffies();
  target->have_prev = true;

  // Populate message
  msg->header.stamp = ros::Time::now();
  msg->header.frame_id = "resource_monitor";
  msg->node_name = target->name;
  msg->cpu_percent = target->cpu_percent_smoothed;
  msg->cpu_time_sec = static_cast<float>(utime + stime) /
                      static_cast<float>(sysconf(_SC_CLK_TCK));
  msg->memory_rss_kb = rss_kb;
  msg->memory_pss_kb = rollup.pss_kb;
  msg->memory_pss_anon_kb = rollup.pss_anon_kb;
  msg->memory_pss_file_kb = rollup.pss_file_kb;
  msg->memory_pss_shmem_kb = rollup.pss_shmem_kb;
  msg->memory_uss_kb = rollup.uss_kb;
  msg->memory_pss_available = pss_available;
  msg->memory_vm_kb = vm_kb;
  msg->pid = target->pid;
  msg->alive = true;
  return true;
}

void LogResourceMessage(const livox_reflective_marker::ResourceUsage& msg,
                        const MonitorConfig& config, bool warn) {
  const double rss_mb = static_cast<double>(msg.memory_rss_kb) / 1024.0;
  const double pss_mb = static_cast<double>(msg.memory_pss_kb) / 1024.0;
  const bool use_pss = config.memory_limit_metric == "pss" &&
                       msg.memory_pss_available;
  const double accounted_mb = use_pss ? pss_mb : rss_mb;
  const double cpu_val = static_cast<double>(msg.cpu_percent);

  const double mem_target = config.memory_target_mb;
  const double cpu_target = config.cpu_target_percent;

  const char* mem_ok = (accounted_mb <= mem_target) ? "OK" : "EXCEEDED";
  const char* cpu_ok = (cpu_val <= cpu_target) ? "OK" : "EXCEEDED";

  ROS_INFO("resource: node=%s pid=%d cpu=%.1f%%/%0.f%%[%s] %s=%.1fMB/%0.fMB[%s] pss=%.1fMB uss=%.1fMB rss=%.1fMB vm=%ldkB",
           msg.node_name.c_str(), msg.pid,
           cpu_val, cpu_target, cpu_ok,
           use_pss ? "pss" : "rss", accounted_mb, mem_target, mem_ok,
           pss_mb, static_cast<double>(msg.memory_uss_kb) / 1024.0, rss_mb,
           static_cast<long>(msg.memory_vm_kb));

  if (!warn) return;

  if (accounted_mb > mem_target) {
    ROS_WARN_THROTTLE(2.0,
      "resource_monitor: %s accounted memory %.1f MB exceeds target %.0f MB",
      msg.node_name.c_str(), accounted_mb, mem_target);
  }
  if (cpu_val > cpu_target) {
    ROS_WARN_THROTTLE(2.0,
      "resource_monitor: %s CPU %.1f%% exceeds target %.0f%%",
      msg.node_name.c_str(), cpu_val, cpu_target);
  }
}

// ---------------------------------------------------------------------------
// Main sampling + publishing logic
// ---------------------------------------------------------------------------
void SampleAndPublish(MonitorState* state) {
  if (state == nullptr) return;

  if (!state->config.aggregate) {
    livox_reflective_marker::ResourceUsage msg;
    if (!SampleTarget(&state->targets.front(), state->config, &msg)) {
      state->pub.publish(msg);
      ROS_INFO("resource: node=%s pid=not_found alive=false",
               msg.node_name.c_str());
      return;
    }
    state->pub.publish(msg);
    LogResourceMessage(msg, state->config, true);
    return;
  }

  livox_reflective_marker::ResourceUsage total;
  total.header.stamp = ros::Time::now();
  total.header.frame_id = "resource_monitor";
  total.node_name = "TOTAL";
  total.cpu_percent = 0.0f;
  total.cpu_time_sec = 0.0f;
  total.memory_rss_kb = 0;
  total.memory_pss_kb = 0;
  total.memory_pss_anon_kb = 0;
  total.memory_pss_file_kb = 0;
  total.memory_pss_shmem_kb = 0;
  total.memory_uss_kb = 0;
  total.memory_pss_available = true;
  total.memory_vm_kb = 0;
  total.pid = -1;
  total.alive = true;

  int alive_count = 0;
  int expected_count = static_cast<int>(state->targets.size());
  for (auto& target : state->targets) {
    livox_reflective_marker::ResourceUsage msg;
    if (!SampleTarget(&target, state->config, &msg)) {
      total.alive = false;
      if (!state->config.ignore_missing_targets) {
        ROS_INFO("resource_total: node=%s pid=not_found alive=false",
                 target.name.c_str());
      }
      continue;
    }
    state->component_pub.publish(msg);
    ++alive_count;
    total.cpu_percent += msg.cpu_percent;
    total.cpu_time_sec += msg.cpu_time_sec;
    total.memory_rss_kb += msg.memory_rss_kb;
    total.memory_pss_kb += msg.memory_pss_kb;
    total.memory_pss_anon_kb += msg.memory_pss_anon_kb;
    total.memory_pss_file_kb += msg.memory_pss_file_kb;
    total.memory_pss_shmem_kb += msg.memory_pss_shmem_kb;
    total.memory_uss_kb += msg.memory_uss_kb;
    total.memory_pss_available = total.memory_pss_available &&
                                 msg.memory_pss_available;
    total.memory_vm_kb += msg.memory_vm_kb;
  }

  // The monitor is not part of the algorithm budget, but publishing its own
  // attributed memory makes the measurement overhead explicit.
  livox_reflective_marker::ResourceUsage monitor_overhead;
  const bool monitor_overhead_available =
      SampleTarget(&state->self_target, state->config, &monitor_overhead);
  if (monitor_overhead_available) {
    state->component_pub.publish(monitor_overhead);
  }

  total.alive = state->config.ignore_missing_targets ? alive_count > 0
                                                     : alive_count == expected_count;
  state->pub.publish(total);

  const double rss_mb = static_cast<double>(total.memory_rss_kb) / 1024.0;
  const double pss_mb = static_cast<double>(total.memory_pss_kb) / 1024.0;
  const bool use_pss = state->config.memory_limit_metric == "pss" &&
                       total.memory_pss_available;
  const double accounted_mb = use_pss ? pss_mb : rss_mb;
  const double monitor_pss_mb = monitor_overhead_available &&
                                  monitor_overhead.memory_pss_available
      ? static_cast<double>(monitor_overhead.memory_pss_kb) / 1024.0
      : 0.0;
  const double cpu_val = static_cast<double>(total.cpu_percent);
  const char* mem_ok =
      (accounted_mb <= state->config.memory_target_mb) ? "OK" : "EXCEEDED";
  const char* cpu_ok =
      (cpu_val <= state->config.cpu_target_percent) ? "OK" : "EXCEEDED";
  ROS_INFO("resource_total: nodes=%d/%d cpu=%.1f%%/%0.f%%[%s] %s=%.1fMB/%0.fMB[%s] pss=%.1fMB uss=%.1fMB rss_sum=%.1fMB monitor_pss=%.1fMB vm_sum=%ldkB",
           alive_count, expected_count,
           cpu_val, state->config.cpu_target_percent, cpu_ok,
           use_pss ? "pss" : "rss_sum", accounted_mb,
           state->config.memory_target_mb, mem_ok,
           pss_mb, static_cast<double>(total.memory_uss_kb) / 1024.0, rss_mb,
           monitor_pss_mb,
           static_cast<long>(total.memory_vm_kb));
}

}  // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  ros::init(argc, argv, "resource_monitor_node");
  ros::NodeHandle pnh("~");

  MonitorState state;
  state.self_target.name = ros::this_node::getName();
  state.self_target.pid = static_cast<int>(getpid());

  // Load config
  pnh.param<std::string>("target_node_name",
      state.config.target_node_name, state.config.target_node_name);
  pnh.param<std::string>("target_node_names",
      state.config.target_node_names, state.config.target_node_names);
  pnh.param("publish_rate_hz", state.config.publish_rate_hz,
      state.config.publish_rate_hz);
  pnh.param("cpu_smoothing_alpha", state.config.cpu_smoothing_alpha,
      state.config.cpu_smoothing_alpha);
  pnh.param("aggregate", state.config.aggregate, state.config.aggregate);
  pnh.param("ignore_missing_targets", state.config.ignore_missing_targets,
      state.config.ignore_missing_targets);
  pnh.param<std::string>("memory_limit_metric", state.config.memory_limit_metric,
      state.config.memory_limit_metric);
  pnh.param("memory_target_mb", state.config.memory_target_mb,
      state.config.memory_target_mb);
  pnh.param("cpu_target_percent", state.config.cpu_target_percent,
      state.config.cpu_target_percent);
  pnh.param<std::string>("resource_monitor/target_node_name",
      state.config.target_node_name, state.config.target_node_name);
  pnh.param<std::string>("resource_monitor/target_node_names",
      state.config.target_node_names, state.config.target_node_names);
  pnh.param("resource_monitor/publish_rate_hz", state.config.publish_rate_hz,
      state.config.publish_rate_hz);
  pnh.param("resource_monitor/cpu_smoothing_alpha",
      state.config.cpu_smoothing_alpha, state.config.cpu_smoothing_alpha);
  pnh.param("resource_monitor/aggregate",
      state.config.aggregate, state.config.aggregate);
  pnh.param("resource_monitor/ignore_missing_targets",
      state.config.ignore_missing_targets, state.config.ignore_missing_targets);
  pnh.param<std::string>("resource_monitor/memory_limit_metric",
      state.config.memory_limit_metric, state.config.memory_limit_metric);
  pnh.param("resource_monitor/memory_target_mb", state.config.memory_target_mb,
      state.config.memory_target_mb);
  pnh.param("resource_monitor/cpu_target_percent",
      state.config.cpu_target_percent, state.config.cpu_target_percent);

  state.config.publish_rate_hz =
      std::max(0.1, std::min(100.0, state.config.publish_rate_hz));
  state.config.cpu_smoothing_alpha =
      std::max(0.0, std::min(1.0, static_cast<double>(state.config.cpu_smoothing_alpha)));
  state.config.memory_target_mb =
      std::max(1.0, state.config.memory_target_mb);
  state.config.cpu_target_percent =
      std::max(1.0, state.config.cpu_target_percent);
  if (state.config.memory_limit_metric != "pss" &&
      state.config.memory_limit_metric != "rss_sum") {
    ROS_WARN("resource_monitor: unknown memory_limit_metric '%s'; using pss",
             state.config.memory_limit_metric.c_str());
    state.config.memory_limit_metric = "pss";
  }

  state.pub = pnh.advertise<livox_reflective_marker::ResourceUsage>(
      "resource_usage", 10);
  state.component_pub = pnh.advertise<livox_reflective_marker::ResourceUsage>(
      "resource_components", 10);

  std::vector<std::string> target_names;
  if (state.config.aggregate) {
    target_names = SplitTargetNames(state.config.target_node_names);
  }
  if (target_names.empty()) {
    target_names.push_back(state.config.target_node_name);
    state.config.aggregate = false;
  }
  state.targets.clear();
  state.targets.reserve(target_names.size());
  for (const auto& name : target_names) {
    state.targets.emplace_back(name);
  }

  ROS_INFO("resource_monitor_node started:");
  ROS_INFO("  mode=%s  rate=%.1f Hz  cpu_alpha=%.2f",
           state.config.aggregate ? "aggregate" : "single",
           state.config.publish_rate_hz,
           state.config.cpu_smoothing_alpha);
  if (state.config.aggregate) {
    ROS_INFO("  target_node_names=%s  ignore_missing=%s",
             state.config.target_node_names.c_str(),
             state.config.ignore_missing_targets ? "true" : "false");
  } else {
    ROS_INFO("  target_node_name=%s", state.config.target_node_name.c_str());
  }
  ROS_INFO("  memory_target=%.0f MB metric=%s  cpu_target=%.0f %%",
           state.config.memory_target_mb, state.config.memory_limit_metric.c_str(),
           state.config.cpu_target_percent);

  ros::Rate rate(state.config.publish_rate_hz);
  while (ros::ok()) {
    SampleAndPublish(&state);
    ros::spinOnce();
    rate.sleep();
  }

  ROS_INFO("resource_monitor_node exit.");
  return 0;
}
