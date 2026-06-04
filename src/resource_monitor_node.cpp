// ---------------------------------------------------------------------------
// resource_monitor_node
//
// Periodically reads /proc/<pid>/stat and /proc/<pid>/status for a target
// ROS node (by default "reflective_board_identifier_node") and publishes
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
  double publish_rate_hz = 2.0;          // how often to sample & publish
  double cpu_smoothing_alpha = 0.5f;     // EMA alpha for CPU percent (0..1)

  // Target thresholds for warnings
  double memory_target_mb = 50.0;        // target max RSS memory in MB
  double cpu_target_percent = 50.0;      // target max CPU usage in percent (half a core)
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
struct MonitorState {
  MonitorConfig config;

  // Cached PID (re-discovered on failure)
  int pid = -1;

  // Previous CPU ticks for delta computation
  unsigned long long prev_utime = 0;
  unsigned long long prev_stime = 0;
  unsigned long long prev_total_jiffies = 0;  // system-wide jiffies at prev sample
  bool have_prev = false;

  // Smoothed CPU percent
  float cpu_percent_smoothed = 0.0f;

  ros::Publisher pub;
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
void SampleAndPublish(MonitorState* state) {
  // Re-discover PID if lost
  if (state->pid < 0) {
    if (!LookupPidByRosNode(state->config.target_node_name, &state->pid)) {
      state->pid = FindPidByName(state->config.target_node_name);
    }
    if (state->pid < 0) {
      livox_reflective_marker::ResourceUsage msg;
      msg.header.stamp = ros::Time::now();
      msg.header.frame_id = "resource_monitor";
      msg.node_name = state->config.target_node_name;
      msg.cpu_percent = 0.0f;
      msg.cpu_time_sec = 0.0f;
      msg.memory_rss_kb = 0;
      msg.memory_vm_kb = 0;
      msg.pid = -1;
      msg.alive = false;
      state->pub.publish(msg);
      ROS_INFO("resource: node=%s pid=not_found alive=false",
               state->config.target_node_name.c_str());
      return;
    }
    ROS_INFO("resource_monitor: found '%s' with PID %d",
             state->config.target_node_name.c_str(), state->pid);
    state->have_prev = false;
  }

  // Read memory
  int64_t rss_kb = 0, vm_kb = 0;
  if (!ReadProcStatus(state->pid, rss_kb, vm_kb)) {
    ROS_WARN_THROTTLE(5.0,
      "resource_monitor: failed to read /proc/%d/status — process may have died",
      state->pid);
    state->pid = -1;
    return;
  }

  // Read CPU ticks
  unsigned long long utime = 0, stime = 0;
  if (!ReadProcStat(state->pid, utime, stime)) {
    ROS_WARN_THROTTLE(5.0,
      "resource_monitor: failed to read /proc/%d/stat — process may have died",
      state->pid);
    state->pid = -1;
    return;
  }

  // Compute CPU percentage
  float cpu_percent = 0.0f;
  if (state->have_prev) {
    unsigned long long delta_proc = (utime - state->prev_utime) +
                                    (stime - state->prev_stime);
    unsigned long long delta_sys =
        ReadSystemJiffies() - state->prev_total_jiffies;

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
  const float alpha = state->config.cpu_smoothing_alpha;
  if (state->have_prev) {
    state->cpu_percent_smoothed =
        alpha * cpu_percent + (1.0f - alpha) * state->cpu_percent_smoothed;
  } else {
    state->cpu_percent_smoothed = cpu_percent;
  }

  state->prev_utime = utime;
  state->prev_stime = stime;
  state->prev_total_jiffies = ReadSystemJiffies();
  state->have_prev = true;

  // Populate message
  livox_reflective_marker::ResourceUsage msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "resource_monitor";
  msg.node_name = state->config.target_node_name;
  msg.cpu_percent = state->cpu_percent_smoothed;
  msg.cpu_time_sec = static_cast<float>(utime + stime) /
                     static_cast<float>(sysconf(_SC_CLK_TCK));
  msg.memory_rss_kb = rss_kb;
  msg.memory_vm_kb = vm_kb;
  msg.pid = state->pid;
  msg.alive = true;

  state->pub.publish(msg);

  const double rss_mb = static_cast<double>(rss_kb) / 1024.0;
  const double cpu_val = static_cast<double>(state->cpu_percent_smoothed);

  const double mem_target = state->config.memory_target_mb;
  const double cpu_target = state->config.cpu_target_percent;

  const char* mem_ok = (rss_mb <= mem_target) ? "OK" : "EXCEEDED";
  const char* cpu_ok = (cpu_val <= cpu_target) ? "OK" : "EXCEEDED";

  ROS_INFO("resource: node=%s pid=%d cpu=%.1f%%/%0.f%%[%s] mem=%.1fMB/%0.fMB[%s] rss=%ldkB vm=%ldkB",
           msg.node_name.c_str(), msg.pid,
           cpu_val, cpu_target, cpu_ok,
           rss_mb, mem_target, mem_ok,
           static_cast<long>(rss_kb), static_cast<long>(vm_kb));

  if (rss_mb > mem_target) {
    ROS_WARN_THROTTLE(2.0,
      "resource_monitor: %s memory %.1f MB exceeds target %.0f MB",
      msg.node_name.c_str(), rss_mb, mem_target);
  }
  if (cpu_val > cpu_target) {
    ROS_WARN_THROTTLE(2.0,
      "resource_monitor: %s CPU %.1f%% exceeds target %.0f%%",
      msg.node_name.c_str(), cpu_val, cpu_target);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  ros::init(argc, argv, "resource_monitor_node");
  ros::NodeHandle pnh("~");

  MonitorState state;

  // Load config
  pnh.param<std::string>("target_node_name",
      state.config.target_node_name, state.config.target_node_name);
  pnh.param("publish_rate_hz", state.config.publish_rate_hz,
      state.config.publish_rate_hz);
  pnh.param("cpu_smoothing_alpha", state.config.cpu_smoothing_alpha,
      state.config.cpu_smoothing_alpha);
  pnh.param("memory_target_mb", state.config.memory_target_mb,
      state.config.memory_target_mb);
  pnh.param("cpu_target_percent", state.config.cpu_target_percent,
      state.config.cpu_target_percent);
  pnh.param<std::string>("resource_monitor/target_node_name",
      state.config.target_node_name, state.config.target_node_name);
  pnh.param("resource_monitor/publish_rate_hz", state.config.publish_rate_hz,
      state.config.publish_rate_hz);
  pnh.param("resource_monitor/cpu_smoothing_alpha",
      state.config.cpu_smoothing_alpha, state.config.cpu_smoothing_alpha);
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

  state.pub = pnh.advertise<livox_reflective_marker::ResourceUsage>(
      "resource_usage", 10);

  ROS_INFO("resource_monitor_node started:");
  ROS_INFO("  target_node_name=%s  rate=%.1f Hz  cpu_alpha=%.2f",
           state.config.target_node_name.c_str(),
           state.config.publish_rate_hz,
           state.config.cpu_smoothing_alpha);
  ROS_INFO("  memory_target=%.0f MB  cpu_target=%.0f %%",
           state.config.memory_target_mb,
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
