/// @file target_manager_node.cpp
/// @brief Three-state target decision manager for BPU recognition results.
///
/// The recognition node stays a pure scorer.  This node owns candidate identity,
/// target selection, periodic audit, target switching, EKF handoff, and the
/// coloured high-reflective visualization cloud.

#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <std_msgs/Empty.h>

#include <livox_reflective_marker/BpuScoreEntry.h>
#include <livox_reflective_marker/BpuScores.h>
#include <livox_reflective_marker/ClusterCloud.h>
#include <livox_reflective_marker/EkfStatus.h>
#include <livox_reflective_marker/RecognitionCommand.h>
#include <livox_reflective_marker/TargetCommand.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint8_t kCommandInit = 0;
constexpr uint8_t kCommandCorrect = 1;
constexpr uint8_t kCommandGateUpdate = 2;

enum class DecisionState { kSearching, kRecognized, kConfirming };

const char* StateName(DecisionState state) {
  switch (state) {
    case DecisionState::kSearching:
      return "SEARCHING";
    case DecisionState::kRecognized:
      return "RECOGNIZED";
    case DecisionState::kConfirming:
      return "CONFIRMING";
  }
  return "UNKNOWN";
}

const char* CommandName(uint8_t action) {
  switch (action) {
    case kCommandInit:
      return "INIT";
    case kCommandCorrect:
      return "CORRECT";
    case kCommandGateUpdate:
      return "GATE_UPDATE";
  }
  return "UNKNOWN";
}

Eigen::Vector3f ToEigen(const geometry_msgs::Point& p) {
  return Eigen::Vector3f(static_cast<float>(p.x),
                         static_cast<float>(p.y),
                         static_cast<float>(p.z));
}

geometry_msgs::Point ToPoint(const Eigen::Vector3f& p) {
  geometry_msgs::Point out;
  out.x = p.x();
  out.y = p.y();
  out.z = p.z();
  return out;
}

float Distance(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
  return (a - b).norm();
}

struct Config {
  // Inputs.
  std::string bpu_scores_topic = "/siamese_bpu_infer_node/bpu_scores";
  std::string cluster_cloud_topic =
      "/siamese_bpu_infer_node/latest_cluster_cloud";
  std::string ekf_status_topic = "/ekf_pose_node/ekf_status";

  // Outputs.
  std::string target_command_topic = "/target_manager/target_command";
  std::string tracking_lost_topic = "/target_manager/tracking_lost";
  std::string recognition_command_topic =
      "/target_manager/recognition_command";
  std::string colored_cloud_topic = "/target_manager/colored_high_points";

  std::string frame_id = "livox_frame";

  // Spatial association.  Candidate IDs are per-window only; persistent
  // identity is created here from 3-D centre motion.
  float track_match_distance_m = 0.50f;
  float association_margin_m = 0.08f;
  double track_timeout_sec = 2.0;
  int max_tracks = 12;
  int history_size = 40;

  // SEARCHING -> RECOGNIZED.
  int init_required_frames = 5;
  float init_score_threshold = 0.3825f;
  float init_margin_threshold = 0.20f;
  float single_candidate_virtual_second_score = 0.10f;

  // Periodic audit while RECOGNIZED.
  double audit_interval_sec = 5.0;
  int audit_required_frames = 3;
  int audit_fail_windows = 2;
  float audit_score_threshold = 0.30f;
  float audit_margin_threshold = 0.05f;

  // CONFIRMING target switch.
  int switch_required_frames = 8;
  float switch_score_threshold = 0.42f;
  float switch_margin_threshold = 0.20f;
  float switch_min_distance_m = 0.25f;
  float reinit_distance_m = 0.60f;

  // Evidence that the active target is actually bad.
  int current_bad_frames = 3;
  float current_bad_score_threshold = 0.25f;
  float current_bad_margin_threshold = 0.10f;
  double confirm_timeout_sec = 8.0;

  // Active target handoff to EKF.  While RECOGNIZED, only an explicitly
  // observed active target is sent downstream as a correction.
  double active_correction_min_interval_sec = 0.25;
  float active_correction_score_threshold = 0.30f;
  float active_correction_margin_threshold = -0.20f;

  // Recognition-node runtime profiles.
  int search_accumulation_frames = 20;
  double search_publish_interval_sec = 0.10;
  int tracking_accumulation_frames = 10;
  double tracking_publish_interval_sec = 0.15;
  int confirm_accumulation_frames = 30;
  double confirm_publish_interval_sec = 0.08;

  // Visualization.
  float target_cluster_match_distance_m = 0.50f;

};

struct ScoreSample {
  uint32_t window_id = 0;
  ros::Time stamp;
  float score = 0.0f;
  float margin_to_best_other = 0.0f;
  bool is_best = false;
};

struct Track {
  int id = -1;
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  float score = 0.0f;
  float margin_to_best_other = 0.0f;
  ros::Time first_seen;
  ros::Time last_seen;
  int missed_windows = 0;
  int observed_run = 0;
  int best_run = 0;
  bool observed_this_window = false;
  bool best_this_window = false;
  std::deque<ScoreSample> samples;
};

struct CandidateStats {
  bool valid = false;
  int sample_count = 0;
  int best_count = 0;
  float avg_score = 0.0f;
  float avg_margin = 0.0f;
};

struct EntryMeta {
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  float score = 0.0f;
  float other_best_score = 0.0f;
  bool is_best = false;
};

class TargetManager {
 public:
  TargetManager(ros::NodeHandle nh, ros::NodeHandle pnh)
      : nh_(nh), pnh_(pnh) {
    LoadConfig();

    target_cmd_pub_ =
        nh_.advertise<livox_reflective_marker::TargetCommand>(
            cfg_.target_command_topic, 4, true);
    tracking_lost_pub_ =
        nh_.advertise<std_msgs::Empty>(cfg_.tracking_lost_topic, 4);
    recognition_cmd_pub_ =
        nh_.advertise<livox_reflective_marker::RecognitionCommand>(
            cfg_.recognition_command_topic, 4, true);
    colored_cloud_pub_ =
        nh_.advertise<sensor_msgs::PointCloud2>(
            cfg_.colored_cloud_topic, 4, true);
    bpu_scores_sub_ = nh_.subscribe<livox_reflective_marker::BpuScores>(
        cfg_.bpu_scores_topic, 4, &TargetManager::BpuScoresCallback, this);
    cluster_cloud_sub_ =
        nh_.subscribe<livox_reflective_marker::ClusterCloud>(
            cfg_.cluster_cloud_topic, 4,
            &TargetManager::ClusterCloudCallback, this);
    ekf_status_sub_ = nh_.subscribe<livox_reflective_marker::EkfStatus>(
        cfg_.ekf_status_topic, 4, &TargetManager::EkfStatusCallback, this);
    state_enter_time_ = ros::Time::now();
    PublishRecognitionConfig();

    ROS_INFO("[target_mgr] ready | state=%s init=%d/%.3f/%.3f "
             "audit=%ds every %.1fs switch=%d/%.3f/%.3f",
             StateName(state_), cfg_.init_required_frames,
             static_cast<double>(cfg_.init_score_threshold),
             static_cast<double>(cfg_.init_margin_threshold),
             cfg_.audit_required_frames, cfg_.audit_interval_sec,
             cfg_.switch_required_frames,
             static_cast<double>(cfg_.switch_score_threshold),
             static_cast<double>(cfg_.switch_margin_threshold));
  }

 private:
  void LoadConfig() {
#define LOAD(name, var) pnh_.param(name, var, var)
    LOAD("bpu_scores_topic", cfg_.bpu_scores_topic);
    LOAD("cluster_cloud_topic", cfg_.cluster_cloud_topic);
    LOAD("ekf_status_topic", cfg_.ekf_status_topic);
    LOAD("target_command_topic", cfg_.target_command_topic);
    LOAD("tracking_lost_topic", cfg_.tracking_lost_topic);
    LOAD("recognition_command_topic", cfg_.recognition_command_topic);
    LOAD("colored_cloud_topic", cfg_.colored_cloud_topic);
    LOAD("frame_id", cfg_.frame_id);

    // Compatibility with existing launch parameter names.
    LOAD("track_match_distance_m", cfg_.track_match_distance_m);
    LOAD("track_timeout_sec", cfg_.track_timeout_sec);
    LOAD("confirm_frames", cfg_.init_required_frames);
    LOAD("reacquire_confirm_frames", cfg_.switch_required_frames);
    LOAD("lost_frames", cfg_.current_bad_frames);
    LOAD("score_threshold", cfg_.init_score_threshold);
    LOAD("init_score_margin", cfg_.init_margin_threshold);
    LOAD("target_cluster_match_distance_m",
         cfg_.target_cluster_match_distance_m);
    LOAD("max_tracks", cfg_.max_tracks);

    // New explicit state-machine parameters.
    LOAD("association_margin_m", cfg_.association_margin_m);
    LOAD("history_size", cfg_.history_size);
    LOAD("init_required_frames", cfg_.init_required_frames);
    LOAD("init_score_threshold", cfg_.init_score_threshold);
    LOAD("init_margin_threshold", cfg_.init_margin_threshold);
    LOAD("single_candidate_virtual_second_score",
         cfg_.single_candidate_virtual_second_score);
    LOAD("audit_interval_sec", cfg_.audit_interval_sec);
    LOAD("audit_required_frames", cfg_.audit_required_frames);
    LOAD("audit_fail_windows", cfg_.audit_fail_windows);
    LOAD("audit_score_threshold", cfg_.audit_score_threshold);
    LOAD("audit_margin_threshold", cfg_.audit_margin_threshold);
    LOAD("switch_required_frames", cfg_.switch_required_frames);
    LOAD("switch_score_threshold", cfg_.switch_score_threshold);
    LOAD("switch_margin_threshold", cfg_.switch_margin_threshold);
    LOAD("switch_min_distance_m", cfg_.switch_min_distance_m);
    LOAD("reinit_distance_m", cfg_.reinit_distance_m);
    LOAD("current_bad_frames", cfg_.current_bad_frames);
    LOAD("current_bad_score_threshold", cfg_.current_bad_score_threshold);
    LOAD("current_bad_margin_threshold", cfg_.current_bad_margin_threshold);
    LOAD("confirm_timeout_sec", cfg_.confirm_timeout_sec);
    LOAD("active_correction_min_interval_sec",
         cfg_.active_correction_min_interval_sec);
    LOAD("active_correction_score_threshold",
         cfg_.active_correction_score_threshold);
    LOAD("active_correction_margin_threshold",
         cfg_.active_correction_margin_threshold);
    LOAD("search_accumulation_frames", cfg_.search_accumulation_frames);
    LOAD("search_publish_interval_sec", cfg_.search_publish_interval_sec);
    LOAD("tracking_accumulation_frames", cfg_.tracking_accumulation_frames);
    LOAD("tracking_publish_interval_sec", cfg_.tracking_publish_interval_sec);
    LOAD("confirm_accumulation_frames", cfg_.confirm_accumulation_frames);
    LOAD("confirm_publish_interval_sec", cfg_.confirm_publish_interval_sec);
#undef LOAD

    cfg_.track_match_distance_m = std::max(0.05f, cfg_.track_match_distance_m);
    cfg_.association_margin_m = std::max(0.0f, cfg_.association_margin_m);
    cfg_.track_timeout_sec = std::max(0.1, cfg_.track_timeout_sec);
    cfg_.max_tracks = std::max(1, cfg_.max_tracks);
    cfg_.init_required_frames = std::max(1, cfg_.init_required_frames);
    cfg_.audit_required_frames = std::max(1, cfg_.audit_required_frames);
    cfg_.audit_fail_windows = std::max(1, cfg_.audit_fail_windows);
    cfg_.switch_required_frames = std::max(1, cfg_.switch_required_frames);
    cfg_.current_bad_frames = std::max(1, cfg_.current_bad_frames);
    cfg_.audit_interval_sec = std::max(0.2, cfg_.audit_interval_sec);
    cfg_.confirm_timeout_sec = std::max(0.5, cfg_.confirm_timeout_sec);
    cfg_.active_correction_min_interval_sec =
        std::max(0.05, cfg_.active_correction_min_interval_sec);
    cfg_.switch_min_distance_m = std::max(0.0f, cfg_.switch_min_distance_m);
    cfg_.reinit_distance_m = std::max(0.0f, cfg_.reinit_distance_m);
    cfg_.history_size = std::max(
        cfg_.history_size,
        std::max(cfg_.init_required_frames,
                 std::max(cfg_.audit_required_frames,
                          cfg_.switch_required_frames)) + 4);
  }

  void BpuScoresCallback(
      const livox_reflective_marker::BpuScores::ConstPtr& scores) {
    ros::Time now = scores->header.stamp;
    if (now.isZero()) now = ros::Time::now();
    last_window_id_ = scores->window_id;

    UpdateTracks(scores->entries, scores->window_id, now);
    LogScoreWindowSummary(*scores);
    switch (state_) {
      case DecisionState::kSearching:
        HandleSearching(now);
        break;
      case DecisionState::kRecognized:
        HandleRecognized(now);
        break;
      case DecisionState::kConfirming:
        HandleConfirming(now);
        break;
    }

    if (state_ == DecisionState::kRecognized) {
      PublishActiveCorrectionIfFresh(now);
    }
    LogDecisionDebug(now);

    PublishColoredCloud();
  }

  void LogScoreWindowSummary(
      const livox_reflective_marker::BpuScores& scores) const {
    int top1 = -1, top2 = -1, top3 = -1;
    for (size_t i = 0; i < scores.entries.size(); ++i) {
      if (top1 < 0 || scores.entries[i].score > scores.entries[top1].score) {
        top3 = top2;
        top2 = top1;
        top1 = static_cast<int>(i);
      } else if (top2 < 0 ||
                 scores.entries[i].score > scores.entries[top2].score) {
        top3 = top2;
        top2 = static_cast<int>(i);
      } else if (top3 < 0 ||
                 scores.entries[i].score > scores.entries[top3].score) {
        top3 = static_cast<int>(i);
      }
    }

    const auto get_score = [&](int idx) {
      return idx >= 0 ? scores.entries[idx].score : -1.0f;
    };
    const float s1 = get_score(top1);
    const float s2 = get_score(top2);
    const float s3 = get_score(top3);
    const float margin = top1 >= 0
                             ? s1 - (top2 >= 0 ? s2
                                                : cfg_.single_candidate_virtual_second_score)
                             : 0.0f;
    geometry_msgs::Point c;
    if (top1 >= 0) c = scores.entries[top1].center;

    ROS_INFO_THROTTLE(
        1.0,
        "[target_mgr] score_window window=%u entries=%zu tracks=%zu state=%s "
        "top1_entry=%d score=%.4f margin=%.4f center=[%.2f,%.2f,%.2f] "
        "top2_entry=%d score=%.4f top3_entry=%d score=%.4f",
        scores.window_id, scores.entries.size(), tracks_.size(),
        StateName(state_), top1, s1, margin, static_cast<double>(c.x),
        static_cast<double>(c.y), static_cast<double>(c.z), top2, s2, top3,
        s3);
  }

  const Track* BestObservedTrack() const {
    const Track* best = nullptr;
    for (const auto& track : tracks_) {
      if (!track.observed_this_window) continue;
      if (best == nullptr || track.score > best->score ||
          (track.score == best->score &&
           track.margin_to_best_other > best->margin_to_best_other)) {
        best = &track;
      }
    }
    return best;
  }

  void LogDecisionDebug(const ros::Time& now) const {
    switch (state_) {
      case DecisionState::kSearching:
        LogSearchingDebug(now);
        break;
      case DecisionState::kRecognized:
        LogRecognizedDebug(now);
        break;
      case DecisionState::kConfirming:
        LogConfirmingDebug(now);
        break;
    }
  }

  void LogSearchingDebug(const ros::Time& now) const {
    const Track* best = BestObservedTrack();
    if (best == nullptr) {
      ROS_INFO_THROTTLE(
          1.0,
          "[target_mgr] search_debug window=%u tracks=%zu observed=0 "
          "reason=no_observed_track need_frames=%d score>%.3f margin>=%.3f",
          last_window_id_, tracks_.size(), cfg_.init_required_frames,
          static_cast<double>(cfg_.init_score_threshold),
          static_cast<double>(cfg_.init_margin_threshold));
      return;
    }

    const CandidateStats stats = LastStats(*best, cfg_.init_required_frames);
    const bool frames_ok = stats.valid;
    const bool best_ok =
        stats.valid && stats.best_count == cfg_.init_required_frames;
    const bool score_ok =
        stats.valid && stats.avg_score > cfg_.init_score_threshold;
    const bool margin_ok =
        stats.valid && stats.avg_margin >= cfg_.init_margin_threshold;

    ROS_INFO_THROTTLE(
        1.0,
        "[target_mgr] search_debug window=%u tracks=%zu best_track=%d "
        "cur_score=%.4f cur_margin=%.4f observed_run=%d best_run=%d "
        "samples=%zu last_seen_age=%.2fs stats_valid=%d stats_count=%d/%d "
        "stats_best=%d/%d avg_score=%.4f need_score>%.3f avg_margin=%.4f "
        "need_margin>=%.3f blockers=%s%s%s%s center=[%.2f,%.2f,%.2f]",
        last_window_id_, tracks_.size(), best->id,
        static_cast<double>(best->score),
        static_cast<double>(best->margin_to_best_other), best->observed_run,
        best->best_run, best->samples.size(),
        static_cast<double>((now - best->last_seen).toSec()), stats.valid ? 1 : 0,
        stats.sample_count, cfg_.init_required_frames, stats.best_count,
        cfg_.init_required_frames, static_cast<double>(stats.avg_score),
        static_cast<double>(cfg_.init_score_threshold),
        static_cast<double>(stats.avg_margin),
        static_cast<double>(cfg_.init_margin_threshold),
        frames_ok ? "" : "frames ",
        best_ok ? "" : "best_count ",
        score_ok ? "" : "score ",
        margin_ok ? "" : "margin ",
        static_cast<double>(best->center.x()),
        static_cast<double>(best->center.y()),
        static_cast<double>(best->center.z()));
  }

  void LogRecognizedDebug(const ros::Time& now) const {
    const Track* active = nullptr;
    for (const auto& track : tracks_) {
      if (track.id == active_track_id_) {
        active = &track;
        break;
      }
    }
    if (active == nullptr) {
      ROS_INFO_THROTTLE(
          1.0,
          "[target_mgr] recognized_debug window=%u active=%d missing tracks=%zu "
          "next_audit_in=%.2fs",
          last_window_id_, active_track_id_, tracks_.size(),
          static_cast<double>((next_audit_time_ - now).toSec()));
      return;
    }
    ROS_INFO_THROTTLE(
        1.0,
        "[target_mgr] recognized_debug window=%u active=%d observed=%d "
        "cur_score=%.4f cur_margin=%.4f observed_run=%d best_run=%d "
        "next_audit_in=%.2fs center=[%.2f,%.2f,%.2f]",
        last_window_id_, active->id, active->observed_this_window ? 1 : 0,
        static_cast<double>(active->score),
        static_cast<double>(active->margin_to_best_other), active->observed_run,
        active->best_run, static_cast<double>((next_audit_time_ - now).toSec()),
        static_cast<double>(active->center.x()),
        static_cast<double>(active->center.y()),
        static_cast<double>(active->center.z()));
  }

  void LogConfirmingDebug(const ros::Time& now) const {
    const Track* best = BestObservedTrack();
    ROS_INFO_THROTTLE(
        1.0,
        "[target_mgr] confirming_debug window=%u active=%d tracks=%zu "
        "current_bad_run=%d/%d confirm_elapsed=%.2fs/%.2fs best_observed=%d "
        "best_score=%.4f best_margin=%.4f",
        last_window_id_, active_track_id_, tracks_.size(), current_bad_run_,
        cfg_.current_bad_frames,
        static_cast<double>((now - confirm_start_time_).toSec()),
        static_cast<double>(cfg_.confirm_timeout_sec), best ? best->id : -1,
        static_cast<double>(best ? best->score : -1.0f),
        static_cast<double>(best ? best->margin_to_best_other : 0.0f));
  }

  void UpdateTracks(
      const std::vector<livox_reflective_marker::BpuScoreEntry>& entries,
      uint32_t window_id, const ros::Time& now) {
    PruneTimedOutTracks(now);
    for (auto& track : tracks_) {
      track.observed_this_window = false;
      track.best_this_window = false;
    }

    const std::vector<EntryMeta> meta = BuildEntryMeta(entries);
    std::vector<int> assignment(entries.size(), -1);
    std::vector<int> entry_best_track(entries.size(), -1);
    Associate(entries, &assignment, &entry_best_track);

    std::vector<bool> matched_track(tracks_.size(), false);
    for (size_t ei = 0; ei < entries.size(); ++ei) {
      const int ti = assignment[ei];
      if (ti < 0 || ti >= static_cast<int>(tracks_.size())) continue;
      UpdateMatchedTrack(&tracks_[ti], meta[ei], window_id, now);
      matched_track[ti] = true;
    }

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
      if (ti < matched_track.size() && matched_track[ti]) continue;
      MarkTrackMissed(&tracks_[ti]);
    }

    for (size_t ei = 0; ei < entries.size(); ++ei) {
      if (assignment[ei] >= 0) continue;
      if (entry_best_track[ei] >= 0) continue;  // near but ambiguous/occupied
      if (static_cast<int>(tracks_.size()) >= cfg_.max_tracks) break;
      CreateTrack(meta[ei], window_id, now);
    }
  }

  std::vector<EntryMeta> BuildEntryMeta(
      const std::vector<livox_reflective_marker::BpuScoreEntry>& entries)
      const {
    std::vector<EntryMeta> meta(entries.size());
    if (entries.empty()) return meta;

    std::vector<size_t> order(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) {
                return entries[a].score > entries[b].score;
              });
    const size_t best_idx = order.front();
    const float best_score = entries[best_idx].score;
    const float second_score =
        order.size() > 1 ? entries[order[1]].score
                         : cfg_.single_candidate_virtual_second_score;

    for (size_t i = 0; i < entries.size(); ++i) {
      meta[i].center = ToEigen(entries[i].center);
      meta[i].score = entries[i].score;
      meta[i].is_best = (i == best_idx);
      meta[i].other_best_score =
          meta[i].is_best ? second_score : best_score;
    }
    return meta;
  }

  void Associate(
      const std::vector<livox_reflective_marker::BpuScoreEntry>& entries,
      std::vector<int>* assignment,
      std::vector<int>* entry_best_track) const {
    if (assignment == nullptr || entry_best_track == nullptr ||
        entries.empty() || tracks_.empty()) {
      return;
    }

    const float max_dist = cfg_.track_match_distance_m;
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<int> track_best_entry(tracks_.size(), -1);
    std::vector<float> track_best_dist(tracks_.size(), inf);
    std::vector<float> track_second_dist(tracks_.size(), inf);
    std::vector<float> entry_best_dist(entries.size(), inf);
    std::vector<float> entry_second_dist(entries.size(), inf);

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
      for (size_t ei = 0; ei < entries.size(); ++ei) {
        const float d = Distance(tracks_[ti].center, ToEigen(entries[ei].center));
        if (d > max_dist) continue;

        if (d < track_best_dist[ti]) {
          track_second_dist[ti] = track_best_dist[ti];
          track_best_dist[ti] = d;
          track_best_entry[ti] = static_cast<int>(ei);
        } else if (d < track_second_dist[ti]) {
          track_second_dist[ti] = d;
        }

        if (d < entry_best_dist[ei]) {
          entry_second_dist[ei] = entry_best_dist[ei];
          entry_best_dist[ei] = d;
          (*entry_best_track)[ei] = static_cast<int>(ti);
        } else if (d < entry_second_dist[ei]) {
          entry_second_dist[ei] = d;
        }
      }
    }

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
      const int ei = track_best_entry[ti];
      if (ei < 0) continue;
      if ((*entry_best_track)[ei] != static_cast<int>(ti)) continue;

      const bool track_ambiguous =
          std::isfinite(track_second_dist[ti]) &&
          (track_second_dist[ti] - track_best_dist[ti] <
           cfg_.association_margin_m);
      const bool entry_ambiguous =
          std::isfinite(entry_second_dist[ei]) &&
          (entry_second_dist[ei] - entry_best_dist[ei] <
           cfg_.association_margin_m);
      if (track_ambiguous || entry_ambiguous) continue;
      (*assignment)[ei] = static_cast<int>(ti);
    }
  }

  void UpdateMatchedTrack(Track* track, const EntryMeta& entry,
                          uint32_t window_id, const ros::Time& now) {
    if (track == nullptr) return;

    bool consecutive = true;
    if (!track->samples.empty()) {
      consecutive = (window_id == track->samples.back().window_id + 1);
    }
    if (!consecutive) {
      track->observed_run = 0;
      track->best_run = 0;
    }

    track->center = entry.center;
    track->score = entry.score;
    track->margin_to_best_other = entry.score - entry.other_best_score;
    track->last_seen = now;
    track->missed_windows = 0;
    track->observed_this_window = true;
    track->best_this_window = entry.is_best;
    track->observed_run = track->observed_run + 1;
    track->best_run = entry.is_best ? track->best_run + 1 : 0;

    ScoreSample sample;
    sample.window_id = window_id;
    sample.stamp = now;
    sample.score = entry.score;
    sample.margin_to_best_other = track->margin_to_best_other;
    sample.is_best = entry.is_best;
    track->samples.push_back(sample);
    while (static_cast<int>(track->samples.size()) > cfg_.history_size) {
      track->samples.pop_front();
    }
  }

  void MarkTrackMissed(Track* track) {
    if (track == nullptr) return;
    track->observed_this_window = false;
    track->best_this_window = false;
    track->missed_windows++;
    track->observed_run = 0;
    track->best_run = 0;
  }

  void CreateTrack(const EntryMeta& entry, uint32_t window_id,
                   const ros::Time& now) {
    Track track;
    track.id = next_track_id_++;
    track.center = entry.center;
    track.first_seen = now;
    UpdateMatchedTrack(&track, entry, window_id, now);
    tracks_.push_back(track);
  }

  void PruneTimedOutTracks(const ros::Time& now) {
    tracks_.erase(
        std::remove_if(
            tracks_.begin(), tracks_.end(),
            [&](const Track& track) {
              if (track.id == active_track_id_) return false;
              if (track.last_seen.isZero()) return false;
              return (now - track.last_seen).toSec() > cfg_.track_timeout_sec;
            }),
        tracks_.end());
  }

  CandidateStats LastStats(const Track& track, int required_frames) const {
    CandidateStats stats;
    if (required_frames <= 0 ||
        static_cast<int>(track.samples.size()) < required_frames) {
      return stats;
    }
    if (track.samples.back().window_id != last_window_id_) {
      return stats;
    }

    const size_t start = track.samples.size() -
                         static_cast<size_t>(required_frames);
    for (size_t i = start; i < track.samples.size(); ++i) {
      if (i > start &&
          track.samples[i].window_id != track.samples[i - 1].window_id + 1) {
        return CandidateStats{};
      }
      stats.sample_count++;
      stats.avg_score += track.samples[i].score;
      stats.avg_margin += track.samples[i].margin_to_best_other;
      if (track.samples[i].is_best) stats.best_count++;
    }

    stats.avg_score /= static_cast<float>(stats.sample_count);
    stats.avg_margin /= static_cast<float>(stats.sample_count);
    stats.valid = true;
    return stats;
  }

  bool MeetsStableRule(const Track& track, int required_frames,
                       float score_threshold, float margin_threshold,
                       CandidateStats* out_stats = nullptr) const {
    const CandidateStats stats = LastStats(track, required_frames);
    if (out_stats != nullptr) *out_stats = stats;
    return stats.valid &&
           stats.best_count == required_frames &&
           stats.avg_score > score_threshold &&
           stats.avg_margin >= margin_threshold;
  }

  Track* FindTrackById(int id) {
    for (auto& track : tracks_) {
      if (track.id == id) return &track;
    }
    return nullptr;
  }

  Track* FindBestInitialCandidate(CandidateStats* out_stats) {
    return FindBestStableCandidate(-1, cfg_.init_required_frames,
                                   cfg_.init_score_threshold,
                                   cfg_.init_margin_threshold, 0.0f,
                                   out_stats);
  }

  Track* FindBestSwitchCandidate(CandidateStats* out_stats) {
    return FindBestStableCandidate(active_track_id_,
                                   cfg_.switch_required_frames,
                                   cfg_.switch_score_threshold,
                                   cfg_.switch_margin_threshold,
                                   cfg_.switch_min_distance_m, out_stats);
  }

  Track* FindBestStableCandidate(int excluded_track_id, int required_frames,
                                 float score_threshold,
                                 float margin_threshold,
                                 float min_distance_from_active,
                                 CandidateStats* out_stats) {
    Track* best_track = nullptr;
    CandidateStats best_stats;

    for (auto& track : tracks_) {
      if (track.id == excluded_track_id) continue;
      if (excluded_track_id >= 0 &&
          min_distance_from_active > 0.0f &&
          Distance(track.center, active_center_) < min_distance_from_active) {
        continue;
      }

      CandidateStats stats;
      if (!MeetsStableRule(track, required_frames, score_threshold,
                           margin_threshold, &stats)) {
        continue;
      }

      if (!best_track || stats.avg_score > best_stats.avg_score ||
          (stats.avg_score == best_stats.avg_score &&
           stats.avg_margin > best_stats.avg_margin)) {
        best_track = &track;
        best_stats = stats;
      }
    }

    if (out_stats != nullptr) *out_stats = best_stats;
    return best_track;
  }

  void HandleSearching(const ros::Time& now) {
    CandidateStats stats;
    Track* candidate = FindBestInitialCandidate(&stats);
    if (!candidate) return;

    ROS_WARN("[target_mgr] initial target recognized track=%d "
             "avg_score=%.3f avg_margin=%.3f pos=[%.2f,%.2f,%.2f]",
             candidate->id, static_cast<double>(stats.avg_score),
             static_cast<double>(stats.avg_margin),
             static_cast<double>(candidate->center.x()),
             static_cast<double>(candidate->center.y()),
             static_cast<double>(candidate->center.z()));
    IssueTarget(candidate, 0, now, "INIT");
  }

  void HandleRecognized(const ros::Time& now) {
    if ((now - next_audit_time_).toSec() < 0.0) return;

    Track* active = FindTrackById(active_track_id_);
    CandidateStats stats;
    const bool audit_ok =
        active && MeetsStableRule(*active, cfg_.audit_required_frames,
                                  cfg_.audit_score_threshold,
                                  cfg_.audit_margin_threshold, &stats);
    if (audit_ok) {
      active_center_ = active->center;
      audit_fail_windows_ = 0;
      next_audit_time_ = now + ros::Duration(cfg_.audit_interval_sec);
      ROS_INFO("[target_mgr] audit OK track=%d avg_score=%.3f "
               "avg_margin=%.3f next=%.1fs",
               active->id, static_cast<double>(stats.avg_score),
               static_cast<double>(stats.avg_margin),
               cfg_.audit_interval_sec);
      return;
    }

    audit_fail_windows_++;
    ROS_WARN("[target_mgr] audit failed %d/%d for active track=%d",
             audit_fail_windows_, cfg_.audit_fail_windows, active_track_id_);
    if (audit_fail_windows_ >= cfg_.audit_fail_windows) {
      EnterConfirming(now, "AUDIT_FAILED");
    } else {
      next_audit_time_ = now + ros::Duration(cfg_.audit_interval_sec);
    }
  }

  void HandleConfirming(const ros::Time& now) {
    Track* active = FindTrackById(active_track_id_);
    CandidateStats active_stats;
    const bool active_recovered =
        active && MeetsStableRule(*active, cfg_.audit_required_frames,
                                  cfg_.audit_score_threshold,
                                  cfg_.audit_margin_threshold,
                                  &active_stats);
    if (active_recovered) {
      ROS_WARN("[target_mgr] active target recovered track=%d "
               "avg_score=%.3f avg_margin=%.3f",
               active->id, static_cast<double>(active_stats.avg_score),
               static_cast<double>(active_stats.avg_margin));
      EnterRecognized(active->id, active->center, now, "ACTIVE_RECOVERED");
      return;
    }

    UpdateCurrentBadRun(active);

    CandidateStats switch_stats;
    Track* challenger = FindBestSwitchCandidate(&switch_stats);
    if (current_bad_run_ >= cfg_.current_bad_frames && challenger) {
      const uint8_t action =
          ShouldReinit(*challenger) ? kCommandInit : kCommandCorrect;
      ROS_WARN("[target_mgr] target switch accepted old=%d new=%d "
               "action=%s bad=%d avg_score=%.3f avg_margin=%.3f",
               active_track_id_, challenger->id,
               CommandName(action), current_bad_run_,
               static_cast<double>(switch_stats.avg_score),
               static_cast<double>(switch_stats.avg_margin));
      IssueTarget(challenger, action, now, "SWITCH");
      return;
    }

    if ((now - confirm_start_time_).toSec() > cfg_.confirm_timeout_sec) {
      if (current_bad_run_ >= cfg_.current_bad_frames || active == nullptr) {
        ROS_WARN("[target_mgr] confirm timeout without a stable replacement");
        tracking_lost_pub_.publish(std_msgs::Empty());
        EnterSearching(now, "CONFIRM_TIMEOUT");
      } else {
        ROS_WARN("[target_mgr] confirm timeout but active target not proven bad");
        EnterRecognized(active->id, active->center, now,
                        "CONFIRM_TIMEOUT_KEEP_ACTIVE");
      }
    }
  }

  void UpdateCurrentBadRun(const Track* active) {
    bool bad = true;
    if (active && active->observed_this_window) {
      bad = active->score <= cfg_.current_bad_score_threshold ||
            active->margin_to_best_other <=
                -cfg_.current_bad_margin_threshold;
    }

    if (bad) {
      current_bad_run_++;
    } else {
      current_bad_run_ = 0;
    }
  }

  bool ShouldReinit(const Track& target) const {
    if (active_track_id_ < 0) return true;
    return Distance(target.center, active_center_) >= cfg_.reinit_distance_m;
  }

  void PublishActiveCorrectionIfFresh(const ros::Time& stamp) {
    Track* active = FindTrackById(active_track_id_);
    if (!active || !active->observed_this_window) return;
    if (active->score <= cfg_.active_correction_score_threshold) return;
    if (active->margin_to_best_other <
        cfg_.active_correction_margin_threshold) {
      return;
    }
    if (!last_target_command_time_.isZero() &&
        (stamp - last_target_command_time_).toSec() <
            cfg_.active_correction_min_interval_sec) {
      return;
    }

    PublishCommand(*active, kCommandGateUpdate, stamp, "TRACK_UPDATE");
    active_center_ = active->center;
  }

  void IssueTarget(Track* track, uint8_t action, const ros::Time& stamp,
                   const std::string& source) {
    if (track == nullptr) return;
    PublishCommand(*track, action, stamp, source);
    EnterRecognized(track->id, track->center, stamp, source);
  }

  void PublishCommand(const Track& track, uint8_t action,
                      const ros::Time& stamp,
                      const std::string& source) {
    livox_reflective_marker::TargetCommand cmd;
    cmd.header.stamp = stamp.isZero() ? ros::Time::now() : stamp;
    cmd.header.frame_id = cfg_.frame_id;
    cmd.action = action;
    cmd.pose.header = cmd.header;
    cmd.pose.pose.position = ToPoint(track.center);
    if (has_last_ekf_orientation_) {
      cmd.pose.pose.orientation = last_ekf_orientation_;
    } else {
      cmd.pose.pose.orientation.w = 1.0;
    }
    cmd.init_cloud = sensor_msgs::PointCloud2{};
    cmd.init_cloud.header = cmd.header;
    target_cmd_pub_.publish(cmd);
    last_target_command_time_ = cmd.header.stamp;

    if (source == "TRACK_UPDATE") {
      ROS_INFO_THROTTLE(
          1.0,
          "[target_mgr] target_command %s/%s track=%d score=%.3f "
          "margin=%.3f pos=[%.2f,%.2f,%.2f]",
          source.c_str(), CommandName(action), track.id,
          static_cast<double>(track.score),
          static_cast<double>(track.margin_to_best_other),
          static_cast<double>(track.center.x()),
          static_cast<double>(track.center.y()),
          static_cast<double>(track.center.z()));
    } else {
      ROS_WARN("[target_mgr] target_command %s/%s track=%d score=%.3f "
               "margin=%.3f pos=[%.2f,%.2f,%.2f]",
               source.c_str(), CommandName(action), track.id,
               static_cast<double>(track.score),
               static_cast<double>(track.margin_to_best_other),
               static_cast<double>(track.center.x()),
               static_cast<double>(track.center.y()),
               static_cast<double>(track.center.z()));
    }
  }

  void EkfStatusCallback(
      const livox_reflective_marker::EkfStatus::ConstPtr& status) {
    last_ekf_orientation_ = status->current_pose.pose.orientation;
    has_last_ekf_orientation_ = true;

    if (status->state != 1) return;
    if (state_ == DecisionState::kRecognized) {
      ROS_WARN("[target_mgr] EKF reported LOST");
      tracking_lost_pub_.publish(std_msgs::Empty());
      EnterConfirming(status->header.stamp, "EKF_LOST");
      current_bad_run_ = cfg_.current_bad_frames;
    } else if (state_ == DecisionState::kConfirming) {
      current_bad_run_ = std::max(current_bad_run_, cfg_.current_bad_frames);
    }
  }

  void EnterSearching(const ros::Time& stamp, const std::string& reason) {
    tracks_.clear();
    active_track_id_ = -1;
    active_center_ = Eigen::Vector3f::Zero();
    audit_fail_windows_ = 0;
    current_bad_run_ = 0;
    SetState(DecisionState::kSearching, stamp, reason);
  }

  void EnterRecognized(int track_id, const Eigen::Vector3f& center,
                       const ros::Time& stamp, const std::string& reason) {
    active_track_id_ = track_id;
    active_center_ = center;
    audit_fail_windows_ = 0;
    current_bad_run_ = 0;
    next_audit_time_ =
        (stamp.isZero() ? ros::Time::now() : stamp) +
        ros::Duration(cfg_.audit_interval_sec);
    SetState(DecisionState::kRecognized, stamp, reason);
  }

  void EnterConfirming(const ros::Time& stamp, const std::string& reason) {
    confirm_start_time_ = stamp.isZero() ? ros::Time::now() : stamp;
    current_bad_run_ = 0;
    SetState(DecisionState::kConfirming, stamp, reason);
  }

  void SetState(DecisionState new_state, const ros::Time& stamp,
                const std::string& reason) {
    const DecisionState old_state = state_;
    state_ = new_state;
    state_enter_time_ = stamp.isZero() ? ros::Time::now() : stamp;
    PublishRecognitionConfig();

    if (old_state != new_state) {
      ROS_WARN("[target_mgr] state %s -> %s (%s)",
               StateName(old_state), StateName(new_state), reason.c_str());
    } else {
      ROS_INFO("[target_mgr] state %s refresh (%s)",
               StateName(state_), reason.c_str());
    }
  }

  void PublishRecognitionConfig() {
    livox_reflective_marker::RecognitionCommand cmd;
    cmd.header.stamp = ros::Time::now();

    switch (state_) {
      case DecisionState::kSearching:
        cmd.max_accumulation_frames = cfg_.search_accumulation_frames;
        cmd.publish_interval_sec = cfg_.search_publish_interval_sec;
        break;
      case DecisionState::kRecognized:
        cmd.max_accumulation_frames = cfg_.tracking_accumulation_frames;
        cmd.publish_interval_sec = cfg_.tracking_publish_interval_sec;
        break;
      case DecisionState::kConfirming:
        cmd.max_accumulation_frames = cfg_.confirm_accumulation_frames;
        cmd.publish_interval_sec = cfg_.confirm_publish_interval_sec;
        break;
    }

    recognition_cmd_pub_.publish(cmd);
    ROS_INFO("[target_mgr] recognition profile state=%s accum=%d interval=%.2f",
             StateName(state_), cmd.max_accumulation_frames,
             cmd.publish_interval_sec);
  }

  void ClusterCloudCallback(
      const livox_reflective_marker::ClusterCloud::ConstPtr& msg) {
    latest_cluster_cloud_ = msg;
    PublishColoredCloud();
  }

  void PublishColoredCloud() {
    if (!latest_cluster_cloud_ ||
        colored_cloud_pub_.getNumSubscribers() == 0) {
      return;
    }

    const auto& cc = *latest_cluster_cloud_;
    int target_cluster_idx = -1;
    if (active_track_id_ >= 0) {
      const float max_dist = cfg_.target_cluster_match_distance_m;
      float best_dist = max_dist;
      for (size_t i = 0; i < cc.cluster_centers.size(); ++i) {
        const float d = Distance(active_center_, ToEigen(cc.cluster_centers[i]));
        if (d < best_dist) {
          best_dist = d;
          target_cluster_idx = static_cast<int>(i);
        }
      }
    }

    const sensor_msgs::PointCloud2& in = cc.cloud;
    if (in.width == 0 || in.point_step < 12 || in.data.empty()) return;
    const size_t num_pts = static_cast<size_t>(in.width) *
                           static_cast<size_t>(std::max(1U, in.height));
    if (in.data.size() < num_pts * in.point_step) return;

    sensor_msgs::PointCloud2 out;
    out.header = cc.header;
    out.height = 1;
    out.is_bigendian = false;
    out.is_dense = true;

    sensor_msgs::PointField f;
    f.datatype = sensor_msgs::PointField::FLOAT32;
    f.count = 1;
    f.name = "x";
    f.offset = 0;
    out.fields.push_back(f);
    f.name = "y";
    f.offset = 4;
    out.fields.push_back(f);
    f.name = "z";
    f.offset = 8;
    out.fields.push_back(f);
    f.name = "rgb";
    f.offset = 12;
    out.fields.push_back(f);

    out.point_step = 16;
    out.width = static_cast<uint32_t>(num_pts);
    out.row_step = out.point_step * out.width;
    out.data.resize(out.row_step);

    const uint32_t white_rgb = (255U << 16) | (255U << 8) | 255U;
    const uint32_t yellow_rgb = (255U << 16) | (200U << 8) | 0U;
    const uint32_t green_rgb = (0U << 16) | (255U << 8) | 0U;
    const uint32_t red_rgb = (255U << 16) | (40U << 8) | 40U;
    const uint32_t gray_rgb = (120U << 16) | (120U << 8) | 120U;

    float* out_buf = reinterpret_cast<float*>(out.data.data());
    for (size_t i = 0; i < num_pts; ++i) {
      const size_t base = i * in.point_step;
      float x = 0.0f, y = 0.0f, z = 0.0f;
      std::memcpy(&x, in.data.data() + base + 0, sizeof(float));
      std::memcpy(&y, in.data.data() + base + 4, sizeof(float));
      std::memcpy(&z, in.data.data() + base + 8, sizeof(float));
      out_buf[i * 4 + 0] = x;
      out_buf[i * 4 + 1] = y;
      out_buf[i * 4 + 2] = z;

      const int32_t cid =
          i < cc.per_point_cluster_id.size() ? cc.per_point_cluster_id[i] : -1;
      uint32_t rgb = gray_rgb;
      if (cid >= 0) {
        switch (state_) {
          case DecisionState::kSearching:
            rgb = white_rgb;
            break;
          case DecisionState::kRecognized:
            rgb = (cid == target_cluster_idx) ? green_rgb : yellow_rgb;
            break;
          case DecisionState::kConfirming:
            rgb = red_rgb;
            break;
        }
      }
      float rgb_f = 0.0f;
      std::memcpy(&rgb_f, &rgb, sizeof(float));
      out_buf[i * 4 + 3] = rgb_f;
    }

    colored_cloud_pub_.publish(out);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  Config cfg_;

  ros::Subscriber bpu_scores_sub_;
  ros::Subscriber cluster_cloud_sub_;
  ros::Subscriber ekf_status_sub_;

  ros::Publisher target_cmd_pub_;
  ros::Publisher tracking_lost_pub_;
  ros::Publisher recognition_cmd_pub_;
  ros::Publisher colored_cloud_pub_;

  DecisionState state_ = DecisionState::kSearching;
  ros::Time state_enter_time_;
  ros::Time next_audit_time_;
  ros::Time confirm_start_time_;

  std::vector<Track> tracks_;
  int next_track_id_ = 0;
  uint32_t last_window_id_ = 0;

  int active_track_id_ = -1;
  Eigen::Vector3f active_center_ = Eigen::Vector3f::Zero();
  int audit_fail_windows_ = 0;
  int current_bad_run_ = 0;
  ros::Time last_target_command_time_;

  geometry_msgs::Quaternion last_ekf_orientation_;
  bool has_last_ekf_orientation_ = false;

  livox_reflective_marker::ClusterCloud::ConstPtr latest_cluster_cloud_;

};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "target_manager_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  TargetManager node(nh, pnh);
  ros::spin();
  return 0;
}
