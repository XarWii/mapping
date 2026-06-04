/// @file target_manager_node.cpp
/// @brief Target management state machine — arbitrates between BPU and
/// traditional scorer, decides when to INIT / CORRECT / LOST the EKF.

#include <ros/ros.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Empty.h>

#include <livox_reflective_marker/BpuScores.h>
#include <livox_reflective_marker/CandidateValidationRequest.h>
#include <livox_reflective_marker/CandidateValidationResult.h>
#include <livox_reflective_marker/TargetCommand.h>
#include <livox_reflective_marker/EkfStatus.h>

#include <Eigen/Dense>

#include <algorithm>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

enum class TrackState { kUnconfirmed, kConfirmed, kLost };

struct Track {
  int id = 0;
  TrackState state = TrackState::kUnconfirmed;
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  float score = 0.0f;
  int consecutive_target = 0;
  int consecutive_interference = 0;
  ros::Time last_seen;
  ros::Time first_seen;
};

enum class ManagerState { kDiscover, kWaitValidation, kTracking };

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct Config {
  // Inputs
  std::string bpu_scores_topic = "/siamese_bpu_infer_node/bpu_scores";
  std::string validation_result_topic = "/siamese_bpu_infer_node/validation_result";
  std::string ekf_status_topic = "/ekf_pose_node/ekf_status";

  // Outputs
  std::string target_command_topic = "/target_manager/target_command";
  std::string validation_request_topic = "/siamese_bpu_infer_node/validation_request";
  std::string tracking_lost_topic = "/siamese_bpu_infer_node/tracking_lost";

  std::string frame_id = "livox_frame";

  // BPU scoring
  float score_threshold = 0.3825f;

  // Tracking
  int confirm_frames = 5;
  int lost_frames = 3;
  float track_match_distance_m = 0.5f;
  double track_timeout_sec = 2.0;
  double init_score_margin = 0.05;

  // Validation
  bool use_traditional_validation = true;
  double validation_timeout_sec = 3.0;
  double validation_accumulation_sec = 1.5;
  float validation_roi_radius_m = 0.45f;

  int max_tracks = 8;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float SquaredDistance3(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
  return (a - b).squaredNorm();
}

std::vector<int> AssociateClustersToTracks(
    const std::vector<livox_reflective_marker::BpuScoreEntry>& entries,
    const std::vector<Track>& tracks, float match_dist_sq) {
  std::vector<int> assignments(entries.size(), -1);
  if (tracks.empty()) return assignments;

  struct Candidate { int e; int t; float d2; };
  std::vector<Candidate> cands;
  for (size_t ei = 0; ei < entries.size(); ++ei) {
    Eigen::Vector3f ec(entries[ei].center.x, entries[ei].center.y, entries[ei].center.z);
    for (size_t ti = 0; ti < tracks.size(); ++ti) {
      float d2 = SquaredDistance3(ec, tracks[ti].center);
      if (d2 <= match_dist_sq)
        cands.push_back({static_cast<int>(ei), static_cast<int>(ti), d2});
    }
  }
  std::sort(cands.begin(), cands.end(),
            [](const Candidate& a, const Candidate& b) { return a.d2 < b.d2; });

  std::vector<bool> used(tracks.size(), false);
  for (const auto& c : cands) {
    if (assignments[c.e] >= 0 || used[c.t]) continue;
    assignments[c.e] = c.t;
    used[c.t] = true;
  }
  return assignments;
}

// ---------------------------------------------------------------------------
// Target Manager
// ---------------------------------------------------------------------------

class TargetManager {
 public:
  TargetManager(ros::NodeHandle nh, ros::NodeHandle pnh) : nh_(nh), pnh_(pnh) {
    LoadConfig();

    // Publishers
    target_cmd_pub_ = nh_.advertise<livox_reflective_marker::TargetCommand>(
        cfg_.target_command_topic, 4, true);
    tracking_lost_pub_ = nh_.advertise<std_msgs::Empty>(
        cfg_.tracking_lost_topic, 4);
    validation_request_pub_ =
        nh_.advertise<livox_reflective_marker::CandidateValidationRequest>(
            cfg_.validation_request_topic, 4, true);

    // Subscribers
    bpu_scores_sub_ = nh_.subscribe<livox_reflective_marker::BpuScores>(
        cfg_.bpu_scores_topic, 4, &TargetManager::BpuScoresCallback, this);
    validation_result_sub_ =
        nh_.subscribe<livox_reflective_marker::CandidateValidationResult>(
            cfg_.validation_result_topic, 4,
            &TargetManager::ValidationResultCallback, this);
    ekf_status_sub_ = nh_.subscribe<livox_reflective_marker::EkfStatus>(
        cfg_.ekf_status_topic, 4, &TargetManager::EkfStatusCallback, this);

    ROS_INFO("[target_mgr] ready | champion mode | confirm=%d lost=%d "
             "thresh=%.3f score_margin=%.3f",
             cfg_.confirm_frames, cfg_.lost_frames,
             static_cast<double>(cfg_.score_threshold),
             cfg_.init_score_margin);
  }

 private:
  void LoadConfig() {
#define LOAD(ns, name, var) ns.param(name, var, var)
    LOAD(pnh_, "bpu_scores_topic", cfg_.bpu_scores_topic);
    LOAD(pnh_, "validation_result_topic", cfg_.validation_result_topic);
    LOAD(pnh_, "ekf_status_topic", cfg_.ekf_status_topic);
    LOAD(pnh_, "target_command_topic", cfg_.target_command_topic);
    LOAD(pnh_, "validation_request_topic", cfg_.validation_request_topic);
    LOAD(pnh_, "tracking_lost_topic", cfg_.tracking_lost_topic);
    LOAD(pnh_, "frame_id", cfg_.frame_id);
    LOAD(pnh_, "score_threshold", cfg_.score_threshold);
    LOAD(pnh_, "confirm_frames", cfg_.confirm_frames);
    LOAD(pnh_, "lost_frames", cfg_.lost_frames);
    LOAD(pnh_, "track_match_distance_m", cfg_.track_match_distance_m);
    LOAD(pnh_, "track_timeout_sec", cfg_.track_timeout_sec);
    LOAD(pnh_, "init_score_margin", cfg_.init_score_margin);
    LOAD(pnh_, "use_traditional_validation", cfg_.use_traditional_validation);
    LOAD(pnh_, "validation_timeout_sec", cfg_.validation_timeout_sec);
    LOAD(pnh_, "validation_accumulation_sec", cfg_.validation_accumulation_sec);
    LOAD(pnh_, "validation_roi_radius_m", cfg_.validation_roi_radius_m);
    LOAD(pnh_, "max_tracks", cfg_.max_tracks);
#undef LOAD
  }

  // ---- BPU scores → track management + audit ---------------------------

  void BpuScoresCallback(
      const livox_reflective_marker::BpuScores::ConstPtr& scores) {
    ros::Time now = scores->header.stamp;

    if (manager_state_ == ManagerState::kWaitValidation) {
      // Check timeout
      if (!validation_request_time_.isZero() &&
          (now - validation_request_time_).toSec() > cfg_.validation_timeout_sec) {
        if (pending_validation_mode_ == 1 && target_was_accepted_) {
          ROS_WARN("[target_mgr] correction validation timeout, keep TRACKING");
          ClearPendingValidation();
          ResetStableChampion();
          manager_state_ = ManagerState::kTracking;
        } else {
          ROS_WARN("[target_mgr] validation timeout, back to DISCOVER");
          ResetToDiscover();
        }
      }
      return;
    }

    // Update tracks from scores
    UpdateTracks(scores->entries, now);

    Track* runner_up = nullptr;
    Track* best = FindBestObservedTrack(now, &runner_up);
    Track* champion = UpdateStableChampion(best, runner_up);
    if (!champion) {
      return;
    }

    const uint8_t action = target_was_accepted_ ? 1 : 0;
    if (action == 0) {
      IssueTargetCommand(*champion, 0, now, "CHAMPION_INIT");
      return;
    }

    if (manager_state_ == ManagerState::kDiscover) {
      IssueTargetCommand(*champion, 1, now, "CHAMPION_REACQUIRE");
      return;
    }

    Track* active = FindTrackById(active_track_id_);
    if (!active || active->state != TrackState::kConfirmed) {
      IssueTargetCommand(*champion, 1, now, "CHAMPION_CORRECT_NO_ACTIVE");
      return;
    }

    if (champion->id == active->id) {
      return;
    }

    ROS_WARN("[target_mgr] champion switch track %d -> %d "
             "score %.3f -> %.3f stable=%d/%d",
             active->id, champion->id,
             static_cast<double>(active->score),
             static_cast<double>(champion->score),
             stable_champion_hits_, cfg_.confirm_frames);
    IssueTargetCommand(*champion, 1, now, "CHAMPION_SWITCH");
  }

  // ---- Track state machine ---------------------------------------------

  void UpdateTracks(
      const std::vector<livox_reflective_marker::BpuScoreEntry>& entries,
      ros::Time now) {
    const float match_sq =
        cfg_.track_match_distance_m * cfg_.track_match_distance_m;

    // Prune timed-out tracks
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [&](const Track& t) {
                         return (now - t.last_seen).toSec() > cfg_.track_timeout_sec;
                       }),
        tracks_.end());

    auto assignments = AssociateClustersToTracks(entries, tracks_, match_sq);

    std::vector<size_t> entry_order;
    entry_order.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) entry_order.push_back(i);
    std::sort(entry_order.begin(), entry_order.end(),
              [&](size_t a, size_t b) {
                return entries[a].score > entries[b].score;
              });

    for (size_t ei : entry_order) {
      float s = entries[ei].score;
      bool is_target = s >= cfg_.score_threshold;

      if (assignments[ei] >= 0) {
        auto& t = tracks_[assignments[ei]];
        t.center = Eigen::Vector3f(entries[ei].center.x, entries[ei].center.y,
                                   entries[ei].center.z);
        t.score = s;
        t.last_seen = now;

        if (is_target) {
          t.consecutive_target++;
          t.consecutive_interference = 0;
        } else {
          t.consecutive_interference++;
          t.consecutive_target = 0;
        }

        if (t.state == TrackState::kUnconfirmed &&
            t.consecutive_target >= cfg_.confirm_frames) {
          t.state = TrackState::kConfirmed;
          ROS_WARN("[target_mgr] track %d CONFIRMED score=%.3f",
                   t.id, static_cast<double>(t.score));
        } else if (t.state == TrackState::kConfirmed &&
                   t.consecutive_interference >= cfg_.lost_frames) {
          t.state = TrackState::kLost;
        }
      } else if (static_cast<int>(tracks_.size()) < cfg_.max_tracks) {
        Track t;
        t.id = next_track_id_++;
        t.state = TrackState::kUnconfirmed;
        t.center = Eigen::Vector3f(entries[ei].center.x, entries[ei].center.y,
                                   entries[ei].center.z);
        t.score = s;
        t.consecutive_target = is_target ? 1 : 0;
        t.consecutive_interference = is_target ? 0 : 1;
        t.first_seen = now;
        t.last_seen = now;
        tracks_.push_back(t);
      }
    }

    // Unmatched tracks
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
      bool matched = false;
      for (int a : assignments)
        if (a == static_cast<int>(ti)) { matched = true; break; }
      if (!matched) {
        auto& t = tracks_[ti];
        t.consecutive_target = 0;
        t.consecutive_interference++;
        if (t.state == TrackState::kConfirmed &&
            t.consecutive_interference >= cfg_.lost_frames)
          t.state = TrackState::kLost;
      }
    }
  }

  // ---- Validation ------------------------------------------------------

  void RequestValidation(const Track& track, uint8_t mode) {
    uint32_t proposal_id = ++next_proposal_id_;
    pending_proposal_id_ = proposal_id;
    pending_track_id_ = track.id;
    validation_request_time_ = ros::Time::now();

    livox_reflective_marker::CandidateValidationRequest req;
    req.header.stamp = validation_request_time_;
    req.header.frame_id = cfg_.frame_id;
    req.proposal_id = proposal_id;
    req.mode = mode;
    req.center.x = track.center.x();
    req.center.y = track.center.y();
    req.center.z = track.center.z();
    req.roi_radius = cfg_.validation_roi_radius_m;
    req.accumulation_sec = static_cast<float>(cfg_.validation_accumulation_sec);
    req.seed_cloud = sensor_msgs::PointCloud2{};
    req.seed_cloud.header = req.header;

    if (validation_request_pub_.getNumSubscribers() == 0) {
      ROS_WARN("[target_mgr] no validation subscriber; request will timeout");
    }
    validation_request_pub_.publish(req);

    manager_state_ = ManagerState::kWaitValidation;
    pending_validation_mode_ = mode;

    ROS_INFO("[target_mgr] validation (%s) requested proposal=%u pos=[%.2f,%.2f,%.2f]",
             mode == 0 ? "INIT" : "CORRECTION", proposal_id,
             static_cast<double>(track.center.x()),
             static_cast<double>(track.center.y()),
             static_cast<double>(track.center.z()));
  }

  Track* FindBestObservedTrack(const ros::Time& now, Track** runner_up) {
    Track* best = nullptr;
    Track* second = nullptr;
    for (auto& t : tracks_) {
      if (!WasObservedThisFrame(t, now)) continue;
      if (!best || t.score > best->score) {
        second = best;
        best = &t;
      } else if (!second || t.score > second->score) {
        second = &t;
      }
    }
    if (runner_up) *runner_up = second;
    return best;
  }

  Track* UpdateStableChampion(Track* best, Track* runner_up) {
    if (!best || best->score < cfg_.score_threshold) {
      ResetStableChampion();
      return nullptr;
    }

    const float second_score = runner_up ? runner_up->score : 0.0f;
    const float margin = best->score - second_score;
    if (runner_up && margin < cfg_.init_score_margin) {
      ROS_INFO_THROTTLE(
          0.5,
          "[target_mgr] champion not stable: best track=%d score=%.3f "
          "second track=%d score=%.3f margin=%.3f < %.3f",
          best->id, static_cast<double>(best->score),
          runner_up->id, static_cast<double>(runner_up->score),
          static_cast<double>(margin), cfg_.init_score_margin);
      ResetStableChampion();
      return nullptr;
    }

    if (stable_champion_id_ != best->id) {
      stable_champion_id_ = best->id;
      stable_champion_hits_ = 1;
    } else {
      stable_champion_hits_++;
    }

    const int required = std::max(1, cfg_.confirm_frames);
    if (stable_champion_hits_ < required) {
      ROS_INFO_THROTTLE(
          0.5,
          "[target_mgr] champion candidate track=%d stable=%d/%d "
          "score=%.3f second=%.3f margin=%.3f",
          best->id, stable_champion_hits_, required,
          static_cast<double>(best->score),
          static_cast<double>(second_score),
          static_cast<double>(margin));
      return nullptr;
    }

    if (best->state != TrackState::kConfirmed) {
      best->state = TrackState::kConfirmed;
      ROS_WARN("[target_mgr] champion track %d CONFIRMED stable=%d score=%.3f",
               best->id, stable_champion_hits_,
               static_cast<double>(best->score));
    }
    return best;
  }

  bool WasObservedThisFrame(const Track& track, const ros::Time& now) const {
    if (now.isZero() || track.last_seen.isZero()) return true;
    return (now - track.last_seen).toSec() <= 1e-3;
  }

  void ResetStableChampion() {
    stable_champion_id_ = -1;
    stable_champion_hits_ = 0;
  }

  Track* FindTrackById(int id) {
    if (id < 0) return nullptr;
    for (auto& t : tracks_) {
      if (t.id == id) return &t;
    }
    return nullptr;
  }

  void IssueTargetCommand(Track& track, uint8_t action,
                          const ros::Time& stamp,
                          const char* source) {
    if (cfg_.use_traditional_validation) {
      RequestValidation(track, action);
    } else {
      PublishCommandFromTrack(track, action, stamp, source);
    }
  }

  void PublishCommandFromTrack(Track& track, uint8_t action,
                               const ros::Time& stamp,
                               const char* source) {
    ros::Time cmd_stamp = stamp.isZero() ? ros::Time::now() : stamp;

    livox_reflective_marker::TargetCommand cmd;
    cmd.header.stamp = cmd_stamp;
    cmd.header.frame_id = cfg_.frame_id;
    cmd.action = action;
    cmd.pose.header = cmd.header;
    cmd.pose.pose.position.x = track.center.x();
    cmd.pose.pose.position.y = track.center.y();
    cmd.pose.pose.position.z = track.center.z();
    if (has_last_ekf_orientation_) {
      cmd.pose.pose.orientation = last_ekf_orientation_;
    } else {
      cmd.pose.pose.orientation.w = 1.0;
    }
    cmd.init_cloud = sensor_msgs::PointCloud2{};
    cmd.init_cloud.header = cmd.header;
    target_cmd_pub_.publish(cmd);

    track.state = TrackState::kConfirmed;
    track.consecutive_interference = 0;
    track.last_seen = cmd_stamp;
    active_track_id_ = track.id;

    target_was_accepted_ = true;
    manager_state_ = ManagerState::kTracking;
    ClearPendingValidation();

    ROS_WARN("[target_mgr] direct target_command (%s/%s) score=%.3f hits=%d pos=[%.2f,%.2f,%.2f]",
             source, action == 0 ? "INIT" : "CORRECT",
             static_cast<double>(track.score), track.consecutive_target,
             static_cast<double>(track.center.x()),
             static_cast<double>(track.center.y()),
             static_cast<double>(track.center.z()));
    ROS_INFO("[target_mgr] -> TRACKING");
  }

  void ValidationResultCallback(
      const livox_reflective_marker::CandidateValidationResult::ConstPtr& result) {
    if (result->proposal_id != pending_proposal_id_) {
      ROS_DEBUG("[target_mgr] stale validation result %u != %u",
                result->proposal_id, pending_proposal_id_);
      return;
    }

    const char* mode_str = result->mode == 0 ? "INIT" : "CORRECTION";
    const int accepted_track_id = pending_track_id_;
    pending_proposal_id_ = 0;
    pending_track_id_ = -1;
    validation_request_time_ = ros::Time();

    if (result->accepted) {
      ROS_WARN("[target_mgr] validation (%s) ACCEPTED score=%.3f",
               mode_str, static_cast<double>(result->score));

      livox_reflective_marker::TargetCommand cmd;
      cmd.header = result->header;
      cmd.action = result->mode;  // 0=INIT, 1=CORRECT
      cmd.pose = result->pose;
      cmd.init_cloud = result->roi_cloud;
      target_cmd_pub_.publish(cmd);

      target_was_accepted_ = true;

      const Eigen::Vector3f accepted_center(
          static_cast<float>(result->pose.pose.position.x),
          static_cast<float>(result->pose.pose.position.y),
          static_cast<float>(result->pose.pose.position.z));

      Track* accepted_track = FindTrackById(accepted_track_id);
      float best_d2 = 0.0f;
      if (!accepted_track) {
        for (auto& t : tracks_) {
          const float d2 = SquaredDistance3(t.center, accepted_center);
          if (!accepted_track || d2 < best_d2) {
            accepted_track = &t;
            best_d2 = d2;
          }
        }
      }
      if (accepted_track) {
        accepted_track->state = TrackState::kConfirmed;
        accepted_track->center = accepted_center;
        accepted_track->score = result->score;
        accepted_track->last_seen = result->header.stamp;
        active_track_id_ = accepted_track->id;
      }

      manager_state_ = ManagerState::kTracking;
      ROS_INFO("[target_mgr] → TRACKING");
    } else {
      ROS_WARN("[target_mgr] validation (%s) REJECTED: %s",
               mode_str, result->reason.c_str());

      if (result->mode == 0) {
        ResetToDiscover();
      } else {
        ResetStableChampion();
        manager_state_ = ManagerState::kTracking;
      }
    }
  }

  // ---- EKF feedback ----------------------------------------------------

  void EkfStatusCallback(
      const livox_reflective_marker::EkfStatus::ConstPtr& status) {
    last_ekf_orientation_ = status->current_pose.pose.orientation;
    has_last_ekf_orientation_ = true;

    if (status->state == 1) {  // LOST
      if (manager_state_ == ManagerState::kTracking) {
        ROS_WARN("[target_mgr] EKF LOST — back to champion search");
        tracking_lost_pub_.publish(std_msgs::Empty());
        ResetToDiscover();
      }
    }
  }

  // ---- State helpers ---------------------------------------------------

  void ResetToDiscover() {
    manager_state_ = ManagerState::kDiscover;
    tracks_.clear();
    active_track_id_ = -1;
    ResetStableChampion();
    ClearPendingValidation();
    ROS_INFO("[target_mgr] → DISCOVER");
  }

  void ClearPendingValidation() {
    pending_proposal_id_ = 0;
    pending_track_id_ = -1;
    validation_request_time_ = ros::Time();
    pending_validation_mode_ = 0;
  }

  // ---- Members ---------------------------------------------------------

  ros::NodeHandle nh_, pnh_;
  Config cfg_;

  ros::Subscriber bpu_scores_sub_;
  ros::Subscriber validation_result_sub_;
  ros::Subscriber ekf_status_sub_;

  ros::Publisher target_cmd_pub_;
  ros::Publisher tracking_lost_pub_;
  ros::Publisher validation_request_pub_;

  ManagerState manager_state_ = ManagerState::kDiscover;
  std::vector<Track> tracks_;
  int next_track_id_ = 0;
  int active_track_id_ = -1;
  int stable_champion_id_ = -1;
  int stable_champion_hits_ = 0;

  uint32_t next_proposal_id_ = 0;
  uint32_t pending_proposal_id_ = 0;
  int pending_track_id_ = -1;
  uint8_t  pending_validation_mode_ = 0;
  ros::Time validation_request_time_;

  bool target_was_accepted_ = false;

  geometry_msgs::Quaternion last_ekf_orientation_;
  bool has_last_ekf_orientation_ = false;
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
