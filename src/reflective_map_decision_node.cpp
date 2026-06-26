#include <geometry_msgs/PointStamped.h>
#include <livox_reflective_marker/RecognitionScoreEntry.h>
#include <livox_reflective_marker/RecognitionScores.h>
#include <livox_reflective_marker/ReflectiveCandidate.h>
#include <livox_reflective_marker/ReflectiveCandidates.h>
#include <livox_reflective_marker/ReflectiveRecognitionRequest.h>
#include <livox_reflective_marker/ReflectiveTargetState.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr uint8_t kSearching = 0;
constexpr uint8_t kTracking = 1;
constexpr uint8_t kLost = 2;

constexpr uint8_t kNewCandidate = 1;
constexpr uint8_t kActiveTargetDisappeared = 2;
constexpr uint8_t kCandidateBecameScoreable = 3;
constexpr uint8_t kRetryTimedOut = 4;
constexpr uint8_t kCandidateGrew = 5;
constexpr uint8_t kScoreStatusScored = 0;

struct Config {
  std::string candidates_topic = "/reflective/candidates";
  std::string scores_topic = "/reflective/recognition_scores";
  std::string recognition_request_topic = "/reflective/recognition_request";
  std::string target_state_topic = "/reflective/target_state";
  std::string active_candidate_topic = "/reflective/active_candidate";
  std::string target_center_topic = "/reflective/target_center";
  std::string target_marker_topic = "/reflective/target_marker";
  float selection_score_threshold = -0.004310191f;
  float selection_margin_threshold = 0.015f;
  float single_candidate_virtual_second_score = -0.004310191f;
  int min_scored_candidates_for_selection = 2;
  int min_candidate_points_for_request = 8;
  int rescore_min_added_points = 8;
  double request_timeout_sec = 1.0;
};

bool SameSnapshot(const std_msgs::Header& first, uint32_t first_epoch,
                  const std_msgs::Header& second, uint32_t second_epoch) {
  return first_epoch == second_epoch && first.stamp == second.stamp &&
         first.frame_id == second.frame_id;
}

const char* ReasonName(uint8_t reason) {
  switch (reason) {
    case kNewCandidate:
      return "new_candidate";
    case kActiveTargetDisappeared:
      return "active_target_disappeared";
    case kCandidateBecameScoreable:
      return "candidate_became_scoreable";
    case kRetryTimedOut:
      return "retry_timed_out";
    case kCandidateGrew:
      return "candidate_grew";
  }
  return "unknown";
}

class ReflectiveMapDecision {
 public:
  ReflectiveMapDecision() : private_nh_("~"), config_(LoadConfig()) {
    if (config_.min_scored_candidates_for_selection < 1 ||
        config_.min_candidate_points_for_request < 1 ||
        config_.rescore_min_added_points < 1 ||
        config_.request_timeout_sec <= 0.0f) {
      throw std::runtime_error("invalid reflective_map_decision parameters");
    }
    candidates_subscriber_ = nh_.subscribe(
        config_.candidates_topic, 2, &ReflectiveMapDecision::HandleCandidates,
        this, ros::TransportHints().tcpNoDelay());
    scores_subscriber_ = nh_.subscribe(
        config_.scores_topic, 2, &ReflectiveMapDecision::HandleScores, this,
        ros::TransportHints().tcpNoDelay());
    request_publisher_ =
        nh_.advertise<livox_reflective_marker::ReflectiveRecognitionRequest>(
            config_.recognition_request_topic, 2);
    target_state_publisher_ =
        nh_.advertise<livox_reflective_marker::ReflectiveTargetState>(
            config_.target_state_topic, 1, true);
    active_candidate_publisher_ =
        nh_.advertise<livox_reflective_marker::ReflectiveCandidate>(
            config_.active_candidate_topic, 1, true);
    target_center_publisher_ = nh_.advertise<geometry_msgs::PointStamped>(
        config_.target_center_topic, 1, true);
    target_marker_publisher_ = nh_.advertise<visualization_msgs::Marker>(
        config_.target_marker_topic, 1, true);
    timeout_timer_ = private_nh_.createTimer(
        ros::Duration(0.1), &ReflectiveMapDecision::HandleTimeout, this);
    ROS_INFO("reflective map decision: candidates=%s scores=%s request=%s",
             config_.candidates_topic.c_str(), config_.scores_topic.c_str(),
             config_.recognition_request_topic.c_str());
  }

 private:
  Config LoadConfig() {
    Config config;
    private_nh_.param("candidates_topic", config.candidates_topic,
                      config.candidates_topic);
    private_nh_.param("scores_topic", config.scores_topic, config.scores_topic);
    private_nh_.param("recognition_request_topic", config.recognition_request_topic,
                      config.recognition_request_topic);
    private_nh_.param("target_state_topic", config.target_state_topic,
                      config.target_state_topic);
    private_nh_.param("active_candidate_topic", config.active_candidate_topic,
                      config.active_candidate_topic);
    private_nh_.param("target_center_topic", config.target_center_topic,
                      config.target_center_topic);
    private_nh_.param("target_marker_topic", config.target_marker_topic,
                      config.target_marker_topic);
    private_nh_.param("selection_score_threshold", config.selection_score_threshold,
                      config.selection_score_threshold);
    private_nh_.param("selection_margin_threshold", config.selection_margin_threshold,
                      config.selection_margin_threshold);
    private_nh_.param("single_candidate_virtual_second_score",
                      config.single_candidate_virtual_second_score,
                      config.single_candidate_virtual_second_score);
    private_nh_.param("min_scored_candidates_for_selection",
                      config.min_scored_candidates_for_selection,
                      config.min_scored_candidates_for_selection);
    private_nh_.param("min_candidate_points_for_request",
                      config.min_candidate_points_for_request,
                      config.min_candidate_points_for_request);
    private_nh_.param("rescore_min_added_points", config.rescore_min_added_points,
                      config.rescore_min_added_points);
    private_nh_.param("request_timeout_sec", config.request_timeout_sec,
                      config.request_timeout_sec);
    return config;
  }

  static size_t CandidatePointCount(
      const livox_reflective_marker::ReflectiveCandidate& candidate) {
    return static_cast<size_t>(candidate.cloud_sensor.width) *
           candidate.cloud_sensor.height;
  }

  void ResetForEpoch(const std_msgs::Header& header, uint32_t map_epoch) {
    PublishTargetMarkerDelete(header);
    have_epoch_ = true;
    map_epoch_ = map_epoch;
    previous_candidate_ids_.clear();
    scoreable_candidate_ids_.clear();
    last_requested_point_counts_.clear();
    active_candidate_id_ = 0;
    active_score_ = 0.0f;
    pending_request_ = false;
    queued_request_ = false;
    PublishState(header, kSearching, 0, 0.0f, "map_epoch_reset");
    ROS_INFO("reflective map decision entered epoch %u", map_epoch_);
  }

  void HandleCandidates(
      const livox_reflective_marker::ReflectiveCandidates::ConstPtr& candidates) {
    if (!have_epoch_ || candidates->map_epoch != map_epoch_) {
      ResetForEpoch(candidates->header, candidates->map_epoch);
    }

    latest_candidates_ = *candidates;
    have_candidates_ = true;
    std::unordered_set<uint32_t> current_ids;
    current_ids.reserve(candidates->candidates.size());
    bool new_scoreable_candidate = false;
    bool candidate_became_scoreable = false;
    bool candidate_grew_after_request = false;

    for (const auto& candidate : candidates->candidates) {
      const uint32_t id = candidate.candidate_id;
      current_ids.insert(id);
      const bool is_new = previous_candidate_ids_.count(id) == 0;
      const bool scoreable = CandidatePointCount(candidate) >=
                             static_cast<size_t>(config_.min_candidate_points_for_request);
      if (!scoreable) continue;
      if (is_new) {
        new_scoreable_candidate = true;
      } else if (scoreable_candidate_ids_.count(id) == 0) {
        candidate_became_scoreable = true;
      }
      scoreable_candidate_ids_.insert(id);
      const auto requested = last_requested_point_counts_.find(id);
      if (requested != last_requested_point_counts_.end() &&
          CandidatePointCount(candidate) >=
              requested->second + static_cast<size_t>(config_.rescore_min_added_points)) {
        candidate_grew_after_request = true;
      }
    }
    for (auto it = scoreable_candidate_ids_.begin();
         it != scoreable_candidate_ids_.end();) {
      if (current_ids.count(*it) == 0) {
        last_requested_point_counts_.erase(*it);
        it = scoreable_candidate_ids_.erase(it);
      } else {
        ++it;
      }
    }

    const bool active_disappeared =
        active_candidate_id_ != 0 && current_ids.count(active_candidate_id_) == 0;
    if (active_disappeared) {
      const uint32_t lost_id = active_candidate_id_;
      active_candidate_id_ = 0;
      active_score_ = 0.0f;
      PublishState(candidates->header, kLost, lost_id, 0.0f,
                   "active_candidate_disappeared");
      PublishTargetMarkerDelete(candidates->header);
      RequestCurrentSnapshot(kActiveTargetDisappeared);
    } else if (new_scoreable_candidate) {
      RequestCurrentSnapshot(kNewCandidate);
    } else if (candidate_became_scoreable) {
      RequestCurrentSnapshot(kCandidateBecameScoreable);
    } else if (active_candidate_id_ == 0 && candidate_grew_after_request) {
      RequestCurrentSnapshot(kCandidateGrew);
    }

    previous_candidate_ids_ = std::move(current_ids);
    PublishActiveTargetIfPresent();
  }

  void RequestCurrentSnapshot(uint8_t reason) {
    if (!have_candidates_ || latest_candidates_.candidates.empty()) return;
    if (pending_request_) {
      queued_request_ = true;
      queued_reason_ = reason;
      return;
    }
	    pending_request_ = true;
	    pending_header_ = latest_candidates_.header;
	    pending_epoch_ = latest_candidates_.map_epoch;
	    pending_request_id_ = ++next_request_id_;
	    pending_candidate_snapshot_id_ = latest_candidates_.candidate_snapshot_id;
	    pending_candidates_ = latest_candidates_;
    pending_time_ = ros::Time::now();
    for (const auto& candidate : pending_candidates_.candidates) {
      last_requested_point_counts_[candidate.candidate_id] =
          CandidatePointCount(candidate);
    }

	    livox_reflective_marker::ReflectiveRecognitionRequest request;
	    request.header = pending_header_;
	    request.map_epoch = pending_epoch_;
	    request.request_id = pending_request_id_;
	    request.candidate_snapshot_id = pending_candidate_snapshot_id_;
	    request.reason = reason;
	    request.candidate_ids.reserve(pending_candidates_.candidates.size());
	    request.support_revisions.reserve(pending_candidates_.candidates.size());
	    for (const auto& candidate : pending_candidates_.candidates) {
	      request.candidate_ids.push_back(candidate.candidate_id);
	      request.support_revisions.push_back(candidate.support_revision);
	    }
	    request_publisher_.publish(request);
	    ROS_INFO("reflective map decision requested BPU request=%llu snapshot=%llu "
	             "epoch=%u candidates=%zu reason=%s",
	             static_cast<unsigned long long>(pending_request_id_),
	             static_cast<unsigned long long>(pending_candidate_snapshot_id_),
	             pending_epoch_, pending_candidates_.candidates.size(), ReasonName(reason));
	  }

  void HandleScores(const livox_reflective_marker::RecognitionScores::ConstPtr& scores) {
	    if (!pending_request_ ||
	        scores->request_id != pending_request_id_ ||
	        scores->candidate_snapshot_id != pending_candidate_snapshot_id_ ||
	        !SameSnapshot(pending_header_, pending_epoch_, scores->header,
	                      scores->map_epoch)) {
      ROS_DEBUG_THROTTLE(2.0,
                         "reflective map decision ignored scores for an unrequested snapshot");
      return;
    }
    pending_request_ = false;
    const int selected_index = SelectCandidate(*scores);
    if (active_candidate_id_ == 0 && selected_index >= 0) {
      const auto& selected = scores->entries[static_cast<size_t>(selected_index)];
      SelectActiveTarget(selected, "selected_from_bpu_batch");
    } else if (active_candidate_id_ != 0) {
      bool saw_active = false;
	      for (const auto& entry : scores->entries) {
	        if (entry.candidate_id == active_candidate_id_ &&
	            entry.score_valid && entry.status == kScoreStatusScored) {
	          saw_active = true;
	          active_score_ = entry.score;
	          break;
        }
      }
      if (!saw_active || active_score_ <= config_.selection_score_threshold) {
        const uint32_t old_id = active_candidate_id_;
        active_candidate_id_ = 0;
        active_score_ = 0.0f;
        PublishState(scores->header, kLost, old_id, 0.0f,
                     saw_active ? "active_score_rejected" : "active_score_missing");
        PublishTargetMarkerDelete(scores->header);
        if (selected_index >= 0) {
          const auto& selected = scores->entries[static_cast<size_t>(selected_index)];
          SelectActiveTarget(selected, "reselected_from_bpu_batch");
        }
      }
    } else {
      PublishState(scores->header, kSearching, 0, 0.0f,
                   "no_candidate_passed_score_and_margin");
    }

    if (queued_request_) {
      const uint8_t reason = queued_reason_;
      queued_request_ = false;
      RequestCurrentSnapshot(reason);
    }
  }

	  int SelectCandidate(const livox_reflective_marker::RecognitionScores& scores) const {
	    std::vector<size_t> valid_entries;
	    valid_entries.reserve(scores.entries.size());
	    for (size_t index = 0; index < scores.entries.size(); ++index) {
	      const auto& entry = scores.entries[index];
	      if (entry.score_valid && entry.status == kScoreStatusScored) {
	        valid_entries.push_back(index);
	      }
	    }
	    if (valid_entries.empty()) return -1;
	    if (valid_entries.size() <
	        static_cast<size_t>(config_.min_scored_candidates_for_selection)) {
	      ROS_INFO("reflective map decision deferred batch valid_entries=%zu need=%d",
	               valid_entries.size(), config_.min_scored_candidates_for_selection);
	      return -1;
	    }
	    std::vector<size_t> order = valid_entries;
	    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
	      return scores.entries[left].score > scores.entries[right].score;
    });
    const auto& best = scores.entries[order.front()];
    const float second_score = order.size() > 1
                                   ? scores.entries[order[1]].score
                                   : config_.single_candidate_virtual_second_score;
    const float margin = best.score - second_score;
    if (best.score <= config_.selection_score_threshold ||
        margin < config_.selection_margin_threshold) {
      ROS_INFO("reflective map decision rejected batch best=%u score=%.4f margin=%.4f",
               best.candidate_id, static_cast<double>(best.score),
               static_cast<double>(margin));
      return -1;
    }
    return static_cast<int>(order.front());
  }

  void SelectActiveTarget(
      const livox_reflective_marker::RecognitionScoreEntry& selected,
      const std::string& reason) {
    active_candidate_id_ = selected.candidate_id;
    active_score_ = selected.score;
    ROS_INFO("reflective map decision selected target=%u score=%.4f epoch=%u",
             active_candidate_id_, static_cast<double>(active_score_), map_epoch_);
    PublishActiveTargetFromSnapshot(selected, reason);
  }

  const livox_reflective_marker::ReflectiveCandidate* FindCandidate(
      const livox_reflective_marker::ReflectiveCandidates& candidates,
      uint32_t candidate_id) const {
    for (const auto& candidate : candidates.candidates) {
      if (candidate.candidate_id == candidate_id) return &candidate;
    }
    return nullptr;
  }

  void PublishActiveTargetFromSnapshot(
      const livox_reflective_marker::RecognitionScoreEntry& score,
      const std::string& reason) {
    const auto* candidate = FindCandidate(pending_candidates_, score.candidate_id);
    if (!candidate) {
      PublishState(pending_header_, kTracking, score.candidate_id, score.score, reason);
      return;
    }
    PublishActiveCandidate(*candidate, pending_header_, score.score, reason);
  }

  void PublishActiveTargetIfPresent() {
    if (active_candidate_id_ == 0 || !have_candidates_) return;
    const auto* candidate = FindCandidate(latest_candidates_, active_candidate_id_);
    if (!candidate) return;
    PublishActiveCandidate(*candidate, latest_candidates_.header, active_score_,
                           "active_candidate_observed");
  }

  void PublishActiveCandidate(
      const livox_reflective_marker::ReflectiveCandidate& candidate,
      const std_msgs::Header& header, float score, const std::string& reason) {
    active_candidate_publisher_.publish(candidate);
    PublishState(header, kTracking, candidate.candidate_id, score, reason,
                 &candidate.center_map);
    geometry_msgs::PointStamped center;
    center.header = header;
    center.point = candidate.center_map;
    target_center_publisher_.publish(center);
    PublishTargetMarker(candidate, header);
  }

  void PublishTargetMarker(const livox_reflective_marker::ReflectiveCandidate& candidate,
                           const std_msgs::Header& header) {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "reflective_active_target";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.035;
    marker.color.g = 1.0f;
    marker.color.a = 1.0f;

    const double half_x = std::max(0.03, candidate.bbox_size.x * 0.5);
    const double half_y = std::max(0.03, candidate.bbox_size.y * 0.5);
    const double half_z = std::max(0.03, candidate.bbox_size.z * 0.5);
    const double center_x = candidate.center_map.x;
    const double center_y = candidate.center_map.y;
    const double center_z = candidate.center_map.z;
    const auto make_point = [](double x, double y, double z) {
      geometry_msgs::Point point;
      point.x = x;
      point.y = y;
      point.z = z;
      return point;
    };
    const geometry_msgs::Point corners[] = {
        make_point(center_x - half_x, center_y - half_y, center_z - half_z),
        make_point(center_x + half_x, center_y - half_y, center_z - half_z),
        make_point(center_x + half_x, center_y + half_y, center_z - half_z),
        make_point(center_x - half_x, center_y + half_y, center_z - half_z),
        make_point(center_x - half_x, center_y - half_y, center_z + half_z),
        make_point(center_x + half_x, center_y - half_y, center_z + half_z),
        make_point(center_x + half_x, center_y + half_y, center_z + half_z),
        make_point(center_x - half_x, center_y + half_y, center_z + half_z),
    };
    constexpr int edges[][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    marker.points.reserve(24);
    for (const auto& edge : edges) {
      marker.points.push_back(corners[edge[0]]);
      marker.points.push_back(corners[edge[1]]);
    }
    target_marker_publisher_.publish(marker);
  }

  void PublishTargetMarkerDelete(const std_msgs::Header& header) {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "reflective_active_target";
    marker.id = 0;
    marker.action = visualization_msgs::Marker::DELETE;
    marker.pose.orientation.w = 1.0;
    target_marker_publisher_.publish(marker);
  }

  void PublishState(const std_msgs::Header& header, uint8_t state,
                    uint32_t candidate_id, float score, const std::string& reason,
                    const geometry_msgs::Point* center = nullptr) {
    livox_reflective_marker::ReflectiveTargetState output;
    output.header = header;
    output.map_epoch = map_epoch_;
    output.state = state;
    output.candidate_id = candidate_id;
    output.score = score;
    output.reason = reason;
    if (center) output.center_map = *center;
    target_state_publisher_.publish(output);
  }

  void HandleTimeout(const ros::TimerEvent&) {
    if (!pending_request_ ||
        (ros::Time::now() - pending_time_).toSec() <= config_.request_timeout_sec) {
      return;
    }
    ROS_WARN("reflective map decision BPU request timed out after %.2f s",
             config_.request_timeout_sec);
    pending_request_ = false;
    RequestCurrentSnapshot(kRetryTimedOut);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  Config config_;
  ros::Subscriber candidates_subscriber_;
  ros::Subscriber scores_subscriber_;
  ros::Publisher request_publisher_;
  ros::Publisher target_state_publisher_;
  ros::Publisher active_candidate_publisher_;
  ros::Publisher target_center_publisher_;
  ros::Publisher target_marker_publisher_;
  ros::Timer timeout_timer_;

  bool have_epoch_ = false;
  uint32_t map_epoch_ = 0;
  bool have_candidates_ = false;
  livox_reflective_marker::ReflectiveCandidates latest_candidates_;
  std::unordered_set<uint32_t> previous_candidate_ids_;
  std::unordered_set<uint32_t> scoreable_candidate_ids_;
  std::unordered_map<uint32_t, size_t> last_requested_point_counts_;
  uint32_t active_candidate_id_ = 0;
  float active_score_ = 0.0f;

  bool pending_request_ = false;
  std_msgs::Header pending_header_;
	  uint32_t pending_epoch_ = 0;
	  uint64_t pending_request_id_ = 0;
	  uint64_t pending_candidate_snapshot_id_ = 0;
	  uint64_t next_request_id_ = 0;
	  livox_reflective_marker::ReflectiveCandidates pending_candidates_;
  ros::Time pending_time_;
  bool queued_request_ = false;
  uint8_t queued_reason_ = kNewCandidate;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "reflective_map_decision");
  try {
    ReflectiveMapDecision node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("reflective map decision initialization failed: %s", error.what());
    return 1;
  }
  return 0;
}
