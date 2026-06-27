#include <livox_reflective_marker/RecognitionScoreEntry.h>
#include <livox_reflective_marker/RecognitionScores.h>
#include <livox_reflective_marker/ReflectiveCandidates.h>
#include <livox_reflective_marker/ReflectiveRecognitionRequest.h>
#include <livox_reflective_marker/pointcloud2_codec.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

#include "siamese_bpu_infer.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint8_t kStatusScored = 0;
constexpr uint8_t kStatusInsufficientPoints = 3;
constexpr uint8_t kStatusNoMatchingCluster = 4;

struct Config {
  std::string candidates_topic = "/reflective/candidates";
  std::string scores_topic = "/reflective/recognition_scores";
  std::string recognition_request_topic = "/reflective/recognition_request";
  std::string bpu_model_path;
  float reflectivity_threshold = 160.0f;
  int min_inference_points = 8;
};

class ReflectiveBpuScorer {
 public:
  ReflectiveBpuScorer() : private_nh_("~"), config_(LoadConfig()) {
    if (config_.bpu_model_path.empty() || config_.min_inference_points < 1) {
      throw std::runtime_error("invalid reflective_bpu_scorer parameters");
    }
    ROS_INFO("reflective BPU scorer loading model: %s",
             config_.bpu_model_path.c_str());
    model_.init(config_.bpu_model_path);
    candidates_subscriber_ = nh_.subscribe(
        config_.candidates_topic, 1, &ReflectiveBpuScorer::HandleCandidates, this,
        ros::TransportHints().tcpNoDelay());
    request_subscriber_ = nh_.subscribe(
        config_.recognition_request_topic, 2,
        &ReflectiveBpuScorer::HandleRecognitionRequest, this,
        ros::TransportHints().tcpNoDelay());
    // Scores are event-driven, so retain one small batch for late-starting
    // diagnostics without retaining any candidate point clouds.
    scores_publisher_ = nh_.advertise<livox_reflective_marker::RecognitionScores>(
        config_.scores_topic, 1, true);
    ROS_INFO("reflective BPU scorer ready: candidates=%s request=%s scores=%s",
             config_.candidates_topic.c_str(),
             config_.recognition_request_topic.c_str(), config_.scores_topic.c_str());
  }

 private:
  Config LoadConfig() {
    Config config;
    private_nh_.param("candidates_topic", config.candidates_topic,
                      config.candidates_topic);
    private_nh_.param("scores_topic", config.scores_topic, config.scores_topic);
    private_nh_.param("recognition_request_topic", config.recognition_request_topic,
                      config.recognition_request_topic);
    private_nh_.param("bpu_model_path", config.bpu_model_path,
                      config.bpu_model_path);
    private_nh_.param("reflectivity_threshold", config.reflectivity_threshold,
                      config.reflectivity_threshold);
    private_nh_.param("min_inference_points", config.min_inference_points,
                      config.min_inference_points);
    return config;
  }

  bool DecodeSensorCloud(const sensor_msgs::PointCloud2& cloud,
                         std::vector<siamese_bpu::BpuPoint>* points) const {
    if (cloud.is_bigendian || cloud.point_step == 0) return false;
    const sensor_msgs::PointField* x = livox_reflective_marker::pointcloud2::FindField(cloud, "x");
    const sensor_msgs::PointField* y = livox_reflective_marker::pointcloud2::FindField(cloud, "y");
    const sensor_msgs::PointField* z = livox_reflective_marker::pointcloud2::FindField(cloud, "z");
    const sensor_msgs::PointField* intensity = livox_reflective_marker::pointcloud2::FindField(cloud, "intensity");
    if (!x || !y || !z || !intensity) return false;
    const size_t count = static_cast<size_t>(cloud.width) * cloud.height;
    if (!livox_reflective_marker::pointcloud2::HasCompleteRows(cloud)) return false;

    points->clear();
    points->reserve(count);
    for (uint32_t row = 0; row < cloud.height; ++row) {
      const size_t row_start = static_cast<size_t>(row) * cloud.row_step;
      for (uint32_t col = 0; col < cloud.width; ++col) {
        const uint8_t* source = cloud.data.data() + row_start +
                                static_cast<size_t>(col) * cloud.point_step;
        float px = 0.0f;
        float py = 0.0f;
        float pz = 0.0f;
        float reflectivity = 0.0f;
        if (!livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *x, &px) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *y, &py) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *z, &pz) ||
            !livox_reflective_marker::pointcloud2::ReadFiniteFloat32(source, *intensity, &reflectivity)) {
          continue;
        }
        siamese_bpu::BpuPoint point;
        point.xyz = Eigen::Vector3f(px, py, pz);
        point.reflectivity = reflectivity;
        points->push_back(point);
      }
    }
    return true;
  }

  static bool SameSnapshot(const std_msgs::Header& first, uint32_t first_epoch,
                           const std_msgs::Header& second, uint32_t second_epoch) {
    return first_epoch == second_epoch && first.stamp == second.stamp &&
           first.frame_id == second.frame_id;
  }

  void HandleCandidates(
      const livox_reflective_marker::ReflectiveCandidates::ConstPtr& candidates) {
    latest_candidates_ = *candidates;
    have_candidates_ = true;
    TryScorePendingRequest();
  }

  void HandleRecognitionRequest(
      const livox_reflective_marker::ReflectiveRecognitionRequest::ConstPtr& request) {
    pending_request_ = *request;
    have_pending_request_ = true;
    TryScorePendingRequest();
  }

  void TryScorePendingRequest() {
	    if (!have_candidates_ || !have_pending_request_ ||
	        latest_candidates_.candidate_snapshot_id !=
	            pending_request_.candidate_snapshot_id ||
	        !SameSnapshot(latest_candidates_.header, latest_candidates_.map_epoch,
	                      pending_request_.header, pending_request_.map_epoch)) {
	      return;
	    }
	    have_pending_request_ = false;
	    ScoreCandidates(latest_candidates_, pending_request_);
	  }

	  void ScoreCandidates(
	    const livox_reflective_marker::ReflectiveCandidates& candidates,
	    const livox_reflective_marker::ReflectiveRecognitionRequest& request) {
	    livox_reflective_marker::RecognitionScores scores;
	    scores.header = candidates.header;
	    scores.map_epoch = candidates.map_epoch;
	    scores.request_id = request.request_id;
	    scores.candidate_snapshot_id = candidates.candidate_snapshot_id;
	    scores.entries.reserve(candidates.candidates.size());

    size_t rejected = 0;
    for (const auto& candidate : candidates.candidates) {
	      std::vector<siamese_bpu::BpuPoint> points;
	      if (!DecodeSensorCloud(candidate.cloud_sensor, &points)) {
	        ++rejected;
	        livox_reflective_marker::RecognitionScoreEntry entry;
	        entry.candidate_id = candidate.candidate_id;
	        entry.support_revision = candidate.support_revision;
	        entry.status = kStatusNoMatchingCluster;
	        entry.score_valid = false;
	        entry.windows_used = 0;
	        entry.center_map = candidate.center_map;
	        entry.center_sensor = candidate.center_sensor;
	        entry.center_sensor_valid = true;
	        entry.center_sensor_frame_id = candidate.cloud_sensor.header.frame_id;
	        entry.center_sensor_stamp = candidate.cloud_sensor.header.stamp;
	        entry.voxel_count = candidate.voxel_count;
	        entry.evidence_count = candidate.evidence_count;
	        scores.entries.push_back(entry);
	        ROS_WARN_THROTTLE(2.0,
	                          "reflective BPU scorer rejected malformed candidate cloud");
	        continue;
      }
      const size_t source_point_count = points.size();
      preprocessor_.prepare(points, config_.reflectivity_threshold);
      const size_t valid_point_count = preprocessor_.validCount();
	      if (valid_point_count <
	          static_cast<size_t>(config_.min_inference_points)) {
	        ++rejected;
	        livox_reflective_marker::RecognitionScoreEntry entry;
	        entry.candidate_id = candidate.candidate_id;
	        entry.support_revision = candidate.support_revision;
	        entry.status = kStatusInsufficientPoints;
	        entry.score_valid = false;
	        entry.windows_used = 0;
	        entry.center_map = candidate.center_map;
	        entry.center_sensor = candidate.center_sensor;
	        entry.center_sensor_valid = true;
	        entry.center_sensor_frame_id = candidate.cloud_sensor.header.frame_id;
	        entry.center_sensor_stamp = candidate.cloud_sensor.header.stamp;
	        entry.voxel_count = candidate.voxel_count;
	        entry.evidence_count = candidate.evidence_count;
	        scores.entries.push_back(entry);
	        ROS_INFO("reflective BPU candidate=%u source_points=%zu valid_points=%zu "
                 "voxel_count=%u evidence=%u result=too_sparse",
                 candidate.candidate_id, source_point_count, valid_point_count,
                 candidate.voxel_count, candidate.evidence_count);
        continue;
      }

	      livox_reflective_marker::RecognitionScoreEntry entry;
	      entry.candidate_id = candidate.candidate_id;
	      entry.support_revision = candidate.support_revision;
	      entry.status = kStatusScored;
	      entry.score_valid = true;
	      entry.windows_used = 1;
	      entry.score = model_.infer(preprocessor_.candidatePoint(),
                                 preprocessor_.candidateMask(),
                                 preprocessor_.candidateMeta(),
                                 preprocessor_.candidateCountScale());
	      entry.center_map = candidate.center_map;
	      entry.center_sensor = candidate.center_sensor;
	      entry.center_sensor_valid = true;
	      entry.center_sensor_frame_id = candidate.cloud_sensor.header.frame_id;
	      entry.center_sensor_stamp = candidate.cloud_sensor.header.stamp;
      entry.voxel_count = candidate.voxel_count;
      entry.evidence_count = candidate.evidence_count;
      scores.entries.push_back(entry);
      ROS_INFO("reflective BPU candidate=%u source_points=%zu valid_points=%zu "
               "voxel_count=%u evidence=%u score=%.4f",
               candidate.candidate_id, source_point_count, valid_point_count,
               candidate.voxel_count, candidate.evidence_count,
               static_cast<double>(entry.score));
    }
    scores_publisher_.publish(scores);

    ROS_INFO_THROTTLE(
        2.0,
	        "reflective BPU epoch=%u candidates=%zu scored=%zu rejected=%zu reason=%u",
	        scores.map_epoch, candidates.candidates.size(), scores.entries.size(), rejected,
	        static_cast<unsigned int>(request.reason));
	  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  Config config_;
  ros::Subscriber candidates_subscriber_;
  ros::Subscriber request_subscriber_;
  ros::Publisher scores_publisher_;
  bool have_candidates_ = false;
  bool have_pending_request_ = false;
  livox_reflective_marker::ReflectiveCandidates latest_candidates_;
  livox_reflective_marker::ReflectiveRecognitionRequest pending_request_;
  siamese_bpu::BpuModel model_;
  siamese_bpu::BpuPreprocessor preprocessor_;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "reflective_bpu_scorer");
  try {
    ReflectiveBpuScorer node;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL("reflective BPU scorer initialization failed: %s", error.what());
    return 1;
  }
  return 0;
}
