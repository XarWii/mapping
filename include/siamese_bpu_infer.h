#pragma once

/// @file siamese_bpu_infer.h
/// @brief V8 C3 bag-split BPU pre-processing and a small hbDNN runtime wrapper.
///
/// Current deployment artifact:
///   siamese_pointnet_explore/deployment/rdk_x5/v8_c3_bag_bpu/v8_c3_bag_qat.bin
///
/// The model contract is V8 C3, T=160/W=10 first deployment:
///   input0 candidate_point        [1, 6, 192, 1] float32
///   input1 candidate_mask         [1, 1, 192, 1] float32
///   input2 candidate_meta         [1, 6, 1, 1]   float32
///   input3 candidate_count_scale  [1, 1, 1, 1]   float32
///   output logits                 [1, 1, 1, 1]   float32
///
/// The output is a logit-like score, not a sigmoid probability.  The deployment
/// threshold is strict: positive iff score > -0.004310191.

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace siamese_bpu {

struct BpuPoint {
  Eigen::Vector3f xyz = Eigen::Vector3f::Zero();
  float reflectivity = 0.0f;
};

class BpuPreprocessor {
 public:
  static constexpr int kNumPoints = 192;
  static constexpr int kPointChannels = 6;
  static constexpr int kMetaDim = 6;
  static constexpr int kNMin = 8;

  static constexpr float kPhysicalScale = 0.22f;
  static constexpr float kXyzClip = 2.0f;
  static constexpr float kRhoClip = 3.0f;
  static constexpr float kRangeScale = 10.0f;

  BpuPreprocessor() { reset(); }

  void prepare(const std::vector<BpuPoint>& raw_points,
               float reflectivity_threshold = 160.0f) {
    reset();

    std::vector<BpuPoint> points;
    points.reserve(raw_points.size());
    for (const auto& p : raw_points) {
      if (p.xyz.allFinite() && std::isfinite(p.reflectivity) &&
          p.reflectivity >= reflectivity_threshold) {
        points.push_back(p);
      }
    }

    raw_count_ = points.size();
    center_ = robustCenter(points);

    std::vector<size_t> selected = deterministicSelect(points, center_);
    valid_count_ = selected.size();

    int clipped_coord_points = 0;
    for (size_t out_i = 0; out_i < selected.size(); ++out_i) {
      const BpuPoint& p = points[selected[out_i]];
      const Eigen::Vector3f q = (p.xyz - center_) / kPhysicalScale;
      const Eigen::Vector3f scaled = q / kXyzClip;
      const bool clipped = std::abs(scaled.x()) > 1.0f ||
                           std::abs(scaled.y()) > 1.0f ||
                           std::abs(scaled.z()) > 1.0f;
      if (clipped) ++clipped_coord_points;

      candidate_point_[0 * kNumPoints + out_i] =
          clamp(scaled.x(), -1.0f, 1.0f);
      candidate_point_[1 * kNumPoints + out_i] =
          clamp(scaled.y(), -1.0f, 1.0f);
      candidate_point_[2 * kNumPoints + out_i] =
          clamp(scaled.z(), -1.0f, 1.0f);
      candidate_point_[3 * kNumPoints + out_i] =
          clamp(q.norm() / kRhoClip, 0.0f, 1.0f);
      candidate_point_[4 * kNumPoints + out_i] =
          clamp(p.reflectivity, 0.0f, 255.0f) / 255.0f;
      candidate_point_[5 * kNumPoints + out_i] =
          clamp((p.reflectivity - 160.0f) / (255.0f - 160.0f), 0.0f, 1.0f);
      candidate_mask_[out_i] = 1.0f;
    }

    const float raw_count_f = static_cast<float>(std::max<size_t>(raw_count_, 1));
    const float valid_count_f = static_cast<float>(valid_count_);
    const float overflow_ratio =
        raw_count_ > kNumPoints
            ? static_cast<float>(raw_count_ - kNumPoints) / raw_count_f
            : 0.0f;
    const float coord_clip_ratio =
        valid_count_ > 0
            ? static_cast<float>(clipped_coord_points) / valid_count_f
            : 0.0f;

    candidate_meta_[0] = valid_count_f / static_cast<float>(kNumPoints);
    candidate_meta_[1] = clamp(overflow_ratio, 0.0f, 1.0f);
    candidate_meta_[2] = clamp(center_.norm() / kRangeScale, 0.0f, 1.0f);
    candidate_meta_[3] = clamp(reflectivity_threshold / 255.0f, 0.0f, 1.0f);
    candidate_meta_[4] = coord_clip_ratio;
    candidate_meta_[5] = valid_count_ >= kNMin ? 1.0f : 0.0f;

    candidate_count_scale_ =
        static_cast<float>(kNumPoints) /
        static_cast<float>(std::max<size_t>(valid_count_, kNMin));
  }

  const float* candidatePoint() const { return candidate_point_.data(); }
  const float* candidateMask() const { return candidate_mask_.data(); }
  const float* candidateMeta() const { return candidate_meta_.data(); }
  float candidateCountScale() const { return candidate_count_scale_; }

  size_t rawCount() const { return raw_count_; }
  size_t validCount() const { return valid_count_; }
  float validRatio() const {
    return static_cast<float>(valid_count_) / static_cast<float>(kNumPoints);
  }
  const Eigen::Vector3f& center() const { return center_; }

  static constexpr size_t candidatePointBytes() {
    return kPointChannels * kNumPoints * sizeof(float);
  }
  static constexpr size_t candidateMaskBytes() {
    return kNumPoints * sizeof(float);
  }
  static constexpr size_t candidateMetaBytes() {
    return kMetaDim * sizeof(float);
  }
  static constexpr size_t candidateCountScaleBytes() {
    return sizeof(float);
  }

  void reset() {
    std::fill(candidate_point_.begin(), candidate_point_.end(), 0.0f);
    std::fill(candidate_mask_.begin(), candidate_mask_.end(), 0.0f);
    std::fill(candidate_meta_.begin(), candidate_meta_.end(), 0.0f);
    candidate_count_scale_ = static_cast<float>(kNumPoints) / kNMin;
    raw_count_ = 0;
    valid_count_ = 0;
    center_ = Eigen::Vector3f::Zero();
  }

 private:
  static float clamp(float v, float lo, float hi) {
    if (!std::isfinite(v)) return lo;
    return std::max(lo, std::min(hi, v));
  }

  static Eigen::Vector3f robustCenter(const std::vector<BpuPoint>& points) {
    if (points.empty()) return Eigen::Vector3f::Zero();

    std::array<std::vector<float>, 3> coords;
    for (auto& v : coords) v.reserve(points.size());
    for (const auto& p : points) {
      coords[0].push_back(p.xyz.x());
      coords[1].push_back(p.xyz.y());
      coords[2].push_back(p.xyz.z());
    }

    Eigen::Vector3f median;
    for (int axis = 0; axis < 3; ++axis) {
      auto& v = coords[axis];
      const size_t mid = v.size() / 2;
      std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
      float m = v[mid];
      if (v.size() % 2 == 0) {
        const float upper = m;
        std::nth_element(v.begin(), v.begin() + static_cast<long>(mid - 1),
                         v.end());
        m = 0.5f * (v[mid - 1] + upper);
      }
      median[axis] = m;
    }

    std::vector<float> distances;
    distances.reserve(points.size());
    for (const auto& p : points) distances.push_back((p.xyz - median).norm());
    const size_t qidx =
        static_cast<size_t>(std::floor(0.75f * static_cast<float>(distances.size() - 1)));
    std::nth_element(distances.begin(), distances.begin() + static_cast<long>(qidx),
                     distances.end());
    const float cutoff = distances[qidx] + 1.0e-8f;

    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    size_t count = 0;
    for (const auto& p : points) {
      if ((p.xyz - median).norm() <= cutoff) {
        sum += p.xyz;
        ++count;
      }
    }
    return count > 0 ? sum / static_cast<float>(count) : median;
  }

  static std::vector<size_t> deterministicSelect(
      const std::vector<BpuPoint>& points, const Eigen::Vector3f& center) {
    const size_t n = points.size();
    const size_t out_n = std::min(n, static_cast<size_t>(kNumPoints));
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), static_cast<size_t>(0));
    if (n > static_cast<size_t>(kNumPoints)) {
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const float da = (points[a].xyz - center).squaredNorm();
        const float db = (points[b].xyz - center).squaredNorm();
        return da < db;
      });
    }
    order.resize(out_n);
    return order;
  }

  // NCHW flattened as [channel][point].
  std::array<float, kPointChannels * kNumPoints> candidate_point_{};
  std::array<float, kNumPoints> candidate_mask_{};
  std::array<float, kMetaDim> candidate_meta_{};
  float candidate_count_scale_ = 0.0f;
  size_t raw_count_ = 0;
  size_t valid_count_ = 0;
  Eigen::Vector3f center_ = Eigen::Vector3f::Zero();
};

class BpuModel {
 public:
  BpuModel() = default;
  ~BpuModel() { release(); }

  BpuModel(const BpuModel&) = delete;
  BpuModel& operator=(const BpuModel&) = delete;
  BpuModel(BpuModel&& other) noexcept { *this = std::move(other); }
  BpuModel& operator=(BpuModel&& other) noexcept {
    if (this != &other) {
      release();
      packed_ = other.packed_;
      dnn_ = other.dnn_;
      inputs_ = std::move(other.inputs_);
      outputs_ = std::move(other.outputs_);
      other.packed_ = nullptr;
      other.dnn_ = nullptr;
    }
    return *this;
  }

  void init(const std::string& model_path) {
    release();

    const char* model_files[] = {model_path.c_str()};
    check(hbDNNInitializeFromFiles(&packed_, model_files, 1),
          "hbDNNInitializeFromFiles");

    const char** model_names = nullptr;
    int32_t model_count = 0;
    check(hbDNNGetModelNameList(&model_names, &model_count, packed_),
          "hbDNNGetModelNameList");
    if (model_count < 1) {
      throw std::runtime_error("BPU model contains no sub-models");
    }

    check(hbDNNGetModelHandle(&dnn_, packed_, model_names[0]),
          "hbDNNGetModelHandle");

    int32_t in_cnt = 0;
    int32_t out_cnt = 0;
    check(hbDNNGetInputCount(&in_cnt, dnn_), "hbDNNGetInputCount");
    check(hbDNNGetOutputCount(&out_cnt, dnn_), "hbDNNGetOutputCount");
    if (in_cnt != 4 || out_cnt != 1) {
      throw std::runtime_error(
          "BPU V8 C3 model expected 4 inputs / 1 output, got " +
          std::to_string(in_cnt) + " / " + std::to_string(out_cnt));
    }

    inputs_.resize(in_cnt);
    outputs_.resize(out_cnt);

    for (int32_t i = 0; i < in_cnt; ++i) {
      check(hbDNNGetInputTensorProperties(&inputs_[i].properties, dnn_, i),
            "hbDNNGetInputTensorProperties");
      allocateTensor(inputs_[i]);
    }
    for (int32_t i = 0; i < out_cnt; ++i) {
      check(hbDNNGetOutputTensorProperties(&outputs_[i].properties, dnn_, i),
            "hbDNNGetOutputTensorProperties");
      allocateTensor(outputs_[i]);
    }

    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl_);
  }

  float infer(const float* candidate_point,
              const float* candidate_mask,
              const float* candidate_meta,
              float candidate_count_scale) {
    // v8_c3_bag_qat.bin model_info order:
    //   input0=candidate_point, input1=candidate_mask,
    //   input2=candidate_meta,  input3=candidate_count_scale.
    writeTensor(inputs_[0], candidate_point,
                BpuPreprocessor::candidatePointBytes());
    writeTensor(inputs_[1], candidate_mask,
                BpuPreprocessor::candidateMaskBytes());
    writeTensor(inputs_[2], candidate_meta,
                BpuPreprocessor::candidateMetaBytes());
    writeTensor(inputs_[3], &candidate_count_scale,
                BpuPreprocessor::candidateCountScaleBytes());

    hbDNNTaskHandle_t task = nullptr;
    hbDNNTensor* output_ptr = outputs_.data();
    check(hbDNNInfer(&task, &output_ptr, inputs_.data(), dnn_, &ctrl_),
          "hbDNNInfer");
    check(hbDNNWaitTaskDone(task, 0), "hbDNNWaitTaskDone");
    check(hbDNNReleaseTask(task), "hbDNNReleaseTask");

    return readScore(outputs_[0]);
  }

 private:
  void release() {
    freeTensors(inputs_);
    freeTensors(outputs_);
    if (dnn_ && packed_) {
      hbDNNRelease(packed_);
    }
    packed_ = nullptr;
    dnn_ = nullptr;
  }

  static void check(int32_t code, const std::string& ctx) {
    if (code != 0) {
      throw std::runtime_error(ctx + " failed, code=" + std::to_string(code));
    }
  }

  static void allocateTensor(hbDNNTensor& t) {
    const auto sz = static_cast<uint32_t>(t.properties.alignedByteSize);
    check(hbSysAllocCachedMem(&t.sysMem[0], sz), "hbSysAllocCachedMem");
  }

  static void writeTensor(hbDNNTensor& t, const void* src, size_t len) {
    const auto cap = static_cast<size_t>(t.properties.alignedByteSize);
    std::memset(t.sysMem[0].virAddr, 0, cap);
    std::memcpy(t.sysMem[0].virAddr, src, std::min(len, cap));
    check(hbSysFlushMem(&t.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
          "hbSysFlushMem input");
  }

  static float readScore(hbDNNTensor& t) {
    check(hbSysFlushMem(&t.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
          "hbSysFlushMem output");
    float v = 0.0f;
    std::memcpy(&v, t.sysMem[0].virAddr, sizeof(float));
    return v;
  }

  static void freeTensors(std::vector<hbDNNTensor>& tensors) {
    for (auto& t : tensors) {
      if (t.sysMem[0].virAddr) {
        hbSysFreeMem(&t.sysMem[0]);
        t.sysMem[0].virAddr = nullptr;
      }
    }
    tensors.clear();
  }

  hbPackedDNNHandle_t packed_ = nullptr;
  hbDNNHandle_t dnn_ = nullptr;
  std::vector<hbDNNTensor> inputs_;
  std::vector<hbDNNTensor> outputs_;
  hbDNNInferCtrlParam ctrl_{};
};

}  // namespace siamese_bpu
