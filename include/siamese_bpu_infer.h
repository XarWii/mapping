#pragma once

/// @file siamese_bpu_infer.h
/// @brief Lightweight C++ port of the Siamese PointNet V3 pre-processing and
///        a minimal RAII wrapper around the RDK X5 BPU (hbDNN) inference API.
///
/// The BPU model expects two inputs:
///   - candidate_nchw       [1, 3, 64, 1]  float32  (NCHW layout)
///   - candidate_valid_ratio [1, 1, 1, 1]  float32
///
/// It outputs one scalar:
///   - score                 [1, 1, 1, 1]  float32  (sigmoid already applied)
///
/// Usage
/// -----
///   BpuModel model;
///   model.init("/path/to/model.bin");
///   BpuPreprocessor pre(64);
///   std::vector<Eigen::Vector3f> points = ...;
///   pre.normalize(points);
///   pre.fixedSize(points);
///   float score = model.infer(pre.candidateNchw(), pre.validRatio());
///

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace siamese_bpu {

// =========================================================================
// Pre-processing  (Python → C++ port of normalize_points + fixed_size)
// =========================================================================

class BpuPreprocessor {
 public:
  static constexpr int kNumPoints = 64;   // N in [1, 3, N, 1]
  static constexpr int kInputChannels = 3;

  BpuPreprocessor() { reset(); }

  /// Normalise a point cloud: centre at origin, scale to unit sphere.
  /// Mutates `points` in-place; also stores per-point validity.
  void normalize(std::vector<Eigen::Vector3f>& points) {
    // Remove non-finite points
    points.erase(
        std::remove_if(points.begin(), points.end(),
                       [](const Eigen::Vector3f& p) {
                         return !p.allFinite();
                       }),
        points.end());

    if (points.empty()) {
      points.push_back(Eigen::Vector3f::Zero());
    }

    // Centre
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (const auto& p : points) mean += p;
    mean /= static_cast<float>(points.size());
    for (auto& p : points) p -= mean;

    // Scale to unit sphere
    float max_radius = 0.0f;
    for (const auto& p : points) {
      max_radius = std::max(max_radius, p.norm());
    }
    if (max_radius > 1e-6f) {
      const float scale = 1.0f / max_radius;
      for (auto& p : points) p *= scale;
    }
  }

  /// Resample to exactly kNumPoints points; build NCHW + valid-ratio tensors.
  void fixedSize(const std::vector<Eigen::Vector3f>& points) {
    const size_t n = points.size();
    if (n == 0) {
      // Should not happen after normalize(), but be defensive
      std::fill(candidate_nchw_.begin(), candidate_nchw_.end(), 0.0f);
      valid_ratio_ = 0.0f;
      return;
    }

    size_t valid = 0;
    std::vector<Eigen::Vector3f> sampled(kNumPoints, Eigen::Vector3f::Zero());

    if (n >= static_cast<size_t>(kNumPoints)) {
      // Uniformly subsample — truncation (same as np.linspace().astype(np.int64))
      for (int i = 0; i < kNumPoints; ++i) {
        size_t idx = static_cast<size_t>(
            static_cast<double>(i) / (kNumPoints - 1) *
            static_cast<double>(n - 1));
        if (idx >= n) idx = n - 1;
        sampled[i] = points[idx];
      }
      valid = kNumPoints;
    } else {
      // Repeat (tile) to fill
      for (int i = 0; i < kNumPoints; ++i) {
        sampled[i] = points[i % n];
      }
      valid = n;
    }

    // Write NCHW layout: [1, 3, 64, 1]
    // candidate_nchw_[c * 64 + i] = sampled[i][c]
    for (int i = 0; i < kNumPoints; ++i) {
      candidate_nchw_[0 * kNumPoints + i] = sampled[i].x();   // channel 0
      candidate_nchw_[1 * kNumPoints + i] = sampled[i].y();   // channel 1
      candidate_nchw_[2 * kNumPoints + i] = sampled[i].z();   // channel 2
    }
    valid_ratio_ = static_cast<float>(valid) / static_cast<float>(kNumPoints);
  }

  /// Accessors for BPU input tensors.
  const float* candidateNchw() const { return candidate_nchw_.data(); }
  float validRatio() const { return valid_ratio_; }

  /// Total bytes for candidate_nchw [1,3,64,1] float32.
  static constexpr size_t candidateNchwBytes() {
    return kInputChannels * kNumPoints * sizeof(float);
  }
  static constexpr size_t validRatioBytes() { return sizeof(float); }

  void reset() {
    std::fill(candidate_nchw_.begin(), candidate_nchw_.end(), 0.0f);
    valid_ratio_ = 1.0f;
  }

 private:
  // NCHW layout: candidate_nchw[c][i] = sampled[i][c]
  // Stored flat: channel 0 (64 floats), channel 1 (64 floats), channel 2 (64 floats)
  std::array<float, kInputChannels * kNumPoints> candidate_nchw_{};
  float valid_ratio_ = 1.0f;
};

// =========================================================================
// BPU Model  (RAII wrapper around hbDNN)
// =========================================================================

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

  /// Load the compiled .bin model.  Must be called once before infer().
  void init(const std::string& model_path) {
    release();

    const char* model_files[] = {model_path.c_str()};
    check(hbDNNInitializeFromFiles(&packed_, model_files, 1),
          "hbDNNInitializeFromFiles");

    const char** model_names = nullptr;
    int32_t model_count = 0;
    check(hbDNNGetModelNameList(&model_names, &model_count, packed_),
          "hbDNNGetModelNameList");
    if (model_count < 1)
      throw std::runtime_error("BPU model contains no sub-models");

    check(hbDNNGetModelHandle(&dnn_, packed_, model_names[0]),
          "hbDNNGetModelHandle");

    int32_t in_cnt = 0, out_cnt = 0;
    check(hbDNNGetInputCount(&in_cnt, dnn_), "hbDNNGetInputCount");
    check(hbDNNGetOutputCount(&out_cnt, dnn_), "hbDNNGetOutputCount");
    if (in_cnt != 2 || out_cnt != 1) {
      throw std::runtime_error(
          "BPU model expected 2 inputs / 1 output, got " +
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

  /// Run inference.  `candidate_nchw` must point to 3*64 floats in NCHW
  /// order; `valid_ratio` is a single float.  Returns sigmoid score [0,1].
  float infer(const float* candidate_nchw, float valid_ratio) {
    // Write input 0: candidate_nchw  [1, 3, 64, 1]
    writeTensor(inputs_[0], candidate_nchw,
                BpuPreprocessor::candidateNchwBytes());
    // Write input 1: valid_ratio     [1, 1, 1, 1]
    writeTensor(inputs_[1], &valid_ratio,
                BpuPreprocessor::validRatioBytes());

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
      // hbDNNRelease releases the whole packed handle; dnn_ becomes invalid.
      hbDNNRelease(packed_);
    }
    packed_ = nullptr;
    dnn_ = nullptr;
  }

  static void check(int32_t code, const std::string& ctx) {
    if (code != 0)
      throw std::runtime_error(ctx + " failed, code=" + std::to_string(code));
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
