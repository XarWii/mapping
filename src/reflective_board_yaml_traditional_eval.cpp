#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace {

struct PointXYZI {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  uint8_t reflectivity = 255;
  float distance = 0.0f;
};

struct BoardPose {
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  Eigen::Vector3f x_axis = Eigen::Vector3f::UnitX();
  Eigen::Vector3f y_axis = Eigen::Vector3f::UnitY();
  Eigen::Vector3f z_axis = Eigen::Vector3f::UnitZ();
};

struct TemplateModel {
  bool loaded = false;
  std::vector<Eigen::Vector2f> points_uv;
  std::vector<std::vector<float>> descriptor_variants;
  std::vector<std::vector<float>> mid_descriptor_variants;
  std::vector<std::vector<float>> coarse_descriptor_variants;
};

struct Config {
  std::string template_path =
      "src/livox_reflective_marker/config/target_board_template.yaml";
  size_t template_grid_cols = 48;
  size_t template_grid_rows = 48;
  size_t template_mid_grid_cols = 32;
  size_t template_mid_grid_rows = 32;
  size_t template_coarse_grid_cols = 24;
  size_t template_coarse_grid_rows = 24;
  size_t min_template_points = 30;
  size_t target_template_match_points = 300;
  size_t far_template_match_points = 120;
  float far_template_distance_m = 8.0f;
  double min_template_score = 0.35;
  double far_min_template_score = 0.28;
  double good_template_score = 0.50;
  float template_plane_tolerance_m = 0.045f;
  double confirm_score_threshold = 0.75;
  double confirm_margin_threshold = 0.30;
};

struct Sample {
  std::string path;
  std::string label;
  std::string run_id;
  int window_frames = 0;
  int candidate_index = -1;
  Eigen::Vector3f candidate_center = Eigen::Vector3f::Zero();
  size_t roi_point_count = 0;
  size_t high_point_count = 0;
  std::vector<PointXYZI> high_points;
};

struct CandidateScore {
  Sample sample;
  double score = 0.0;
  double template_score = 0.0;
  size_t plane_points = 0;
  size_t required_points = 0;
  double required_template_score = 0.0;
  bool valid_pose = false;
  std::string reason = "not scored";
};

std::string Trim(const std::string& s) {
  const size_t begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const size_t end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

std::string Unquote(std::string s) {
  s = Trim(s);
  if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}

std::string ValueAfterColon(const std::string& line) {
  const size_t pos = line.find(':');
  if (pos == std::string::npos) {
    return "";
  }
  return Trim(line.substr(pos + 1));
}

float Clamp01(float v) {
  return std::max(0.0f, std::min(1.0f, v));
}

Eigen::Vector3f ToEigen(const PointXYZI& point) {
  return Eigen::Vector3f(point.x, point.y, point.z);
}

bool ParseVector3(const std::string& line, Eigen::Vector3f* out) {
  static const std::regex re(
      R"(\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+))");
  std::smatch match;
  if (!std::regex_search(line, match, re) || match.size() < 4) {
    return false;
  }
  *out = Eigen::Vector3f(std::stof(match[1]), std::stof(match[2]),
                         std::stof(match[3]));
  return true;
}

bool ParseVector2(const std::string& line, Eigen::Vector2f* out) {
  static const std::regex re(
      R"(\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+))");
  std::smatch match;
  if (!std::regex_search(line, match, re) || match.size() < 3) {
    return false;
  }
  *out = Eigen::Vector2f(std::stof(match[1]), std::stof(match[2]));
  return true;
}

PointXYZI MakePoint(const Eigen::Vector3f& v) {
  PointXYZI p;
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
  p.reflectivity = 255;
  p.distance = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
  return p;
}

std::vector<Eigen::Vector2f> NormalizeUvPoints(
    const std::vector<Eigen::Vector2f>& points) {
  std::vector<Eigen::Vector2f> normalized;
  if (points.size() < 3) {
    return normalized;
  }

  Eigen::Vector2f mean = Eigen::Vector2f::Zero();
  for (const auto& point : points) {
    mean += point;
  }
  mean /= static_cast<float>(points.size());

  Eigen::Matrix2f covariance = Eigen::Matrix2f::Zero();
  for (const auto& point : points) {
    const Eigen::Vector2f d = point - mean;
    covariance += d * d.transpose();
  }
  covariance /= static_cast<float>(points.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return normalized;
  }

  Eigen::Matrix2f axes;
  axes.col(0) = solver.eigenvectors().col(1).normalized();
  axes.col(1) = solver.eigenvectors().col(0).normalized();

  normalized.reserve(points.size());
  float min_x = std::numeric_limits<float>::infinity();
  float max_x = -std::numeric_limits<float>::infinity();
  float min_y = std::numeric_limits<float>::infinity();
  float max_y = -std::numeric_limits<float>::infinity();
  for (const auto& point : points) {
    const Eigen::Vector2f q = axes.transpose() * (point - mean);
    normalized.push_back(q);
    min_x = std::min(min_x, q.x());
    max_x = std::max(max_x, q.x());
    min_y = std::min(min_y, q.y());
    max_y = std::max(max_y, q.y());
  }

  const float span_x = std::max(1e-4f, max_x - min_x);
  const float span_y = std::max(1e-4f, max_y - min_y);
  const float scale = std::max(span_x, span_y);
  for (auto& point : normalized) {
    point /= scale;
  }
  return normalized;
}

std::vector<float> BuildUvDescriptor(const std::vector<Eigen::Vector2f>& points,
                                     size_t cols,
                                     size_t rows,
                                     bool flip_x,
                                     bool flip_y) {
  std::vector<float> grid(cols * rows, 0.0f);
  if (points.empty() || cols == 0 || rows == 0) {
    return grid;
  }

  for (const auto& point : points) {
    float x = flip_x ? -point.x() : point.x();
    float y = flip_y ? -point.y() : point.y();
    x += 0.5f;
    y += 0.5f;
    if (x < 0.0f || x > 1.0f || y < 0.0f || y > 1.0f) {
      continue;
    }
    const size_t col =
        std::min(cols - 1, static_cast<size_t>(std::floor(x * cols)));
    const size_t row =
        std::min(rows - 1, static_cast<size_t>(std::floor(y * rows)));
    grid[row * cols + col] += 1.0f;
  }

  float norm = 0.0f;
  for (float& value : grid) {
    value = std::sqrt(value);
    norm += value * value;
  }
  norm = std::sqrt(norm);
  if (norm > 1e-6f) {
    for (float& value : grid) {
      value /= norm;
    }
  }
  return grid;
}

float CosineSimilarity(const std::vector<float>& a,
                       const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) {
    return 0.0f;
  }
  float dot = 0.0f;
  float an = 0.0f;
  float bn = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    an += a[i] * a[i];
    bn += b[i] * b[i];
  }
  if (an < 1e-6f || bn < 1e-6f) {
    return 0.0f;
  }
  return Clamp01(dot / std::sqrt(an * bn));
}

std::vector<std::vector<float>> BuildDescriptorVariants(
    const std::vector<Eigen::Vector2f>& points,
    size_t cols,
    size_t rows) {
  const std::vector<Eigen::Vector2f> normalized = NormalizeUvPoints(points);
  std::vector<std::vector<float>> variants;
  variants.push_back(BuildUvDescriptor(normalized, cols, rows, false, false));
  variants.push_back(BuildUvDescriptor(normalized, cols, rows, true, false));
  variants.push_back(BuildUvDescriptor(normalized, cols, rows, false, true));
  variants.push_back(BuildUvDescriptor(normalized, cols, rows, true, true));
  return variants;
}

bool LoadTemplateModel(const Config& config, TemplateModel* model) {
  if (model == nullptr) {
    return false;
  }
  *model = TemplateModel();

  std::ifstream in(config.template_path);
  if (!in.is_open()) {
    std::cerr << "failed to open template: " << config.template_path << "\n";
    return false;
  }

  bool in_points = false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find("points_local_uv:") != std::string::npos) {
      in_points = true;
      continue;
    }
    if (!in_points) {
      continue;
    }
    Eigen::Vector2f uv;
    if (ParseVector2(line, &uv)) {
      model->points_uv.push_back(uv);
    }
  }

  if (model->points_uv.size() < config.min_template_points) {
    std::cerr << "template has too few points: " << model->points_uv.size()
              << "/" << config.min_template_points << "\n";
    return false;
  }

  model->descriptor_variants =
      BuildDescriptorVariants(model->points_uv, config.template_grid_cols,
                              config.template_grid_rows);
  model->mid_descriptor_variants =
      BuildDescriptorVariants(model->points_uv, config.template_mid_grid_cols,
                              config.template_mid_grid_rows);
  model->coarse_descriptor_variants =
      BuildDescriptorVariants(model->points_uv, config.template_coarse_grid_cols,
                              config.template_coarse_grid_rows);
  model->loaded = true;
  return true;
}

void LoadTask1Config(const std::string& path, Config* config) {
  std::ifstream in(path);
  if (!in.is_open() || config == nullptr) {
    return;
  }

  bool in_task1 = false;
  std::string line;
  while (std::getline(in, line)) {
    if (line == "task1:") {
      in_task1 = true;
      continue;
    }
    if (in_task1 && !line.empty() && line.front() != ' ' && line.front() != '#') {
      break;
    }
    if (!in_task1) {
      continue;
    }

    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }
    const size_t colon = trimmed.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string key = trimmed.substr(0, colon);
    const std::string value = Unquote(trimmed.substr(colon + 1));
    if (value.empty()) {
      continue;
    }

    try {
      if (key == "template_path" && !value.empty()) {
        config->template_path = value;
      } else if (key == "template_grid_cols") {
        config->template_grid_cols = static_cast<size_t>(std::stoul(value));
      } else if (key == "template_grid_rows") {
        config->template_grid_rows = static_cast<size_t>(std::stoul(value));
      } else if (key == "template_mid_grid_cols") {
        config->template_mid_grid_cols = static_cast<size_t>(std::stoul(value));
      } else if (key == "template_mid_grid_rows") {
        config->template_mid_grid_rows = static_cast<size_t>(std::stoul(value));
      } else if (key == "template_coarse_grid_cols") {
        config->template_coarse_grid_cols = static_cast<size_t>(std::stoul(value));
      } else if (key == "template_coarse_grid_rows") {
        config->template_coarse_grid_rows = static_cast<size_t>(std::stoul(value));
      } else if (key == "min_template_points") {
        config->min_template_points = static_cast<size_t>(std::stoul(value));
      } else if (key == "target_template_match_points") {
        config->target_template_match_points = static_cast<size_t>(std::stoul(value));
      } else if (key == "far_template_match_points") {
        config->far_template_match_points = static_cast<size_t>(std::stoul(value));
      } else if (key == "far_template_distance_m") {
        config->far_template_distance_m = std::stof(value);
      } else if (key == "min_template_score") {
        config->min_template_score = std::stod(value);
      } else if (key == "far_min_template_score") {
        config->far_min_template_score = std::stod(value);
      } else if (key == "good_template_score") {
        config->good_template_score = std::stod(value);
      } else if (key == "template_plane_tolerance_m") {
        config->template_plane_tolerance_m = std::stof(value);
      } else if (key == "confirm_score_threshold") {
        config->confirm_score_threshold = std::stod(value);
      } else if (key == "confirm_margin_threshold") {
        config->confirm_margin_threshold = std::stod(value);
      }
    } catch (const std::exception&) {
    }
  }
}

Sample ParseSample(const std::filesystem::path& path) {
  Sample sample;
  sample.path = path.string();
  std::ifstream in(path);
  bool in_high_points = false;
  std::string line;
  while (std::getline(in, line)) {
    if (StartsWith(line, "label:")) {
      sample.label = Unquote(ValueAfterColon(line));
    } else if (StartsWith(line, "run_id:")) {
      sample.run_id = Unquote(ValueAfterColon(line));
    } else if (StartsWith(line, "window_frames:")) {
      sample.window_frames = std::stoi(ValueAfterColon(line));
    } else if (StartsWith(line, "candidate_index:")) {
      sample.candidate_index = std::stoi(ValueAfterColon(line));
    } else if (StartsWith(line, "candidate_center_xyz:")) {
      ParseVector3(line, &sample.candidate_center);
    } else if (StartsWith(line, "roi_point_count:")) {
      sample.roi_point_count = static_cast<size_t>(std::stoul(ValueAfterColon(line)));
    } else if (StartsWith(line, "high_point_count:")) {
      sample.high_point_count = static_cast<size_t>(std::stoul(ValueAfterColon(line)));
    } else if (StartsWith(line, "high_points_xyz:")) {
      in_high_points = true;
      continue;
    } else if (in_high_points) {
      if (StartsWith(line, "  - [")) {
        Eigen::Vector3f v;
        if (ParseVector3(line, &v)) {
          sample.high_points.push_back(MakePoint(v));
        }
      } else if (!line.empty() && line.front() != ' ') {
        in_high_points = false;
      }
    }
  }
  if (sample.high_point_count == 0) {
    sample.high_point_count = sample.high_points.size();
  }
  return sample;
}

bool EstimatePoseFromHighPoints(const std::vector<PointXYZI>& high_points,
                                const Config& config,
                                BoardPose* pose) {
  if (pose == nullptr || high_points.size() < config.min_template_points) {
    return false;
  }

  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
  for (const auto& point : high_points) {
    origin += ToEigen(point);
  }
  origin /= static_cast<float>(high_points.size());

  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (const auto& point : high_points) {
    const Eigen::Vector3f d = ToEigen(point) - origin;
    covariance += d * d.transpose();
  }
  covariance /= static_cast<float>(high_points.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  Eigen::Vector3f z_axis = solver.eigenvectors().col(0).normalized();
  Eigen::Vector3f x_axis = solver.eigenvectors().col(2).normalized();
  x_axis = x_axis - x_axis.dot(z_axis) * z_axis;
  if (x_axis.norm() < 1e-4f) {
    return false;
  }
  x_axis.normalize();

  Eigen::Vector3f y_axis = z_axis.cross(x_axis);
  if (y_axis.norm() < 1e-4f) {
    return false;
  }
  y_axis.normalize();

  const Eigen::Vector3f to_lidar = -origin;
  if (to_lidar.norm() > 1e-4f && z_axis.dot(to_lidar.normalized()) < 0.0f) {
    z_axis = -z_axis;
    y_axis = -y_axis;
  }

  pose->origin = origin;
  pose->x_axis = x_axis;
  pose->y_axis = y_axis;
  pose->z_axis = z_axis;
  return true;
}

size_t RequiredTemplatePoints(double distance_m, const Config& config) {
  if (distance_m >= config.far_template_distance_m) {
    return config.far_template_match_points;
  }
  const double t =
      Clamp01(static_cast<float>(distance_m /
                                 std::max(1e-3f, config.far_template_distance_m)));
  return static_cast<size_t>(
      std::round((1.0 - t) * static_cast<double>(config.target_template_match_points) +
                 t * static_cast<double>(config.far_template_match_points)));
}

double RequiredTemplateScore(double distance_m, const Config& config) {
  if (distance_m >= config.far_template_distance_m) {
    return config.far_min_template_score;
  }
  const double t =
      Clamp01(static_cast<float>(distance_m /
                                 std::max(1e-3f, config.far_template_distance_m)));
  return (1.0 - t) * config.min_template_score +
         t * config.far_min_template_score;
}

double TemplateScore(const std::vector<PointXYZI>& high_points,
                     const BoardPose& board,
                     const TemplateModel& model,
                     const Config& config,
                     size_t* plane_points) {
  if (plane_points != nullptr) {
    *plane_points = 0;
  }
  if (!model.loaded || high_points.size() < config.min_template_points) {
    return 0.0;
  }

  std::vector<Eigen::Vector2f> uv_points;
  uv_points.reserve(high_points.size());
  const float max_plane_offset =
      std::max(0.08f, 2.0f * config.template_plane_tolerance_m);
  for (const auto& point : high_points) {
    const Eigen::Vector3f rel = ToEigen(point) - board.origin;
    const float w = rel.dot(board.z_axis);
    if (std::fabs(w) > max_plane_offset) {
      continue;
    }
    uv_points.emplace_back(rel.dot(board.x_axis), rel.dot(board.y_axis));
  }
  if (plane_points != nullptr) {
    *plane_points = uv_points.size();
  }

  if (uv_points.size() < config.min_template_points) {
    return 0.0;
  }

  size_t cols = config.template_grid_cols;
  size_t rows = config.template_grid_rows;
  const std::vector<std::vector<float>>* template_variants =
      &model.descriptor_variants;
  if (uv_points.size() < 0.5 * config.target_template_match_points) {
    cols = config.template_coarse_grid_cols;
    rows = config.template_coarse_grid_rows;
    template_variants = &model.coarse_descriptor_variants;
  } else if (uv_points.size() < config.target_template_match_points) {
    cols = config.template_mid_grid_cols;
    rows = config.template_mid_grid_rows;
    template_variants = &model.mid_descriptor_variants;
  }

  const std::vector<std::vector<float>> candidate_variants =
      BuildDescriptorVariants(uv_points, cols, rows);
  double best = 0.0;
  for (const auto& candidate : candidate_variants) {
    for (const auto& templ : *template_variants) {
      best =
          std::max(best, static_cast<double>(CosineSimilarity(candidate, templ)));
    }
  }
  return best;
}

double CalibratedTemplateScore(double template_score,
                               double min_template_score,
                               const Config& config) {
  if (template_score < min_template_score) {
    return 0.0;
  }
  const double normalized_template =
      (template_score - min_template_score) /
      std::max(1e-6, config.good_template_score - min_template_score);
  const double confidence = Clamp01(static_cast<float>(normalized_template));
  return std::min(1.0,
                  config.confirm_score_threshold +
                      (1.0 - config.confirm_score_threshold) * confidence);
}

CandidateScore ScoreSample(const Sample& sample,
                           const TemplateModel& model,
                           const Config& config) {
  CandidateScore result;
  result.sample = sample;
  const double distance_m = static_cast<double>(sample.candidate_center.norm());
  result.required_points = RequiredTemplatePoints(distance_m, config);
  result.required_template_score = RequiredTemplateScore(distance_m, config);

  if (!model.loaded) {
    result.reason = "template not loaded";
    return result;
  }
  if (sample.high_points.size() < config.min_template_points) {
    result.reason = "too few high-reflectivity points";
    return result;
  }
  if (sample.high_points.size() < result.required_points) {
    std::ostringstream oss;
    oss << "waiting for template points: " << sample.high_points.size() << "/"
        << result.required_points;
    result.reason = oss.str();
    return result;
  }

  BoardPose pose;
  if (!EstimatePoseFromHighPoints(sample.high_points, config, &pose)) {
    result.reason = "could not estimate pose";
    return result;
  }

  result.template_score =
      TemplateScore(sample.high_points, pose, model, config, &result.plane_points);
  if (result.template_score < result.required_template_score) {
    std::ostringstream oss;
    oss << "template score too low: " << result.template_score << "/"
        << result.required_template_score;
    result.reason = oss.str();
    return result;
  }

  result.valid_pose = true;
  result.score =
      CalibratedTemplateScore(result.template_score,
                              result.required_template_score, config);
  result.reason = "ok(template)";
  return result;
}

std::vector<Sample> CollectSamples(const std::string& root) {
  std::vector<Sample> samples;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::filesystem::path path = entry.path();
    const std::string name = path.filename().string();
    if (name.rfind("sample_", 0) == 0 && path.extension() == ".yaml") {
      samples.push_back(ParseSample(path));
    }
  }
  return samples;
}

std::string CsvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n") == std::string::npos) {
    return value;
  }
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"') {
      out += '"';
    }
    out += ch;
  }
  out += '"';
  return out;
}

int Main(int argc, char** argv) {
  std::string dataset_root = "test-2";
  std::string params_path = "src/livox_reflective_marker/config/params.yaml";
  std::string output_path = "test-2/traditional_cpp_eval.csv";
  Config config;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dataset-root" && i + 1 < argc) {
      dataset_root = argv[++i];
    } else if (arg == "--params" && i + 1 < argc) {
      params_path = argv[++i];
    } else if (arg == "--template" && i + 1 < argc) {
      config.template_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0]
                << " [--dataset-root test-2] [--params params.yaml]"
                << " [--template target_board_template.yaml]"
                << " [--output result.csv]\n";
      return 0;
    }
  }

  LoadTask1Config(params_path, &config);
  TemplateModel model;
  if (!LoadTemplateModel(config, &model)) {
    return 2;
  }

  const std::vector<Sample> samples = CollectSamples(dataset_root);
  std::map<std::pair<std::string, int>, std::vector<CandidateScore>> grouped;
  for (const auto& sample : samples) {
    grouped[{sample.run_id, sample.window_frames}].push_back(
        ScoreSample(sample, model, config));
  }

  struct Row {
    std::string run_id;
    int window_frames = 0;
    size_t candidate_count = 0;
    bool decision = false;
    bool correct = false;
    std::string predicted_label = "none";
    int predicted_candidate_index = -1;
    std::string best_actual_label;
    double best_score = 0.0;
    double second_score = 0.0;
    double margin = 0.0;
    double best_template_score = 0.0;
    size_t best_high_points = 0;
    size_t best_required_points = 0;
    size_t best_plane_points = 0;
    std::string best_reason;
    int target_rank = -1;
    double target_score = 0.0;
    double target_template_score = 0.0;
    size_t target_high_points = 0;
    size_t target_required_points = 0;
    std::string target_reason;
    int first_decision_window = -1;
    int first_decision_correct = -1;
    int earliest_correct_window = -1;
    int correct_windows = 0;
    int decision_windows = 0;
    int tested_windows = 0;
  };

  std::vector<Row> rows;
  for (auto& [key, scores] : grouped) {
    std::sort(scores.begin(), scores.end(),
              [](const CandidateScore& a, const CandidateScore& b) {
                return a.score > b.score;
              });
    Row row;
    row.run_id = key.first;
    row.window_frames = key.second;
    row.candidate_count = scores.size();
    if (!scores.empty()) {
      const CandidateScore& best = scores.front();
      row.best_actual_label = best.sample.label;
      row.best_score = best.score;
      row.second_score = scores.size() > 1 ? scores[1].score : 0.0;
      row.margin = row.best_score - row.second_score;
      row.best_template_score = best.template_score;
      row.best_high_points = best.sample.high_points.size();
      row.best_required_points = best.required_points;
      row.best_plane_points = best.plane_points;
      row.best_reason = best.reason;
      row.decision = best.valid_pose &&
                     best.score >= config.confirm_score_threshold &&
                     row.margin >= config.confirm_margin_threshold;
      if (row.decision) {
        row.predicted_label = best.sample.label;
        row.predicted_candidate_index = best.sample.candidate_index;
      }
      row.correct = row.decision && row.predicted_label == "target";
    }
    for (size_t i = 0; i < scores.size(); ++i) {
      if (scores[i].sample.label == "target") {
        row.target_rank = static_cast<int>(i + 1);
        row.target_score = scores[i].score;
        row.target_template_score = scores[i].template_score;
        row.target_high_points = scores[i].sample.high_points.size();
        row.target_required_points = scores[i].required_points;
        row.target_reason = scores[i].reason;
        break;
      }
    }
    rows.push_back(row);
  }

  std::map<std::string, std::vector<size_t>> by_run;
  for (size_t i = 0; i < rows.size(); ++i) {
    by_run[rows[i].run_id].push_back(i);
  }
  for (auto& [run_id, indices] : by_run) {
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) {
                return rows[a].window_frames < rows[b].window_frames;
              });
    int first_decision = -1;
    int first_decision_correct = -1;
    int earliest_correct = -1;
    int correct_count = 0;
    int decision_count = 0;
    for (size_t idx : indices) {
      if (rows[idx].decision) {
        ++decision_count;
        if (first_decision < 0) {
          first_decision = rows[idx].window_frames;
          first_decision_correct = rows[idx].correct ? 1 : 0;
        }
      }
      if (rows[idx].correct) {
        ++correct_count;
        if (earliest_correct < 0) {
          earliest_correct = rows[idx].window_frames;
        }
      }
    }
    for (size_t idx : indices) {
      rows[idx].first_decision_window = first_decision;
      rows[idx].first_decision_correct = first_decision_correct;
      rows[idx].earliest_correct_window = earliest_correct;
      rows[idx].correct_windows = correct_count;
      rows[idx].decision_windows = decision_count;
      rows[idx].tested_windows = static_cast<int>(indices.size());
    }
  }

  std::ofstream out(output_path);
  if (!out.is_open()) {
    std::cerr << "failed to open output: " << output_path << "\n";
    return 3;
  }
  out << "run_id,window_frames,candidate_count,decision,correct,"
         "predicted_label,predicted_candidate_index,best_actual_label,"
         "best_score,second_score,margin,best_template_score,best_high_points,"
         "best_required_points,best_plane_points,best_reason,target_rank,"
         "target_score,target_template_score,target_high_points,"
         "target_required_points,target_reason,first_decision_window,"
         "first_decision_correct,earliest_correct_window,correct_windows,"
         "decision_windows,tested_windows,mode\n";
  out << std::fixed << std::setprecision(6);
  for (const auto& row : rows) {
    out << CsvEscape(row.run_id) << ","
        << row.window_frames << ","
        << row.candidate_count << ","
        << (row.decision ? 1 : 0) << ","
        << (row.correct ? 1 : 0) << ","
        << row.predicted_label << ","
        << (row.predicted_candidate_index >= 0 ? std::to_string(row.predicted_candidate_index) : "") << ","
        << row.best_actual_label << ","
        << row.best_score << ","
        << row.second_score << ","
        << row.margin << ","
        << row.best_template_score << ","
        << row.best_high_points << ","
        << row.best_required_points << ","
        << row.best_plane_points << ","
        << CsvEscape(row.best_reason) << ","
        << (row.target_rank >= 0 ? std::to_string(row.target_rank) : "") << ","
        << row.target_score << ","
        << row.target_template_score << ","
        << row.target_high_points << ","
        << row.target_required_points << ","
        << CsvEscape(row.target_reason) << ","
        << (row.first_decision_window >= 0 ? std::to_string(row.first_decision_window) : "") << ","
        << (row.first_decision_correct >= 0 ? std::to_string(row.first_decision_correct) : "") << ","
        << (row.earliest_correct_window >= 0 ? std::to_string(row.earliest_correct_window) : "") << ","
        << row.correct_windows << ","
        << row.decision_windows << ","
        << row.tested_windows << ","
        << "cpp_current_traditional_yaml\n";
  }

  std::cout << "wrote " << output_path << "\n";
  std::cout << "samples=" << samples.size() << " run_windows=" << rows.size()
            << " runs=" << by_run.size() << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return Main(argc, argv);
}
