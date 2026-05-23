// ===================================================================================
// test_phase2_classifier_config.cpp
//
// Sprint 1.C.1 Phase 2 回归：验证 `Classifier` 段拆分，并验证
// `min_confidence` 已从 DetectorTraditionalConfig 迁到 ClassifierConfig。
//
// 旧代码中 ClassifierConfig 没有 min_confidence，DetectorTraditionalConfig 仍持有该字段，
// 且 standard3.yaml 仍在 root 放置 classify_model / min_confidence，因此本测试会先 RED。
// ===================================================================================
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

#include "app_config/app_config.hpp"
#include <yaml-cpp/yaml.h>

namespace
{

template <typename T, typename = void>
struct has_min_confidence : std::false_type
{};

template <typename T>
struct has_min_confidence<T, std::void_t<decltype(std::declval<T &>().min_confidence)>>
: std::true_type
{};

static_assert(
  has_min_confidence<app_config::ClassifierConfig>::value,
  "ClassifierConfig must own min_confidence after Sprint 1.C.1 Phase 2");

static_assert(
  !has_min_confidence<app_config::DetectorTraditionalConfig>::value,
  "DetectorTraditionalConfig must not own min_confidence after Sprint 1.C.1 Phase 2");

std::filesystem::path find_repo_root()
{
  auto dir = std::filesystem::current_path();
  while (true) {
    if (std::filesystem::exists(dir / "src/config/standard3.yaml") &&
        std::filesystem::exists(dir / "src/core/app_config")) {
      return dir;
    }
    if (!dir.has_parent_path() || dir.parent_path() == dir) {
      return {};
    }
    dir = dir.parent_path();
  }
}

int fail(const std::string & msg)
{
  std::cerr << "[FAIL] " << msg << '\n';
  return 1;
}

bool same_double(const double lhs, const double rhs)
{
  return std::abs(lhs - rhs) < 1e-12;
}

}  // namespace

int main()
{
  const auto repo_root = find_repo_root();
  if (repo_root.empty()) {
    return fail("cannot locate repo root");
  }

  const auto src_yaml = repo_root / "src/config/standard3.yaml";
  const auto yaml = YAML::LoadFile(src_yaml.string());

  if (yaml["classify_model"]) {
    return fail("standard3.yaml still has root classify_model");
  }
  if (yaml["min_confidence"]) {
    return fail("standard3.yaml still has root min_confidence");
  }

  const auto classifier_yaml = yaml["Classifier"];
  if (!classifier_yaml) {
    return fail("standard3.yaml missing Classifier section");
  }
  if (!classifier_yaml["classify_model"]) {
    return fail("Classifier.classify_model missing in standard3.yaml");
  }
  if (!classifier_yaml["min_confidence"]) {
    return fail("Classifier.min_confidence missing in standard3.yaml");
  }

  const auto expected_model = classifier_yaml["classify_model"].as<std::string>();
  const auto expected_min_confidence = classifier_yaml["min_confidence"].as<double>();

  const auto cfg = app_config::AppConfig::load(src_yaml.string());
  if (cfg.detector.classifier.classify_model != expected_model) {
    std::cerr << "[FAIL] classifier.classify_model mismatch\n"
              << "  expected=" << expected_model << "\n"
              << "  got=" << cfg.detector.classifier.classify_model << "\n";
    return 1;
  }
  if (!same_double(cfg.detector.classifier.min_confidence, expected_min_confidence)) {
    std::cerr << "[FAIL] classifier.min_confidence mismatch\n"
              << "  expected=" << expected_min_confidence << "\n"
              << "  got=" << cfg.detector.classifier.min_confidence << "\n";
    return 1;
  }

  std::cout << "[PASS] Classifier.min_confidence loaded from yaml Classifier section\n";
  return 0;
}
