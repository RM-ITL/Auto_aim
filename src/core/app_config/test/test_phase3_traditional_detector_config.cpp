// ===================================================================================
// test_phase3_traditional_detector_config.cpp
//
// Sprint 1.C.1 Phase 3 回归：验证 Traditional_Detector 9 个几何字段已从
// yaml root 迁入顶层 `Traditional_Detector` 段，同时保持 required 行为。
// ===================================================================================
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "app_config/app_config.hpp"
#include <yaml-cpp/yaml.h>

namespace
{

struct TraditionalField
{
  const char * key;
  double app_config::DetectorTraditionalConfig::* member;
};

constexpr std::array<TraditionalField, 9> kTraditionalFields{{
  {"threshold", &app_config::DetectorTraditionalConfig::threshold},
  {"max_angle_error", &app_config::DetectorTraditionalConfig::max_angle_error},
  {"min_lightbar_ratio", &app_config::DetectorTraditionalConfig::min_lightbar_ratio},
  {"max_lightbar_ratio", &app_config::DetectorTraditionalConfig::max_lightbar_ratio},
  {"min_lightbar_length", &app_config::DetectorTraditionalConfig::min_lightbar_length},
  {"min_armor_ratio", &app_config::DetectorTraditionalConfig::min_armor_ratio},
  {"max_armor_ratio", &app_config::DetectorTraditionalConfig::max_armor_ratio},
  {"max_side_ratio", &app_config::DetectorTraditionalConfig::max_side_ratio},
  {"max_rectangular_error", &app_config::DetectorTraditionalConfig::max_rectangular_error},
}};

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

struct ScopedTempFile
{
  explicit ScopedTempFile(std::filesystem::path p) : path(std::move(p)) {}

  ~ScopedTempFile()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  std::filesystem::path path;
};

bool write_without_traditional_detector(
  const YAML::Node & yaml,
  const std::filesystem::path & path,
  std::string & error)
{
  YAML::Node copy = YAML::Clone(yaml);
  if (!copy.remove("Traditional_Detector")) {
    error = "cannot remove Traditional_Detector from cloned yaml";
    return false;
  }

  YAML::Emitter emitter;
  emitter << copy;
  if (!emitter.good()) {
    error = emitter.GetLastError();
    return false;
  }

  std::ofstream out(path);
  if (!out) {
    error = "cannot open temp yaml for writing: " + path.string();
    return false;
  }
  out << emitter.c_str();
  return true;
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

  const auto traditional_yaml = yaml["Traditional_Detector"];
  if (!traditional_yaml) {
    return fail("standard3.yaml missing top-level Traditional_Detector section");
  }

  for (const auto & field : kTraditionalFields) {
    if (yaml[field.key]) {
      return fail(std::string("standard3.yaml still has root ") + field.key);
    }
    if (!traditional_yaml[field.key]) {
      return fail(std::string("Traditional_Detector missing ") + field.key);
    }
  }

  const auto cfg = app_config::AppConfig::load(src_yaml.string());
  for (const auto & field : kTraditionalFields) {
    const auto expected = traditional_yaml[field.key].as<double>();
    const auto got = cfg.detector.traditional.*(field.member);
    if (!same_double(got, expected)) {
      std::cerr << "[FAIL] detector.traditional." << field.key << " mismatch\n"
                << "  expected=" << expected << "\n"
                << "  got=" << got << "\n";
      return 1;
    }
  }

  const auto temp_yaml = std::filesystem::temp_directory_path() /
    ("app_config_phase3_missing_traditional_" +
    std::to_string(std::filesystem::file_size(src_yaml)) + ".yaml");
  ScopedTempFile cleanup(temp_yaml);

  std::string write_error;
  if (!write_without_traditional_detector(yaml, temp_yaml, write_error)) {
    return fail(write_error);
  }

  try {
    (void)app_config::AppConfig::load(temp_yaml.string());
  } catch (const std::exception & e) {
    std::cout << "[PASS] missing Traditional_Detector fails as required: " << e.what() << '\n';
    std::cout << "[PASS] Traditional_Detector geometry fields loaded from nested section\n";
    return 0;
  }

  return fail("AppConfig::load silently accepted yaml without Traditional_Detector");
}
