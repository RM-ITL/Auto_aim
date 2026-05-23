// ===================================================================================
// test_phase1_camera_intri_profiles.cpp
//
// Sprint 1.C.3 Phase 1 回归：验证相机内参已从注释切换迁移为
// camera.lens + CalibParam...data.profiles.{4mm,6mm,8mm}，并验证 AppConfig
// 将选中的 profile 装入 SolverConfig.camera_intri。Phase 2 后 legacy 三份
// 内参字段已删除，本测试只验证 profile 形态和单一来源加载值。
// ===================================================================================
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "app_config/app_config.hpp"
#include <yaml-cpp/yaml.h>

namespace
{

const std::vector<std::string> kConfigFiles{
  "standard3.yaml",
  "standard4.yaml",
  "sentry.yaml",
  "hero.yaml",
  "uav.yaml",
  "dart.yaml",
  "config.yaml",
};

const std::vector<std::string> kLensProfiles{"4mm", "6mm", "8mm"};

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

bool vec_eq(const std::vector<double> & lhs, const std::vector<double> & rhs)
{
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

std::string vec_to_string(const std::vector<double> & values)
{
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(values[i]);
  }
  out += "]";
  return out;
}

int assert_vec_eq(
  const std::vector<double> & got,
  const std::vector<double> & expected,
  const std::string & name)
{
  if (!vec_eq(got, expected)) {
    std::cerr << "[FAIL] " << name << " mismatch\n"
              << "  expected=" << vec_to_string(expected) << "\n"
              << "  got     =" << vec_to_string(got) << "\n";
    return 1;
  }
  return 0;
}

YAML::Node camera_intri_data_node(const YAML::Node & root)
{
  if (!root["CalibParam"] || !root["CalibParam"]["INTRI"] ||
      !root["CalibParam"]["INTRI"]["Camera"]) {
    return YAML::Node{};
  }

  const auto camera_node = root["CalibParam"]["INTRI"]["Camera"][0]["value"];
  if (camera_node["ptr_wrapper"] && camera_node["ptr_wrapper"]["data"]) {
    return camera_node["ptr_wrapper"]["data"];
  }
  return camera_node;
}

int assert_profiles_shape(const YAML::Node & yaml, const std::string & file)
{
  const auto data = camera_intri_data_node(yaml);
  if (!data) {
    return fail(file + ": missing CalibParam.INTRI.Camera[0].value.ptr_wrapper.data");
  }
  if (!data["img_width"] || !data["img_height"]) {
    return fail(file + ": CalibParam...data missing img_width/img_height");
  }
  if (!data["profiles"]) {
    return fail(file + ": CalibParam...data.profiles missing");
  }

  for (const auto & lens : kLensProfiles) {
    const auto profile = data["profiles"][lens];
    if (!profile) {
      return fail(file + ": CalibParam...data.profiles." + lens + " missing");
    }
    if (!profile["focal_length"]) {
      return fail(file + ": profile " + lens + " missing focal_length");
    }
    if (!profile["principal_point"]) {
      return fail(file + ": profile " + lens + " missing principal_point");
    }
    if (!profile["disto_param"]) {
      return fail(file + ": profile " + lens + " missing disto_param");
    }
  }
  return 0;
}

int assert_loaded_intri_matches_yaml(
  const app_config::AppConfig & cfg,
  const YAML::Node & yaml,
  const std::string & file)
{
  const auto selected_profile =
    camera_intri_data_node(yaml)["profiles"][cfg.camera.lens];
  if (!selected_profile) {
    return fail(file + ": selected profile " + cfg.camera.lens + " missing in YAML");
  }

  const auto expected_focal =
    selected_profile["focal_length"].as<std::vector<double>>();
  const auto expected_principal =
    selected_profile["principal_point"].as<std::vector<double>>();
  const auto expected_disto =
    selected_profile["disto_param"].as<std::vector<double>>();

  if (const int ret = assert_vec_eq(
        cfg.solver.camera_intri.focal_length, expected_focal,
        file + ": solver.camera_intri.focal_length")) {
    return ret;
  }
  if (const int ret = assert_vec_eq(
        cfg.solver.camera_intri.principal_point, expected_principal,
        file + ": solver.camera_intri.principal_point")) {
    return ret;
  }
  if (const int ret = assert_vec_eq(
        cfg.solver.camera_intri.disto_param, expected_disto,
        file + ": solver.camera_intri.disto_param")) {
    return ret;
  }

  return 0;
}

int assert_all_configs(const std::filesystem::path & repo_root)
{
  for (const auto & file : kConfigFiles) {
    const auto yaml_path = repo_root / "src/config" / file;
    const auto yaml = YAML::LoadFile(yaml_path.string());

    if (!yaml["camera"]) {
      return fail(file + ": camera section missing");
    }
    if (!yaml["camera"]["lens"]) {
      return fail(file + ": camera.lens missing");
    }
    const auto lens = yaml["camera"]["lens"].as<std::string>();

    if (const int ret = assert_profiles_shape(yaml, file)) {
      return ret;
    }

    const auto cfg = app_config::AppConfig::load(yaml_path.string());
    if (cfg.camera.lens != lens) {
      return fail(file + ": AppConfig camera.lens expected YAML value " + lens +
        ", got " + cfg.camera.lens);
    }
    if (const int ret = assert_loaded_intri_matches_yaml(cfg, yaml, file)) {
      return ret;
    }
  }

  return 0;
}

int assert_reverse_lens_switch(const std::filesystem::path & repo_root)
{
  const auto src_yaml = repo_root / "src/config/standard3.yaml";
  auto yaml = YAML::LoadFile(src_yaml.string());
  yaml["camera"]["lens"] = "4mm";

  const auto temp_dir = std::filesystem::path{"/tmp/opencode"};
  std::filesystem::create_directories(temp_dir);
  const auto temp_yaml = temp_dir / "test_phase1_camera_intri_profiles_standard3_4mm.yaml";

  YAML::Emitter emitter;
  emitter << yaml;
  {
    std::ofstream out(temp_yaml);
    out << emitter.c_str();
  }

  const auto cleanup = [&]() {
    std::error_code ec;
    std::filesystem::remove(temp_yaml, ec);
  };

  const auto cfg = app_config::AppConfig::load(temp_yaml.string());
  const auto expected_profile = camera_intri_data_node(yaml)["profiles"]["4mm"];
  if (!expected_profile) {
    cleanup();
    return fail("standard3.yaml 4mm profile missing after temporary lens switch");
  }

  const auto expected_focal = expected_profile["focal_length"].as<std::vector<double>>();
  const auto expected_principal = expected_profile["principal_point"].as<std::vector<double>>();
  const auto expected_disto = expected_profile["disto_param"].as<std::vector<double>>();

  if (cfg.camera.lens != "4mm") {
    cleanup();
    return fail("temporary standard3 camera.lens expected 4mm, got " + cfg.camera.lens);
  }
  if (const int ret = assert_vec_eq(
        cfg.solver.camera_intri.focal_length, expected_focal,
        "temporary standard3 4mm focal_length")) {
    cleanup();
    return ret;
  }
  if (const int ret = assert_vec_eq(
        cfg.solver.camera_intri.principal_point, expected_principal,
        "temporary standard3 4mm principal_point")) {
    cleanup();
    return ret;
  }
  if (const int ret = assert_vec_eq(
        cfg.solver.camera_intri.disto_param, expected_disto,
        "temporary standard3 4mm disto_param")) {
    cleanup();
    return ret;
  }

  cleanup();
  return 0;
}

}  // namespace

int main()
{
  const auto repo_root = find_repo_root();
  if (repo_root.empty()) {
    return fail("cannot locate repo root");
  }

  if (const int ret = assert_all_configs(repo_root)) {
    return ret;
  }
  if (const int ret = assert_reverse_lens_switch(repo_root)) {
    return ret;
  }

  std::cout << "[PASS] camera lens profiles loaded into solver.camera_intri correctly\n";
  return 0;
}
