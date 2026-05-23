// ===================================================================================
// test_phase2_camera_intri_single_source.cpp
//
// Sprint 1.C.3 Phase 2 回归：PnPSolverConfig / CoordConverterConfig /
// YawOptimizerConfig 不再持有内参字段；SolverConfig.camera_intri 是唯一
// SubConfig 内参来源。
// ===================================================================================
#include <filesystem>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "app_config/app_config.hpp"
#include <yaml-cpp/yaml.h>

namespace
{

template <typename T, typename = void>
struct has_focal_length : std::false_type
{
};

template <typename T>
struct has_focal_length<T, std::void_t<decltype(std::declval<T &>().focal_length)>>
: std::true_type
{
};

template <typename T, typename = void>
struct has_principal_point : std::false_type
{
};

template <typename T>
struct has_principal_point<T, std::void_t<decltype(std::declval<T &>().principal_point)>>
: std::true_type
{
};

template <typename T, typename = void>
struct has_disto_param : std::false_type
{
};

template <typename T>
struct has_disto_param<T, std::void_t<decltype(std::declval<T &>().disto_param)>>
: std::true_type
{
};

template <typename T>
constexpr bool has_any_camera_intri_member_v =
  has_focal_length<T>::value || has_principal_point<T>::value || has_disto_param<T>::value;

static_assert(
  !has_any_camera_intri_member_v<app_config::PnPSolverConfig>,
  "PnPSolverConfig must not contain focal_length/principal_point/disto_param");
static_assert(
  !has_any_camera_intri_member_v<app_config::CoordConverterConfig>,
  "CoordConverterConfig must not contain focal_length/principal_point/disto_param");
static_assert(
  !has_any_camera_intri_member_v<app_config::YawOptimizerConfig>,
  "YawOptimizerConfig must not contain focal_length/principal_point/disto_param");

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
  if (lhs.size() != rhs.size()) return false;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) return false;
  }
  return true;
}

int assert_vec_eq(
  const std::vector<double> & got,
  const std::vector<double> & expected,
  const std::string & name)
{
  if (vec_eq(got, expected)) return 0;

  std::cerr << "[FAIL] " << name << " mismatch\n"
            << "  expected size=" << expected.size() << "\n"
            << "  got size     =" << got.size() << "\n";
  return 1;
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

int assert_standard3_selected_profile()
{
  const auto repo_root = find_repo_root();
  if (repo_root.empty()) {
    return fail("cannot locate repo root");
  }

  const auto yaml_path = repo_root / "src/config/standard3.yaml";
  const auto yaml = YAML::LoadFile(yaml_path.string());
  const auto cfg = app_config::AppConfig::load(yaml_path.string());

  const auto data = camera_intri_data_node(yaml);
  if (!data || !data["profiles"]) {
    return fail("standard3.yaml missing CalibParam...data.profiles");
  }
  const auto selected_profile = data["profiles"][cfg.camera.lens];
  if (!selected_profile) {
    return fail("standard3.yaml missing selected profile: " + cfg.camera.lens);
  }

  if (const int ret = assert_vec_eq(
        cfg.solver.camera_intri.focal_length,
        selected_profile["focal_length"].as<std::vector<double>>(),
        "solver.camera_intri.focal_length")) {
    return ret;
  }
  if (const int ret = assert_vec_eq(
        cfg.solver.camera_intri.principal_point,
        selected_profile["principal_point"].as<std::vector<double>>(),
        "solver.camera_intri.principal_point")) {
    return ret;
  }
  if (const int ret = assert_vec_eq(
        cfg.solver.camera_intri.disto_param,
        selected_profile["disto_param"].as<std::vector<double>>(),
        "solver.camera_intri.disto_param")) {
    return ret;
  }

  return 0;
}

}  // namespace

int main()
{
  if (const int ret = assert_standard3_selected_profile()) {
    return ret;
  }

  std::cout << "[PASS] camera_intri is the single SubConfig source\n";
  return 0;
}
