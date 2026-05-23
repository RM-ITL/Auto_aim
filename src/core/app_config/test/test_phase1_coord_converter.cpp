// ===================================================================================
// test_phase1_coord_converter.cpp
//
// Sprint 1.C.1 Phase 1 回归：验证 `t_camera_to_gimbal` 已归位到
// `Solver.coord_converter`。
//
// 做法：直接加载已经完成 Phase 1 迁移后的 standard3.yaml，从 YAML 自身读取
// `Solver.coord_converter.t_camera_to_gimbal` 作为期望值，断言 AppConfig 读取结果一致。
//
// 旧代码只读 yaml root 的 `t_camera_to_gimbal`，而该字段已从 root 移走，因此测试
// 必然失败。
// ===================================================================================
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "app_config/app_config.hpp"
#include <yaml-cpp/yaml.h>

namespace
{

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

}  // namespace

int main()
{
  const auto repo_root = find_repo_root();
  if (repo_root.empty()) {
    return fail("cannot locate repo root");
  }

  const auto src_yaml = repo_root / "src/config/standard3.yaml";
  const auto yaml = YAML::LoadFile(src_yaml.string());
  if (yaml["t_camera_to_gimbal"]) {
    return fail("standard3.yaml still has root t_camera_to_gimbal");
  }

  const auto expected =
    yaml["Solver"]["coord_converter"]["t_camera_to_gimbal"].as<std::vector<double>>();
  if (expected.empty()) {
    return fail("Solver.coord_converter.t_camera_to_gimbal is empty in standard3.yaml");
  }

  const auto cfg = app_config::AppConfig::load(src_yaml.string());
  if (!vec_eq(cfg.solver.coord_converter.t_camera_to_gimbal, expected)) {
    std::cerr << "[FAIL] t_camera_to_gimbal mismatch\n";
    std::cerr << "  got size=" << cfg.solver.coord_converter.t_camera_to_gimbal.size() << "\n";
    for (const auto v : cfg.solver.coord_converter.t_camera_to_gimbal) {
      std::cerr << "  got=" << v << "\n";
    }
    return 1;
  }

  std::cout << "[PASS] Solver.coord_converter.t_camera_to_gimbal loaded correctly\n";
  return 0;
}
