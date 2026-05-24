// ===================================================================================
// utils::yaml — yaml 加载与字段读取 helper。
//
// **范围说明（Sprint 1.C.4 注明）**：
//   - 本 helper 仅供 `src/core/app_config/` 与 `src/core/auto_buff/`（整包冻结）使用。
//   - 外部消费者已在 Sprint 1.C / 1.C.1 / 1.C.3 全部迁出。
//   - **新代码不要再 include 此 helper**；统一走 AppConfig::load + SubConfig 注入。
//   - 物理位置保留在 utils/ 是因为 auto_buff 冻结区仍在用；待 auto_buff 命运决策
//     后再考虑是否挪到 app_config/ 内部。
// ===================================================================================
#ifndef TOOLS__YAML_HPP
#define TOOLS__YAML_HPP

#include <yaml-cpp/yaml.h>

#include "logger.hpp"

namespace utils
{
inline YAML::Node load(const std::string & path)
{
  try {
    return YAML::LoadFile(path);
  } catch (const YAML::BadFile & e) {
    utils::logger()->error("[YAML] Failed to load file: {}", e.what());
    exit(1);
  } catch (const YAML::ParserException & e) {
    utils::logger()->error("[YAML] Parser error: {}", e.what());
    exit(1);
  }
}

template <typename T>
inline T read(const YAML::Node & yaml, const std::string & key)
{
  if (yaml[key]) return yaml[key].as<T>();
  utils::logger()->error("[YAML] {} not found!", key);
  exit(1);
}

template <typename T>
inline T read(const YAML::Node & yaml, const std::string & key, const T & default_value)
{
  if (yaml[key]) return yaml[key].as<T>();
  return default_value;
}

}  // namespace tools

#endif  // TOOLS__YAML_HPP
