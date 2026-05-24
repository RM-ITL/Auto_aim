#pragma once

#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "app_config/app_config.hpp"

namespace app_config
{

std::vector<std::string> validate_collect_errors(const AppConfig & cfg, const YAML::Node & root);
void validate(const AppConfig & cfg, const YAML::Node & root);

}  // namespace app_config
