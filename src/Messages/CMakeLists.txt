cmake_minimum_required(VERSION 3.5)
project(autoaim_msgs)

# 设置CMake策略以避免警告
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# 生成消息接口
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Target.msg"
  "msg/Targets.msg"
  "msg/Armor.msg"
  "msg/Armorresult.msg"
  "msg/Prediction.msg"
  "msg/Predictions.msg"
  "msg/Serialcmd.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

# 安装头文件，让其他包可以使用
install(
  DIRECTORY include/
  DESTINATION include
)

# 导出包含目录，让其他包可以找到头文件
ament_export_include_directories(include)
# 导出依赖，让其他包知道需要这些依赖
ament_export_dependencies(std_msgs geometry_msgs sensor_msgs)

ament_package()