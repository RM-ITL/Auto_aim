cmake_minimum_required(VERSION 3.8)
project(tracker)

# 设置C++标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 添加编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(autoaim_msgs REQUIRED)  # 自定义消息包
find_package(cv_bridge REQUIRED)

# 查找外部库
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

find_package(utils REQUIRED)  # 添加utils包依赖
find_package(solver REQUIRED)
find_package(detected REQUIRED)
find_package(predictor REQUIRED)
find_package(serial REQUIRED)
find_package(gimbal REQUIRED)


# 包含目录
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# 创建tracker库
add_library(tracker_lib
  src/tracker.cpp
)

# 设置tracker库的依赖
ament_target_dependencies(tracker_lib
  rclcpp
  cv_bridge
  autoaim_msgs
  utils
  solver
  predictor
  detected
  serial
  gimbal
)

# # 链接外部库到tracker库
# target_link_libraries(tracker_lib
#   ${OpenCV_LIBS}
#   yaml-cpp
#   Eigen3::Eigen
# )

# # 创建tracker_node可执行文件
# add_executable(tracker_node
#   src/tracker_node.cpp
# )

# # 设置tracker_node的依赖
# ament_target_dependencies(tracker_node
#   rclcpp
#   cv_bridge
#   sensor_msgs
#   geometry_msgs
#   autoaim_msgs
#   tf2
#   utils
#   solver
#   predictor
#   detected
#   serial
#   gimbal
# )

# # 链接库到可执行文件
# target_link_libraries(tracker_node
#   tracker_lib
#   yaml-cpp
#   Eigen3::Eigen
# )

# 安装可执行文件
install(TARGETS
  # tracker_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装库文件（如果需要被其他包使用）
install(TARGETS
  tracker_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 安装头文件
install(DIRECTORY include/
  DESTINATION include
)

# 安装launch文件（如果有）
# install(DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}
# )

# 安装配置文件（如果有）
# install(DIRECTORY config
#   DESTINATION share/${PROJECT_NAME}
# )

# 导出依赖信息
ament_export_include_directories(include)
ament_export_libraries(tracker_lib)
ament_export_dependencies(
  rclcpp
  sensor_msgs
  cv_bridge
  geometry_msgs
  autoaim_msgs
  tf2
  Eigen3
  yaml-cpp
  OpenCV
  utils
  solver
  predictor
  detected
  serial
  gimbal
)

# 添加测试支持（可选）
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # 以下行会跳过版权检查器，根据需要取消注释
  # set(ament_cmake_copyright_FOUND TRUE)
  # 以下行会跳过cpplint（仅适用于Google风格）
  # set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()