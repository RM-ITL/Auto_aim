cmake_minimum_required(VERSION 3.8)
project(solver)

# ============================================================================
# 编译设置
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 优化选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================================================
# 查找依赖包
# ============================================================================

# ROS2核心包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# 消息接口包
find_package(autoaim_msgs REQUIRED)

# 第三方库
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(utils REQUIRED)      
  
# 依赖的项目包
find_package(detected REQUIRED)  # 使用detected包导出的armor数据结构

# ============================================================================
# 设置包含目录
# ============================================================================
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${fmt_INCLUDE_DIRS}
)

# ============================================================================
# 构建求解器库
# ============================================================================

# 1. 基础数据结构和工具库
# 注意：这里不再使用detected::armor_data_lib，而是通过ament_target_dependencies来链接
add_library(solver_base SHARED
  src/module/solver.cpp
)

target_include_directories(solver_base PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${fmt_INCLUDE_DIRS}
)

target_link_libraries(solver_base PUBLIC
  ${OpenCV_LIBS}
  Eigen3::Eigen
  fmt
)

# 设置solver_base的ROS依赖
# detected包会自动提供它导出的所有库
ament_target_dependencies(solver_base PUBLIC
  detected  # 这会自动链接detected导出的所有库，包括armor_data_lib
  rclcpp    # 如果solver.cpp需要ROS日志功能
  utils
  fmt
)

# 2. 坐标转换模块
add_library(coord_converter SHARED
  src/module/coord_converter.cpp
)

target_include_directories(coord_converter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${fmt_INCLUDE_DIRS}
)

target_link_libraries(coord_converter PUBLIC
  solver_base
  ${OpenCV_LIBS}
  Eigen3::Eigen
  yaml-cpp
  fmt
)

# 如果coord_converter需要使用detected的功能，也添加依赖
ament_target_dependencies(coord_converter PUBLIC
  detected
  rclcpp
  utils
  fmt
)

# 3. PnP求解模块  
add_library(pnp_solver_lib SHARED
  src/module/pnp_solver.cpp
)

target_include_directories(pnp_solver_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${fmt_INCLUDE_DIRS}
)

target_link_libraries(pnp_solver_lib PUBLIC
  solver_base
  ${OpenCV_LIBS}
  Eigen3::Eigen
  fmt
)

# PnP求解器需要ROS日志功能和detected的armor定义
ament_target_dependencies(pnp_solver_lib PUBLIC
  rclcpp
  detected  # 如果pnp_solver需要armor数据结构
  utils
  fmt
)

# yaw优化
add_library(yaw_optimzier_lib SHARED
  src/module/optimize_yaw.cpp
)

target_include_directories(yaw_optimzier_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${fmt_INCLUDE_DIRS}
)

target_link_libraries(yaw_optimzier_lib PUBLIC
  solver_base
  ${OpenCV_LIBS}
  Eigen3::Eigen
  fmt
)

# PnP求解器需要ROS日志功能和detected的armor定义
ament_target_dependencies(yaw_optimzier_lib PUBLIC
  rclcpp
  detected  # 如果pnp_solver需要armor数据结构
  utils
  fmt
)

# 4. 主求解器接口库
add_library(solver_interface SHARED
  src/solver_node.cpp  # 虽然含node.cpp，但实际上只是接口实现
)

target_include_directories(solver_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${fmt_INCLUDE_DIRS}
)

target_link_libraries(solver_interface PUBLIC
  solver_base
  coord_converter
  pnp_solver_lib
  yaw_optimzier_lib
)

# 设置主接口的ROS依赖
ament_target_dependencies(solver_interface PUBLIC
  rclcpp
  autoaim_msgs
  std_msgs
  geometry_msgs
  detected  # 确保接口也能访问armor数据结构
  utils
  fmt
)

# ============================================================================
# 创建一个汇总库，方便其他包一次性链接所有功能
# ============================================================================
add_library(solver INTERFACE)
target_link_libraries(solver INTERFACE
  solver_base
  coord_converter
  pnp_solver_lib
  solver_interface
  yaw_optimzier_lib
)

# ============================================================================
# 安装配置 (只保留一份！)
# ============================================================================

# 安装库文件
install(TARGETS
  solver_base
  coord_converter
  pnp_solver_lib
  yaw_optimzier_lib
  solver_interface
  solver  # 安装汇总接口
  EXPORT solver_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 导出targets文件，让其他包可以通过find_package找到这些库
install(EXPORT solver_targets
  FILE solverTargets.cmake
  NAMESPACE solver::
  DESTINATION share/${PROJECT_NAME}/cmake
)

# 安装头文件
install(DIRECTORY include/
  DESTINATION include
)

# 如果有配置文件，也安装它们
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL  # 如果不存在也不报错
)

# ============================================================================
# 测试配置
# ============================================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# ============================================================================
# 导出依赖项和库
# ============================================================================

# 导出include目录
ament_export_include_directories(include)

# 导出库
ament_export_libraries(
  solver_base
  coord_converter
  pnp_solver_lib
  solver_interface
  yaw_optimzier_lib
)

# 导出targets
ament_export_targets(solver_targets HAS_LIBRARY_TARGET)

# 导出依赖项
ament_export_dependencies(
  rclcpp
  std_msgs
  geometry_msgs
  tf2_geometry_msgs
  autoaim_msgs
  OpenCV
  Eigen3
  yaml-cpp
  detected  # 导出对detected包的依赖
  utils
  fmt
)

ament_package()