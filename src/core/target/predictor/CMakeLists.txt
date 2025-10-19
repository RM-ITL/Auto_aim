cmake_minimum_required(VERSION 3.5)
project(predictor)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(autoaim_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(utils REQUIRED)  # 添加utils包依赖
find_package(solver REQUIRED)
find_package(detected REQUIRED)
find_package(fmt REQUIRED)

# 如果需要motion_state_pool，保留这行
find_package(motion_model REQUIRED)

# 包含目录
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${fmt_INCLUDE_DIRS}
)

# ========================================
# 创建静态库
# ========================================
add_library(${PROJECT_NAME} SHARED
  src/target.cpp
)

# 设置库的公共头文件目录
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

# 链接依赖
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_msgs
  geometry_msgs  
  visualization_msgs
  tf2
  tf2_geometry_msgs
  autoaim_msgs
  utils  # 添加utils包依赖
  motion_model  # 如果需要的话取消注释
  solver
  detected
  fmt
)


target_link_libraries(${PROJECT_NAME}
  motion_model::motion_model
)

# ========================================
# 导出库信息
# ========================================
# 导出目标，使其他包可以找到这个库
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
  autoaim_msgs
  utils  # 导出utils依赖
  Eigen3
  motion_model  # 如果需要的话取消注释
  solver
  detected
  fmt
)

# 导出包含目录
ament_export_include_directories(include)

# ========================================
# 安装规则
# ========================================
# 安装库文件
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# 安装头文件
install(
  DIRECTORY include/
  DESTINATION include
)

# 安装导出的cmake配置文件
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)

# ========================================
# 完成包配置
# ========================================
ament_package()