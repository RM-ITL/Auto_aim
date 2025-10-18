cmake_minimum_required(VERSION 3.8)
project(utils)

# 默认为C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
endif()

# ============================================
# 查找所有依赖包
# ============================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

# ============================================
# 创建静态库
# ============================================
add_library(${PROJECT_NAME} STATIC
  src/imu_data_processor.cpp
  src/math_tools.cpp
  src/logger.cpp
)

# ============================================
# 设置包含目录 - 这是关键！
# 必须在创建目标后立即设置，确保编译时能找到所有头文件
# ============================================
target_include_directories(${PROJECT_NAME} PUBLIC
  # 本项目的头文件
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  # Eigen3的头文件路径 - 显式添加
  ${EIGEN3_INCLUDE_DIRS}
  # OpenCV的头文件路径 - 显式添加  
  ${OpenCV_INCLUDE_DIRS}
  # yaml-cpp的头文件路径（如果需要）
  ${yaml-cpp_INCLUDE_DIRS}
  ${fmt_INCLUDE_DIRS}
)

# ============================================
# 处理编译定义（如果Eigen需要）
# ============================================
if(DEFINED EIGEN3_DEFINITIONS)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ${EIGEN3_DEFINITIONS})
endif()

# ============================================
# 链接库文件
# 注意：先设置非ROS库，再用ament处理ROS包
# ============================================
# 链接系统库
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${EIGEN3_LIBRARIES}
  yaml-cpp
  fmt
)

# 使用ament处理ROS包依赖
# 注意：这必须在target_link_libraries之后调用
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
)

# ============================================
# 创建别名目标（向后兼容）
# ============================================
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# ============================================
# 导出配置
# ============================================
# 导出包含目录
ament_export_include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# 导出库
ament_export_libraries(
  ${PROJECT_NAME}
  yaml-cpp
  ${OpenCV_LIBS}
  ${EIGEN3_LIBRARIES}
  fmt
)

# 导出依赖
ament_export_dependencies(
  rclcpp
  OpenCV
  Eigen3
  yaml-cpp
  fmt
)

# 导出目标
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)

# ============================================
# 安装规则
# ============================================
# 安装头文件
install(
  DIRECTORY include/
  DESTINATION include
)

# 安装库和导出配置
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib  
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# 安装CMake导出文件
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)

# ============================================
# 生成包配置
# ============================================
ament_package()