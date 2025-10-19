cmake_minimum_required(VERSION 3.8)
project(cboard)

# 默认为C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(utils REQUIRED)  # 自定义工具包，包含logger、thread_safe_queue等
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/cboard.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  # 本项目的头文件目录
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  # Eigen3的头文件路径 - 显式添加
  ${EIGEN3_INCLUDE_DIRS}
  # yaml-cpp的头文件路径
  ${yaml-cpp_INCLUDE_DIRS}
)

if(DEFINED EIGEN3_DEFINITIONS)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ${EIGEN3_DEFINITIONS})
endif()

# 链接系统库
target_link_libraries(${PROJECT_NAME}
  ${EIGEN3_LIBRARIES}
  yaml-cpp
  utils::utils  # 链接自定义工具库
)

# 使用ament处理ROS包依赖
# 注意：这必须在target_link_libraries之后调用
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  utils
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# 导出包含目录
ament_export_include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# 导出库
ament_export_libraries(
  ${PROJECT_NAME}
  yaml-cpp
  ${EIGEN3_LIBRARIES}
)

# 导出依赖
ament_export_dependencies(
  rclcpp
  utils
  Eigen3
  yaml-cpp
)

# 导出目标
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)

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

ament_package()