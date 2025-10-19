cmake_minimum_required(VERSION 3.8)
project(gimbal)

# 设置C++标准为C++17，这是ROS2推荐的版本
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 编译选项配置，开启常用的编译警告
# -Wall: 开启所有常用警告
# -Wextra: 开启额外的警告
# -Wpedantic: 严格遵循C++标准
# -fPIC: 生成位置无关代码，这对动态库是必需的
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
endif()

# 查找ROS2和其他依赖包
# ament_cmake是ROS2的构建系统
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(utils REQUIRED)  # 你的自定义工具包，包含logger等
find_package(yaml-cpp REQUIRED)
find_package(serial  REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SERIAL REQUIRED libserial)

add_library(${PROJECT_NAME} SHARED
  src/gimbal.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${SERIAL_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  yaml-cpp
  ${SERIAL_LIBRARIES}
  utils::utils
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  utils
  serial
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

ament_export_include_directories(
  include
  ${SERIAL_INCLUDE_DIRS}
)

ament_export_libraries(
  ${PROJECT_NAME}
  yaml-cpp
  ${SERIAL_LIBRARIES}
)

ament_export_dependencies(
  rclcpp
  utils
  yaml-cpp
)


ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)


install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)


install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)
ament_package()