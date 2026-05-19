cmake_minimum_required(VERSION 3.10)
project(app_config LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(utils REQUIRED)

# ====================================================================================
# app_config 库
# ====================================================================================
add_library(${PROJECT_NAME} STATIC
  src/app_config.cpp
)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
    yaml-cpp
    fmt::fmt
    utils::utils
)

ament_target_dependencies(${PROJECT_NAME}
  PUBLIC
    rclcpp
)

# ====================================================================================
# test_load 烟雾测试可执行 (默认不入正式 build，BUILD_TESTING ON 才编)
# ====================================================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  add_executable(test_load test/test_load.cpp)
  target_link_libraries(test_load
    ${PROJECT_NAME}
    utils::utils
    fmt::fmt
  )
  install(TARGETS test_load
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

# ====================================================================================
# 安装与导出
# ====================================================================================
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_export_include_directories(include)
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  yaml-cpp
  fmt
  Eigen3
  utils
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
