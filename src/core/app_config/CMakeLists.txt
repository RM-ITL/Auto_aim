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
  src/validator.cpp
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

  add_executable(test_phase1_coord_converter test/test_phase1_coord_converter.cpp)
  target_link_libraries(test_phase1_coord_converter
    ${PROJECT_NAME}
    utils::utils
    fmt::fmt
  )

  add_executable(test_phase2_classifier_config test/test_phase2_classifier_config.cpp)
  target_link_libraries(test_phase2_classifier_config
    ${PROJECT_NAME}
    utils::utils
    fmt::fmt
  )

  add_executable(test_phase3_traditional_detector_config test/test_phase3_traditional_detector_config.cpp)
  target_link_libraries(test_phase3_traditional_detector_config
    ${PROJECT_NAME}
    utils::utils
    fmt::fmt
  )

  add_executable(test_phase1_camera_intri_profiles test/test_phase1_camera_intri_profiles.cpp)
  target_link_libraries(test_phase1_camera_intri_profiles
    ${PROJECT_NAME}
    utils::utils
    fmt::fmt
  )

  add_executable(test_phase2_camera_intri_single_source test/test_phase2_camera_intri_single_source.cpp)
  target_link_libraries(test_phase2_camera_intri_single_source
    ${PROJECT_NAME}
    utils::utils
    fmt::fmt
  )

  add_executable(test_phase1_validator test/test_phase1_validator.cpp)
  target_link_libraries(test_phase1_validator
    ${PROJECT_NAME}
    utils::utils
    fmt::fmt
    yaml-cpp
  )

  install(TARGETS test_load
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  install(TARGETS test_phase1_coord_converter
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  install(TARGETS test_phase2_classifier_config
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  install(TARGETS test_phase3_traditional_detector_config
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  install(TARGETS test_phase1_camera_intri_profiles
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  install(TARGETS test_phase2_camera_intri_single_source
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
  install(TARGETS test_phase1_validator
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )

  # ==================================================================================
  # 向 ctest 注册测试，使 `colcon test` / CI 能自动发现并执行。
  # 6 个 phase_* 测试自行向上查找仓库根 (定位 src/config/*.yaml)，
  # 故 WORKING_DIRECTORY 指向包源码目录即可（其父链包含仓库根）。
  # test_load 需要一个 yaml 路径参数。
  # ==================================================================================
  enable_testing()

  add_test(
    NAME test_load
    COMMAND test_load ${CMAKE_CURRENT_SOURCE_DIR}/../../config/standard3.yaml
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  add_test(NAME test_phase1_coord_converter COMMAND test_phase1_coord_converter
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME test_phase2_classifier_config COMMAND test_phase2_classifier_config
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME test_phase3_traditional_detector_config COMMAND test_phase3_traditional_detector_config
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME test_phase1_camera_intri_profiles COMMAND test_phase1_camera_intri_profiles
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME test_phase2_camera_intri_single_source COMMAND test_phase2_camera_intri_single_source
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME test_phase1_validator COMMAND test_phase1_validator
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
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
