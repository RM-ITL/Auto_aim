cmake_minimum_required(VERSION 3.10)
project(auto_aim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_srvs REQUIRED)

find_package(autoaim_msgs REQUIRED)
find_package(utils REQUIRED)
find_package(deveice REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)

# ====================================================================================
# auto_aim_detected
# ====================================================================================
set(AUTO_AIM_DETECTED_SOURCES
  detected/src/armor.cpp
  detected/src/detector.cpp
  detected/src/yolo11.cpp
  detected/src/classifier.cpp
  detected/src/detect_node.cpp
)

add_library(auto_aim_detected STATIC ${AUTO_AIM_DETECTED_SOURCES})
add_library(auto_aim::detected ALIAS auto_aim_detected)

target_include_directories(auto_aim_detected
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/detected/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_aim_detected
  PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
    openvino::runtime
    utils::utils
)

ament_target_dependencies(auto_aim_detected
  PUBLIC
    rclcpp
)

# ====================================================================================
# auto_aim_motion_model
# ====================================================================================
add_library(auto_aim_motion_model STATIC
  target/motion_model/src/full_model.cpp
)
add_library(auto_aim::motion_model ALIAS auto_aim_motion_model)

target_include_directories(auto_aim_motion_model
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/target/motion_model/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_aim_motion_model
  PUBLIC
    Eigen3::Eigen
)

# ====================================================================================
# auto_aim_solver
# ====================================================================================
set(AUTO_AIM_SOLVER_SOURCES
  solver/src/module/solver.cpp
  solver/src/module/coord_converter.cpp
  solver/src/module/pnp_solver.cpp
  solver/src/module/optimize_yaw.cpp
  solver/src/solver_node.cpp
)

add_library(auto_aim_solver STATIC ${AUTO_AIM_SOLVER_SOURCES})
add_library(auto_aim::solver ALIAS auto_aim_solver)

target_include_directories(auto_aim_solver
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/solver/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_aim_solver
  PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    yaml-cpp
    fmt::fmt
    auto_aim_detected
    utils::utils
)

ament_target_dependencies(auto_aim_solver
  PUBLIC
    rclcpp
    std_msgs
    geometry_msgs
    autoaim_msgs
)

# ====================================================================================
# auto_aim_predictor
# ====================================================================================
add_library(auto_aim_predictor STATIC
  target/predictor/src/target.cpp
  target/predictor/src/outpost_target.cpp
)
add_library(auto_aim::predictor ALIAS auto_aim_predictor)

target_include_directories(auto_aim_predictor
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/target/predictor/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_aim_predictor
  PUBLIC
    Eigen3::Eigen
    fmt::fmt
    auto_aim_motion_model
    auto_aim_solver
    utils::utils
)

ament_target_dependencies(auto_aim_predictor
  PUBLIC
    rclcpp
    std_msgs
    geometry_msgs
    autoaim_msgs
)

# ====================================================================================
# auto_aim_planner
# ====================================================================================
set(TINYMPC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/planner/tinympc)
set(TINYMPC_INCLUDE_SRC_DIR ${TINYMPC_SOURCE_DIR}/include)

set(AUTO_AIM_PLANNER_SOURCES
  planner/src/planner.cpp
  planner/tinympc/src/tiny_api.cpp
  planner/tinympc/src/codegen.cpp
  planner/tinympc/src/rho_benchmark.cpp
  planner/tinympc/src/admm.cpp
)

add_library(auto_aim_planner STATIC ${AUTO_AIM_PLANNER_SOURCES})
add_library(auto_aim::planner ALIAS auto_aim_planner)

target_include_directories(auto_aim_planner
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/planner/include>
    $<BUILD_INTERFACE:${TINYMPC_INCLUDE_SRC_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_aim_planner
  PUBLIC
    Eigen3::Eigen
    auto_aim_predictor
    utils::utils
)

ament_target_dependencies(auto_aim_planner
  PUBLIC
    utils
)

# ====================================================================================
# auto_aim_pointer
# ====================================================================================
add_library(auto_aim_pointer STATIC
  pointer/src/image_processor.cpp
  pointer/src/lightbar_detector.cpp
)
add_library(auto_aim::pointer ALIAS auto_aim_pointer)

target_include_directories(auto_aim_pointer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/pointer/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_aim_pointer
  PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
)

# ====================================================================================
# auto_aim_aimer
# ====================================================================================
add_library(auto_aim_aimer STATIC
  aimer/src/aimer.cpp
)
add_library(auto_aim::aimer ALIAS auto_aim_aimer)

target_include_directories(auto_aim_aimer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/aimer/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_aim_aimer
  PUBLIC
    Eigen3::Eigen
    yaml-cpp
    fmt::fmt
    deveice::cboard
    auto_aim_predictor
    utils::utils
)

ament_target_dependencies(auto_aim_aimer
  PUBLIC
    rclcpp
)

# ====================================================================================
# auto_aim_tracker
# ====================================================================================
add_library(auto_aim_tracker STATIC
  tracker/src/tracker.cpp
)
add_library(auto_aim::tracker ALIAS auto_aim_tracker)

target_include_directories(auto_aim_tracker
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tracker/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_aim_tracker
  PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    yaml-cpp
    auto_aim_solver
    auto_aim_predictor
    auto_aim_aimer
    utils::utils
)

ament_target_dependencies(auto_aim_tracker
  PUBLIC
    rclcpp
    sensor_msgs
    geometry_msgs
    autoaim_msgs
    cv_bridge
)

# ====================================================================================
# 集成共享库
# ====================================================================================
set(AUTO_AIM_PUBLIC_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/aimer/include
  ${CMAKE_CURRENT_SOURCE_DIR}/detected/include
  ${CMAKE_CURRENT_SOURCE_DIR}/planner/include
  ${TINYMPC_INCLUDE_SRC_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/pointer/include
  ${CMAKE_CURRENT_SOURCE_DIR}/solver/include
  ${CMAKE_CURRENT_SOURCE_DIR}/target/predictor/include
  ${CMAKE_CURRENT_SOURCE_DIR}/target/motion_model/include
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/include
)
set(AUTO_AIM_DUMMY_SRC ${CMAKE_CURRENT_BINARY_DIR}/auto_aim_force_link.cpp)
if(NOT EXISTS ${AUTO_AIM_DUMMY_SRC})
  file(WRITE ${AUTO_AIM_DUMMY_SRC}
    "namespace auto_aim {\n"
    "void force_link_symbols() {}\n"
    "}\n")
endif()
set_source_files_properties(${AUTO_AIM_DUMMY_SRC} PROPERTIES GENERATED TRUE)

add_library(auto_aim SHARED ${AUTO_AIM_DUMMY_SRC})
add_library(auto_aim::auto_aim ALIAS auto_aim)

set(AUTO_AIM_MODULE_LIBS
  auto_aim_aimer
  auto_aim_detected
  auto_aim_planner
  auto_aim_pointer
  auto_aim_solver
  auto_aim_motion_model
  auto_aim_predictor
  auto_aim_tracker
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_link_libraries(auto_aim
    PUBLIC
      -Wl,--whole-archive
      ${AUTO_AIM_MODULE_LIBS}
      -Wl,--no-whole-archive
  )
else()
  target_link_libraries(auto_aim
    PUBLIC
      ${AUTO_AIM_MODULE_LIBS}
  )
endif()

target_include_directories(auto_aim
  PUBLIC
    $<BUILD_INTERFACE:${AUTO_AIM_PUBLIC_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
)

# ====================================================================================
# 安装与导出
# ====================================================================================
install(
  TARGETS
    auto_aim
    auto_aim_aimer
    auto_aim_detected
    auto_aim_planner
    auto_aim_pointer
    auto_aim_solver
    auto_aim_motion_model
    auto_aim_predictor
    auto_aim_tracker
  EXPORT auto_aimTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

foreach(dir IN ITEMS aimer detected planner pointer solver target/predictor target/motion_model tracker)
  install(
    DIRECTORY ${dir}/include/
    DESTINATION include
  )
endforeach()

install(
  DIRECTORY planner/tinympc/include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_targets(auto_aimTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  std_srvs
  autoaim_msgs
  utils
  deveice
  OpenCV
  OpenVINO
  Eigen3
  yaml-cpp
  fmt
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
