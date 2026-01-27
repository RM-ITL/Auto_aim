cmake_minimum_required(VERSION 3.10)
project(auto_base LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(utils REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(deveice REQUIRED)

# ====================================================================================
# base_hit_detect 库
# ====================================================================================
set(AUTO_BASE_DETECT_SOURCES
  detect/src/openvino_infer.cpp
  detect/src/detector.cpp
)

add_library(auto_base_detect STATIC ${AUTO_BASE_DETECT_SOURCES})
add_library(auto_base::detect ALIAS auto_base_detect)

target_include_directories(auto_base_detect
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/detect/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_base_detect
  PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
    fmt::fmt
    openvino::runtime
    utils::utils
)

# ====================================================================================
# auto_base_target 库
# ====================================================================================
set(AUTO_BASE_TARGET_SOURCES
  target/filter/src/OneEurofilter.cpp
  target/predictor/src/light_model.cpp
)

add_library(auto_base_target STATIC ${AUTO_BASE_TARGET_SOURCES})
add_library(auto_base::target ALIAS auto_base_target)

target_include_directories(auto_base_target
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/target/filter/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/target/predictor/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/detect/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_base_target
  PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    utils::utils
    auto_base_detect
)

# ====================================================================================
# auto_base_tracker 库
# ====================================================================================
set(AUTO_BASE_TRACKER_SOURCES
  tracker/src/light_tracker.cpp
)

add_library(auto_base_tracker STATIC ${AUTO_BASE_TRACKER_SOURCES})
add_library(auto_base::tracker ALIAS auto_base_tracker)

target_include_directories(auto_base_tracker
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tracker/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_base_tracker
  PUBLIC
    auto_base_detect
    auto_base_target
    deveice::lower_dart
    Eigen3::Eigen
)

# ====================================================================================
# 集成共享库
# ====================================================================================
set(AUTO_BASE_PUBLIC_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/detect/include
  ${CMAKE_CURRENT_SOURCE_DIR}/target/filter/include
  ${CMAKE_CURRENT_SOURCE_DIR}/target/predictor/include
  ${CMAKE_CURRENT_SOURCE_DIR}/tracker/include
)

set(AUTO_BASE_DUMMY_SRC ${CMAKE_CURRENT_BINARY_DIR}/auto_base_force_link.cpp)
if(NOT EXISTS ${AUTO_BASE_DUMMY_SRC})
  file(WRITE ${AUTO_BASE_DUMMY_SRC}
    "namespace auto_base {\n"
    "void force_link_symbols() {}\n"
    "}\n")
endif()
set_source_files_properties(${AUTO_BASE_DUMMY_SRC} PROPERTIES GENERATED TRUE)

add_library(auto_base SHARED ${AUTO_BASE_DUMMY_SRC})
add_library(auto_base::auto_base ALIAS auto_base)

set(AUTO_BASE_MODULE_LIBS
  auto_base_detect
  auto_base_target
  auto_base_tracker
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_link_libraries(auto_base
    PUBLIC
      -Wl,--whole-archive
      ${AUTO_BASE_MODULE_LIBS}
      -Wl,--no-whole-archive
  )
else()
  target_link_libraries(auto_base
    PUBLIC
      ${AUTO_BASE_MODULE_LIBS}
  )
endif()

target_include_directories(auto_base
  PUBLIC
    $<BUILD_INTERFACE:${AUTO_BASE_PUBLIC_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
)

# ====================================================================================
# 安装与导出
# ====================================================================================
install(
  TARGETS
    auto_base
    auto_base_detect
    auto_base_target
    auto_base_tracker
  EXPORT auto_baseTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  DIRECTORY detect/include/
  DESTINATION include
)

install(
  DIRECTORY target/filter/include/
  DESTINATION include
)

install(
  DIRECTORY target/predictor/include/
  DESTINATION include
)

install(
  DIRECTORY tracker/include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_targets(auto_baseTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  OpenCV
  OpenVINO
  yaml-cpp
  fmt
  utils
  Eigen3
  deveice
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
