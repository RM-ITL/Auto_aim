cmake_minimum_required(VERSION 3.10)
project(base_hit LANGUAGES CXX)

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

# ====================================================================================
# base_hit_detect 库
# ====================================================================================
set(BASE_HIT_DETECT_SOURCES
  detect/src/openvino_infer.cpp
  detect/src/detector.cpp
)

add_library(base_hit_detect STATIC ${BASE_HIT_DETECT_SOURCES})
add_library(base_hit::detect ALIAS base_hit_detect)

target_include_directories(base_hit_detect
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/detect/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(base_hit_detect
  PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
    fmt::fmt
    openvino::runtime
    utils::utils
)

# ====================================================================================
# 集成共享库
# ====================================================================================
set(BASE_HIT_PUBLIC_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/detect/include
)

set(BASE_HIT_DUMMY_SRC ${CMAKE_CURRENT_BINARY_DIR}/base_hit_force_link.cpp)
if(NOT EXISTS ${BASE_HIT_DUMMY_SRC})
  file(WRITE ${BASE_HIT_DUMMY_SRC}
    "namespace base_hit {\n"
    "void force_link_symbols() {}\n"
    "}\n")
endif()
set_source_files_properties(${BASE_HIT_DUMMY_SRC} PROPERTIES GENERATED TRUE)

add_library(base_hit SHARED ${BASE_HIT_DUMMY_SRC})
add_library(base_hit::base_hit ALIAS base_hit)

set(BASE_HIT_MODULE_LIBS
  base_hit_detect
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_link_libraries(base_hit
    PUBLIC
      -Wl,--whole-archive
      ${BASE_HIT_MODULE_LIBS}
      -Wl,--no-whole-archive
  )
else()
  target_link_libraries(base_hit
    PUBLIC
      ${BASE_HIT_MODULE_LIBS}
  )
endif()

target_include_directories(base_hit
  PUBLIC
    $<BUILD_INTERFACE:${BASE_HIT_PUBLIC_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
)

# ====================================================================================
# 安装与导出
# ====================================================================================
install(
  TARGETS
    base_hit
    base_hit_detect
  EXPORT base_hitTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  DIRECTORY detect/include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_targets(base_hitTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  OpenCV
  OpenVINO
  yaml-cpp
  fmt
  utils
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
