cmake_minimum_required(VERSION 3.10)
project(auto_buff LANGUAGES CXX)

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
find_package(Eigen3 REQUIRED)
find_package(utils REQUIRED)
find_package(serial REQUIRED)
find_package(deveice REQUIRED)
find_package(auto_aim REQUIRED)

# ====================================================================================
# auto_buff_data_type
# ====================================================================================
set(AUTO_BUFF_DATA_TYPE_SOURCES
  buff_data_type/src/buff_type.cpp
)

add_library(auto_buff_data_type STATIC ${AUTO_BUFF_DATA_TYPE_SOURCES})
add_library(auto_buff::data_type ALIAS auto_buff_data_type)

target_include_directories(auto_buff_data_type
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/buff_data_type/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_buff_data_type
  PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    utils::utils
)

ament_target_dependencies(auto_buff_data_type
  PUBLIC
    utils
)

# ====================================================================================
# auto_buff_detect
# ====================================================================================
set(AUTO_BUFF_DETECT_SOURCES
  buff_detect/src/buff_detector.cpp
  buff_detect/src/yolo11_buff.cpp
)

add_library(auto_buff_detect STATIC ${AUTO_BUFF_DETECT_SOURCES})
add_library(auto_buff::detect ALIAS auto_buff_detect)

target_include_directories(auto_buff_detect
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/buff_detect/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_buff_detect
  PUBLIC
    ${OpenCV_LIBS}
    openvino::runtime
    yaml-cpp
    utils::utils
    auto_buff_data_type
)

ament_target_dependencies(auto_buff_detect
  PUBLIC
    utils
)

# ====================================================================================
# auto_buff_solver
# ====================================================================================
set(AUTO_BUFF_SOLVER_SOURCES
  buff_solver/src/buff_solver.cpp
)

add_library(auto_buff_solver STATIC ${AUTO_BUFF_SOLVER_SOURCES})
add_library(auto_buff::solver ALIAS auto_buff_solver)

target_include_directories(auto_buff_solver
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/buff_solver/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_buff_solver
  PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    yaml-cpp
    utils::utils
    auto_buff_data_type
)

ament_target_dependencies(auto_buff_solver
  PUBLIC
    utils
)

# ====================================================================================
# auto_buff_tagret
# ====================================================================================
set(AUTO_BUFF_TAGRET_SOURCES
  buff_tagret/src/buff_target.cpp
)

add_library(auto_buff_tagret STATIC ${AUTO_BUFF_TAGRET_SOURCES})
add_library(auto_buff::tagret ALIAS auto_buff_tagret)

target_include_directories(auto_buff_tagret
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/buff_tagret/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_buff_tagret
  PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    utils::utils
    auto_buff_data_type
    auto_buff_detect
)

ament_target_dependencies(auto_buff_tagret
  PUBLIC
    utils
)

# ====================================================================================
# auto_buff_aimer
# ====================================================================================
set(AUTO_BUFF_AIMER_SOURCES
  buff_aimer/src/buff_aimer.cpp
)

add_library(auto_buff_aimer STATIC ${AUTO_BUFF_AIMER_SOURCES})
add_library(auto_buff::aimer ALIAS auto_buff_aimer)

target_include_directories(auto_buff_aimer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/buff_aimer/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(auto_buff_aimer
  PUBLIC
    Eigen3::Eigen
    yaml-cpp
    utils::utils
    auto_buff_data_type
    auto_buff_detect
    auto_buff_tagret
    auto_aim::auto_aim
    deveice::deveice
)

ament_target_dependencies(auto_buff_aimer
  PUBLIC
    utils
    auto_aim
    deveice
)

# ====================================================================================
# 集成共享库
# ====================================================================================
set(AUTO_BUFF_PUBLIC_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/buff_data_type/include
  ${CMAKE_CURRENT_SOURCE_DIR}/buff_detect/include
  ${CMAKE_CURRENT_SOURCE_DIR}/buff_solver/include
  ${CMAKE_CURRENT_SOURCE_DIR}/buff_tagret/include
  ${CMAKE_CURRENT_SOURCE_DIR}/buff_aimer/include
)

set(AUTO_BUFF_DUMMY_SRC ${CMAKE_CURRENT_BINARY_DIR}/auto_buff_force_link.cpp)
if(NOT EXISTS ${AUTO_BUFF_DUMMY_SRC})
  file(WRITE ${AUTO_BUFF_DUMMY_SRC}
    "namespace auto_buff {\n"
    "void force_link_symbols() {}\n"
    "}\n")
endif()
set_source_files_properties(${AUTO_BUFF_DUMMY_SRC} PROPERTIES GENERATED TRUE)

add_library(auto_buff SHARED ${AUTO_BUFF_DUMMY_SRC})
add_library(auto_buff::auto_buff ALIAS auto_buff)

set(AUTO_BUFF_MODULE_LIBS
  auto_buff_data_type
  auto_buff_detect
  auto_buff_solver
  auto_buff_tagret
  auto_buff_aimer
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_link_libraries(auto_buff
    PUBLIC
      -Wl,--whole-archive
      ${AUTO_BUFF_MODULE_LIBS}
      -Wl,--no-whole-archive
  )
else()
  target_link_libraries(auto_buff
    PUBLIC
      ${AUTO_BUFF_MODULE_LIBS}
  )
endif()

target_include_directories(auto_buff
  PUBLIC
    $<BUILD_INTERFACE:${AUTO_BUFF_PUBLIC_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
)

# ====================================================================================
# 安装与导出
# ====================================================================================
install(
  TARGETS
    auto_buff
    auto_buff_data_type
    auto_buff_detect
    auto_buff_solver
    auto_buff_tagret
    auto_buff_aimer
  EXPORT auto_buffTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

foreach(dir IN ITEMS buff_data_type buff_detect buff_solver buff_tagret buff_aimer)
  install(
    DIRECTORY ${dir}/include/
    DESTINATION include
  )
endforeach()

ament_export_include_directories(include)
ament_export_targets(auto_buffTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  OpenCV
  OpenVINO
  Eigen3
  yaml-cpp
  utils
  auto_aim
  deveice
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
