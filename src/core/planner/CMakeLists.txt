cmake_minimum_required(VERSION 3.8)
project(planner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(utils REQUIRED)
find_package(predictor REQUIRED)

set(TINYMPC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tinympc)
set(TINYMPC_INCLUDE_SRC_DIR ${TINYMPC_SOURCE_DIR}/include)
set(TINYMPC_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_includes)
set(TINYMPC_GENERATED_PUBLIC_DIR ${TINYMPC_GENERATED_INCLUDE_DIR}/tinympc)

file(MAKE_DIRECTORY ${TINYMPC_GENERATED_PUBLIC_DIR})

file(GLOB TINYMPC_HEADERS "${TINYMPC_INCLUDE_SRC_DIR}/*.hpp")
foreach(header ${TINYMPC_HEADERS})
  get_filename_component(header_name ${header} NAME)
  configure_file(${header} ${TINYMPC_GENERATED_PUBLIC_DIR}/${header_name} COPYONLY)
endforeach()

add_library(${PROJECT_NAME} SHARED
  src/planner.cpp
  tinympc/src/tiny_api.cpp
  tinympc/src/codegen.cpp
  tinympc/src/rho_benchmark.cpp
  tinympc/src/admm.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${TINYMPC_GENERATED_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${TINYMPC_INCLUDE_SRC_DIR}
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
)

ament_target_dependencies(${PROJECT_NAME}
  PUBLIC
    utils
    predictor
)

install(TARGETS ${PROJECT_NAME}
  EXPORT plannerTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY tinympc/include/
  DESTINATION include/tinympc
)

install(EXPORT plannerTargets
  FILE plannerTargets.cmake
  NAMESPACE planner::
  DESTINATION share/${PROJECT_NAME}/cmake
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(plannerTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(utils predictor Eigen3)

ament_package()
