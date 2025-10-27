cmake_minimum_required(VERSION 3.8)
project(detected)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(utils REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
  ${OpenVINO_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
)

set(DATA_STRUCTURE_SOURCES
  src/common/armor.cpp
)

set(DETECTOR_SOURCES
  src/detector/common_detector.cpp
  src/detector/yolo11.cpp
)

set(CLASSIFIER_SOURCES
  src/classifier/armor_classifier.cpp
)

set(MODULE_SOURCES
  src/node/armor_detector_node.cpp
)

add_library(armor_data_lib SHARED
  ${DATA_STRUCTURE_SOURCES}
)

target_link_libraries(armor_data_lib
  ${OpenCV_LIBS}
)

add_library(armor_detector_lib SHARED
  ${DETECTOR_SOURCES}
)

target_include_directories(armor_detector_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenVINO_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
)

target_link_libraries(armor_detector_lib
  armor_data_lib
  ${OpenCV_LIBS}
  yaml-cpp
  openvino::runtime
  utils::utils
)

ament_target_dependencies(armor_detector_lib
  utils
)

add_library(armor_classifier_lib SHARED
  ${CLASSIFIER_SOURCES}
)

target_include_directories(armor_classifier_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenVINO_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
)

target_link_libraries(armor_classifier_lib
  armor_data_lib
  ${OpenCV_LIBS}
  yaml-cpp
  openvino::runtime
  utils::utils
)

ament_target_dependencies(armor_classifier_lib
  utils
)

add_library(detected SHARED
  ${MODULE_SOURCES}
)

target_include_directories(detected PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenVINO_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../utils/include
)

target_link_libraries(detected
  armor_detector_lib
  ${OpenCV_LIBS}
  yaml-cpp
  utils::utils
)

ament_target_dependencies(detected
  utils
)

install(TARGETS
  armor_data_lib
  armor_detector_lib
  armor_classifier_lib
  detected
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(
  OpenCV
  yaml-cpp
  OpenVINO
  utils
)

ament_export_libraries(
  armor_data_lib
  armor_detector_lib
  armor_classifier_lib
  detected
)

ament_export_include_directories(include)

ament_package()
