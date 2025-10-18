cmake_minimum_required(VERSION 3.8)
project(hikvision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(autoaim_msgs REQUIRED)
find_package(utils REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

set(MV_CAMERA_SDK_PATH "/opt/MVS" CACHE PATH "Path to Hikvision MVS SDK")
set(MV_CAMERA_INCLUDE_DIR ${MV_CAMERA_SDK_PATH}/include)
set(MV_CAMERA_LIB_DIR ${MV_CAMERA_SDK_PATH}/lib/64)

if(NOT EXISTS "${MV_CAMERA_LIB_DIR}/libMvCameraControl.so")
  message(FATAL_ERROR "Hikvision camera library not found at: ${MV_CAMERA_LIB_DIR}/libMvCameraControl.so")
endif()

include_directories(
  include
  ${MV_CAMERA_INCLUDE_DIR}
  ${LIBUSB_INCLUDE_DIRS}
  utils
)

set(CAMERA_CATCH_SRC
  src/hikcamera.cpp
)

add_executable(hik_camera_node ${CAMERA_CATCH_SRC})

target_include_directories(hik_camera_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${MV_CAMERA_INCLUDE_DIR}
  ${LIBUSB_INCLUDE_DIRS}
  utils
)

ament_target_dependencies(
  hik_camera_node
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  image_transport
  utils
)

target_link_libraries(hik_camera_node
  yaml-cpp
  ${MV_CAMERA_LIB_DIR}/libMvCameraControl.so
  ${MV_CAMERA_LIB_DIR}/libMvCameraControlWrapper.so
  pthread
  ${LIBUSB_LIBRARIES}
)

target_compile_options(hik_camera_node PRIVATE ${LIBUSB_CFLAGS_OTHER})

set_target_properties(hik_camera_node PROPERTIES 
  INSTALL_RPATH "${MV_CAMERA_LIB_DIR}"
  BUILD_RPATH "${MV_CAMERA_LIB_DIR}"
)

install(TARGETS
  hik_camera_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  autoaim_msgs
  OpenCV
  yaml-cpp
  image_transport
  OpenVINO
  utils
)

ament_package()