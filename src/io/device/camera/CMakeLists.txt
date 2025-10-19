cmake_minimum_required(VERSION 3.8)
project(hikvision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(utils REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

set(MV_CAMERA_SDK_PATH "/opt/MVS" CACHE PATH "Path to Hikvision MVS SDK")
set(MV_CAMERA_INCLUDE_DIR ${MV_CAMERA_SDK_PATH}/include)
set(MV_CAMERA_LIB_DIR ${MV_CAMERA_SDK_PATH}/lib/64)

if(NOT EXISTS "${MV_CAMERA_LIB_DIR}/libMvCameraControl.so")
  message(FATAL_ERROR "Hikvision camera library not found at: ${MV_CAMERA_LIB_DIR}/libMvCameraControl.so")
endif()

set(CAMERA_CATCH_SRC
  src/hikcamera.cpp
)

add_library(hikcamera SHARED ${CAMERA_CATCH_SRC})
add_library(hikcamera::hikcamera ALIAS hikcamera)

target_include_directories(hikcamera PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${MV_CAMERA_INCLUDE_DIR}
  ${LIBUSB_INCLUDE_DIRS}
)

ament_target_dependencies(
  hikcamera
  rclcpp
  utils
)

target_link_libraries(hikcamera
  ${OpenCV_LIBS}
  yaml-cpp
  utils::utils
  ${MV_CAMERA_LIB_DIR}/libMvCameraControl.so
  ${MV_CAMERA_LIB_DIR}/libMvCameraControlWrapper.so
  pthread
  ${LIBUSB_LIBRARIES}
)

target_compile_options(hikcamera PRIVATE ${LIBUSB_CFLAGS_OTHER})

set_target_properties(hikcamera PROPERTIES 
  INSTALL_RPATH "${MV_CAMERA_LIB_DIR}"
  BUILD_RPATH "${MV_CAMERA_LIB_DIR}"
)

install(TARGETS
  hikcamera
  EXPORT hikcameraTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY
  include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(
  rclcpp
  OpenCV
  yaml-cpp
  utils
)

ament_export_include_directories(
  include
  ${MV_CAMERA_INCLUDE_DIR}
  ${LIBUSB_INCLUDE_DIRS}
)

ament_export_libraries(hikcamera)

ament_export_targets(hikcameraTargets HAS_LIBRARY_TARGET)

install(
  EXPORT hikcameraTargets
  FILE hikcameraTargets.cmake
  NAMESPACE hikcamera::
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_package()
