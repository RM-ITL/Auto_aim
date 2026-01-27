cmake_minimum_required(VERSION 3.10)
project(deveice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(utils REQUIRED)
find_package(serial REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)
pkg_check_modules(LIBSERIAL REQUIRED libserial)

set(MV_CAMERA_SDK_PATH "/opt/MVS" CACHE PATH "Path to Hikvision MVS SDK")
set(MV_CAMERA_INCLUDE_DIR "${MV_CAMERA_SDK_PATH}/include")
set(MV_CAMERA_LIB_DIR "${MV_CAMERA_SDK_PATH}/lib/64")

if(NOT EXISTS "${MV_CAMERA_LIB_DIR}/libMvCameraControl.so")
  message(FATAL_ERROR "Hikvision camera library not found at: ${MV_CAMERA_LIB_DIR}/libMvCameraControl.so")
endif()

# MindVision SDK 路径配置
set(MINDVISION_SDK_PATH "/home/guo/linux_sdk" CACHE PATH "Path to MindVision SDK")
set(MINDVISION_INCLUDE_DIR "${MINDVISION_SDK_PATH}/include")
set(MINDVISION_LIB_DIR "${MINDVISION_SDK_PATH}/lib/x64")

if(NOT EXISTS "${MINDVISION_LIB_DIR}/libMVSDK.so")
  message(WARNING "MindVision camera library not found at: ${MINDVISION_LIB_DIR}/libMVSDK.so")
endif()

# ====================================================================================
# deveice_camera
# ====================================================================================
set(deveice_CAMERA_SOURCES
  camera/src/hikcamera.cpp
  camera/src/mindvision.cpp
  camera/src/camera.cpp
)

add_library(deveice_camera STATIC ${deveice_CAMERA_SOURCES})
add_library(deveice::camera ALIAS deveice_camera)
set_target_properties(deveice_camera PROPERTIES EXPORT_NAME camera)

target_include_directories(deveice_camera
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/camera/include>
    $<INSTALL_INTERFACE:include>
    ${MV_CAMERA_INCLUDE_DIR}
    ${MINDVISION_INCLUDE_DIR}
    ${LIBUSB_INCLUDE_DIRS}
)

target_link_libraries(deveice_camera
  PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
    utils::utils
    pthread
    ${LIBUSB_LIBRARIES}
    ${MV_CAMERA_LIB_DIR}/libMvCameraControl.so
    ${MV_CAMERA_LIB_DIR}/libMvCameraControlWrapper.so
    ${MINDVISION_LIB_DIR}/libMVSDK.so
)

target_compile_options(deveice_camera PRIVATE ${LIBUSB_CFLAGS_OTHER})

ament_target_dependencies(deveice_camera
  PUBLIC
    rclcpp
    utils
)

# ====================================================================================
# deveice_cboard
# ====================================================================================
add_library(deveice_cboard STATIC
  cboard/src/cboard.cpp
)
add_library(deveice::cboard ALIAS deveice_cboard)
set_target_properties(deveice_cboard PROPERTIES EXPORT_NAME cboard)

target_include_directories(deveice_cboard
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cboard/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(deveice_cboard
  PUBLIC
    Eigen3::Eigen
    yaml-cpp
    utils::utils
)

ament_target_dependencies(deveice_cboard
  PUBLIC
    rclcpp
    utils
)

# ====================================================================================
# deveice_gimbal
# ====================================================================================
add_library(deveice_gimbal STATIC
  gimbal/src/gimbal.cpp
)
add_library(deveice::gimbal ALIAS deveice_gimbal)
set_target_properties(deveice_gimbal PROPERTIES EXPORT_NAME gimbal)

target_include_directories(deveice_gimbal
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gimbal/include>
    $<INSTALL_INTERFACE:include>
    ${LIBSERIAL_INCLUDE_DIRS}
)

target_link_libraries(deveice_gimbal
  PUBLIC
    yaml-cpp
    utils::utils
    serial
    ${LIBSERIAL_LIBRARIES}
)

ament_target_dependencies(deveice_gimbal
  PUBLIC
    rclcpp
    utils
    serial
)

# ====================================================================================
# deveice_imu
# ====================================================================================
add_library(deveice_imu STATIC
  imu/src/imu_driver.cpp
)
add_library(deveice::imu ALIAS deveice_imu)
set_target_properties(deveice_imu PROPERTIES EXPORT_NAME imu)

target_include_directories(deveice_imu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/imu/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(deveice_imu
  PUBLIC
    Eigen3::Eigen
    utils::utils
    serial
)

ament_target_dependencies(deveice_imu
  PUBLIC
    Eigen3
    utils
    serial
)

# ====================================================================================
# deveice_old_gimbal
# ====================================================================================
add_library(deveice_old_gimbal STATIC
  old_gimbal/src/old_gimbal.cpp
)
add_library(deveice::old_gimbal ALIAS deveice_old_gimbal)
set_target_properties(deveice_old_gimbal PROPERTIES EXPORT_NAME old_gimbal)

target_include_directories(deveice_old_gimbal
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/old_gimbal/include>
    $<INSTALL_INTERFACE:include>
    ${LIBSERIAL_INCLUDE_DIRS}
)

target_link_libraries(deveice_old_gimbal
  PUBLIC
    yaml-cpp
    utils::utils
    serial
    ${LIBSERIAL_LIBRARIES}
)

ament_target_dependencies(deveice_old_gimbal
  PUBLIC
    rclcpp
    utils
    serial
)

# ====================================================================================
# deveice_lower_dart
# ====================================================================================
add_library(deveice_lower_dart STATIC
  lower_dart/src/lower_dart.cpp
)
add_library(deveice::lower_dart ALIAS deveice_lower_dart)
set_target_properties(deveice_lower_dart PROPERTIES EXPORT_NAME lower_dart)

target_include_directories(deveice_lower_dart
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lower_dart/include>
    $<INSTALL_INTERFACE:include>
    ${LIBSERIAL_INCLUDE_DIRS}
)

target_link_libraries(deveice_lower_dart
  PUBLIC
    Eigen3::Eigen
    yaml-cpp
    utils::utils
    serial
    ${LIBSERIAL_LIBRARIES}
)

ament_target_dependencies(deveice_lower_dart
  PUBLIC
    rclcpp
    utils
    serial
    Eigen3
)

# ====================================================================================
# 集成共享库
# ====================================================================================
set(deveice_PUBLIC_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/camera/include
  ${CMAKE_CURRENT_SOURCE_DIR}/cboard/include
  ${CMAKE_CURRENT_SOURCE_DIR}/gimbal/include
  ${CMAKE_CURRENT_SOURCE_DIR}/imu/include
  ${CMAKE_CURRENT_SOURCE_DIR}/old_gimbal/include
  ${CMAKE_CURRENT_SOURCE_DIR}/lower_dart/include
  ${MV_CAMERA_INCLUDE_DIR}
  ${MINDVISION_INCLUDE_DIR}
  ${LIBUSB_INCLUDE_DIRS}
  ${LIBSERIAL_INCLUDE_DIRS}
)

set(deveice_DUMMY_SRC ${CMAKE_CURRENT_BINARY_DIR}/deveice_force_link.cpp)
if(NOT EXISTS ${deveice_DUMMY_SRC})
  file(WRITE ${deveice_DUMMY_SRC}
    "namespace deveice {\n"
    "void force_link_symbols() {}\n"
    "}\n")
endif()
set_source_files_properties(${deveice_DUMMY_SRC} PROPERTIES GENERATED TRUE)

add_library(deveice SHARED ${deveice_DUMMY_SRC})
add_library(deveice::deveice ALIAS deveice)

set(deveice_MODULE_LIBS
  deveice_camera
  deveice_cboard
  deveice_gimbal
  deveice_imu
  deveice_old_gimbal
  deveice_lower_dart
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_link_libraries(deveice
    PUBLIC
      -Wl,--whole-archive
      ${deveice_MODULE_LIBS}
      -Wl,--no-whole-archive
  )
else()
  target_link_libraries(deveice
    PUBLIC
      ${deveice_MODULE_LIBS}
  )
endif()

target_include_directories(deveice
  PUBLIC
    $<BUILD_INTERFACE:${deveice_PUBLIC_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(deveice PROPERTIES
  BUILD_RPATH "${MV_CAMERA_LIB_DIR};${MINDVISION_LIB_DIR}"
  INSTALL_RPATH "${MV_CAMERA_LIB_DIR};${MINDVISION_LIB_DIR}"
)

# ====================================================================================
# 安装与导出
# ====================================================================================
install(
  TARGETS
    deveice
    deveice_camera
    deveice_cboard
    deveice_gimbal
    deveice_imu
    deveice_old_gimbal
    deveice_lower_dart
  EXPORT deveiceTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

foreach(dir IN ITEMS camera/include cboard/include gimbal/include imu/include old_gimbal/include lower_dart/include)
  install(
    DIRECTORY ${dir}/
    DESTINATION include
  )
endforeach()

install(
  DIRECTORY imu/config/
  DESTINATION share/${PROJECT_NAME}/imu/config
  OPTIONAL
)

install(
  EXPORT deveiceTargets
  FILE deveiceTargets.cmake
  NAMESPACE deveice::
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_export_include_directories(include ${MV_CAMERA_INCLUDE_DIR} ${MINDVISION_INCLUDE_DIR} ${LIBUSB_INCLUDE_DIRS} ${LIBSERIAL_INCLUDE_DIRS})
ament_export_targets(deveiceTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  utils
  serial
  Eigen3
  OpenCV
  yaml-cpp
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
