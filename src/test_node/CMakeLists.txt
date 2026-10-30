cmake_minimum_required(VERSION 3.8)
project(test_pipeline)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(utils REQUIRED)
find_package(app_config REQUIRED)
find_package(serial REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(autoaim_msgs REQUIRED)
find_package(deveice REQUIRED)
find_package(auto_aim REQUIRED)
find_package(auto_buff REQUIRED)
find_package(auto_base REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Ceres 2.0 QUIET)

add_executable(deep_node
  src/test_node_deep.cpp
)

add_executable(video_node
  src/video_node.cpp
)

add_executable(base_hit_node
  src/base_hit_node.cpp
)

add_executable(cv_node
  src/test_node_cv.cpp
)

add_executable(aimer_node
  src/test_node_aimer.cpp
)

add_executable(capture_node
  src/capture_node.cpp
)

add_executable(gimbal_comm_test
  src/gimbal_comm_test.cpp
)

add_executable(camera_calib_node
  src/camera_calib_node.cpp
)

add_executable(handeye_calib_node
  src/handeye_calib_node.cpp
)

add_executable(gimbal_imu_calib_node
  src/gimbal_imu_calib_node.cpp
)

add_executable(record_node
  src/record_node.cpp
)

add_executable(usb_detect_node
  src/usb_detect_node.cpp
)

add_executable(replay_node
  src/replay_node.cpp
)

target_include_directories(deep_node PRIVATE
  include
)

target_include_directories(video_node PRIVATE
  include
)

target_include_directories(base_hit_node PRIVATE
  include
)

target_include_directories(cv_node PRIVATE
  include
)

target_include_directories(capture_node PRIVATE
  include
)

target_include_directories(camera_calib_node PRIVATE
  include
)

target_include_directories(handeye_calib_node PRIVATE
  include
)

target_include_directories(gimbal_imu_calib_node PRIVATE
  include
)

target_include_directories(aimer_node  PRIVATE
  include
)

target_include_directories(record_node PRIVATE
  include
)

target_include_directories(usb_detect_node PRIVATE
  include
)

target_include_directories(replay_node PRIVATE
  include
)

ament_target_dependencies(deep_node
  rclcpp
  Eigen3
  deveice
  auto_aim
  utils
  app_config
  serial
  autoaim_msgs
)

ament_target_dependencies(video_node
  rclcpp
  Eigen3
  utils
  app_config
  deveice
  auto_aim
  autoaim_msgs
)

ament_target_dependencies(base_hit_node
  rclcpp
  autoaim_msgs
  utils
  app_config
  deveice
  auto_base
)

ament_target_dependencies(cv_node
  rclcpp
  Eigen3
  deveice
  auto_aim
  utils
  app_config
  serial
  autoaim_msgs
)

ament_target_dependencies(capture_node
  rclcpp
  Eigen3
  deveice
  utils
  app_config
  serial
)

ament_target_dependencies(gimbal_comm_test
  utils
  serial
)

ament_target_dependencies(camera_calib_node
  utils
)

ament_target_dependencies(handeye_calib_node
  utils
  app_config
)

ament_target_dependencies(gimbal_imu_calib_node
  utils
  app_config
)

ament_target_dependencies(aimer_node
  rclcpp
  Eigen3
  deveice
  auto_aim
  utils
  app_config
  serial
  autoaim_msgs
)

ament_target_dependencies(record_node
  rclcpp
  Eigen3
  deveice
  utils
  app_config
  serial
  autoaim_msgs
  sensor_msgs
  cv_bridge
)

ament_target_dependencies(usb_detect_node
  rclcpp
  Eigen3
  deveice
  auto_aim
  utils
  app_config
  serial
  autoaim_msgs
)

ament_target_dependencies(replay_node
  rclcpp
  Eigen3
  auto_aim
  utils
  app_config
  autoaim_msgs
  sensor_msgs
  cv_bridge
)

target_link_libraries(deep_node
  ${OpenCV_LIBS}
  deveice::deveice
  auto_aim::auto_aim
)

target_link_libraries(usb_detect_node
  ${OpenCV_LIBS}
  deveice::deveice
  auto_aim::auto_aim
  yaml-cpp
)

# libutils.a 同时静态链进了本可执行文件、libauto_aim.so、libdeveice.so，三份 utils::logger_ 等全局量。
# 不隐藏静态库导出符号时，本节点进程退出析构阶段稳定报 "corrupted double-linked list" 崩溃
# (2026-10-29 实测；加上本选项后 glibc malloc 检查无报错)。
target_link_options(usb_detect_node PRIVATE "LINKER:--exclude-libs,ALL")

target_link_libraries(video_node
  ${OpenCV_LIBS}
  deveice::deveice
  auto_aim::auto_aim
)

target_link_libraries(base_hit_node
  ${OpenCV_LIBS}
  deveice::deveice
  auto_base::auto_base
)

target_link_libraries(cv_node
  ${OpenCV_LIBS}
  deveice::deveice
  auto_aim::auto_aim
)

target_link_libraries(capture_node
  ${OpenCV_LIBS}
  deveice::deveice
)

target_link_libraries(camera_calib_node
  ${OpenCV_LIBS}
  yaml-cpp
)

target_link_libraries(handeye_calib_node
  ${OpenCV_LIBS}
  yaml-cpp
)

target_link_libraries(gimbal_imu_calib_node
  ${OpenCV_LIBS}
  yaml-cpp
)

if(Ceres_FOUND)
  target_compile_definitions(handeye_calib_node PRIVATE HANDEYE_WITH_CERES)
  target_link_libraries(handeye_calib_node Ceres::ceres)
  message(STATUS "handeye_calib_node: Ceres ${Ceres_VERSION} found, BA enabled")
else()
  message(STATUS "handeye_calib_node: Ceres not found, BA disabled (closed-form solver still builds)")
endif()

target_link_libraries(aimer_node
  ${OpenCV_LIBS}
  deveice::deveice
  auto_aim::auto_aim
)

target_link_libraries(record_node
  ${OpenCV_LIBS}
  deveice::deveice
)

target_link_libraries(replay_node
  ${OpenCV_LIBS}
  auto_aim::auto_aim
)

install(TARGETS
  deep_node
  video_node
  base_hit_node
  cv_node
  capture_node
  camera_calib_node
  handeye_calib_node
  gimbal_imu_calib_node
  aimer_node
  gimbal_comm_test
  record_node
  usb_detect_node
  replay_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
