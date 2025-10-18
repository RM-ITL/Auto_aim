cmake_minimum_required(VERSION 3.8)
project(dm_imu)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# 查找libserial库
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBSERIAL REQUIRED libserial)

# 添加包含目录 - 修正路径
include_directories(include)
include_directories(${LIBSERIAL_INCLUDE_DIRS})

# 创建主要的可执行文件 dm_imu_node (保持与ROS1版本一致的命名)
add_executable(${PROJECT_NAME}_node
  src/bsp_crc.cpp
  src/imu_driver.cpp
  src/test_imu.cpp
)

target_include_directories(${PROJECT_NAME}_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${LIBSERIAL_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}_node 
  ${LIBSERIAL_LIBRARIES}
)

target_link_directories(${PROJECT_NAME}_node PUBLIC
  ${LIBSERIAL_LIBRARY_DIRS}
)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_geometry_msgs
)

# 可选：创建库文件供其他包使用
add_library(${PROJECT_NAME} 
  src/imu_driver.cpp
  src/bsp_crc.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${LIBSERIAL_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${LIBSERIAL_LIBRARIES}
)

target_link_directories(${PROJECT_NAME} PUBLIC
  ${LIBSERIAL_LIBRARY_DIRS}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_geometry_msgs
)

# 安装
install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
  OPTIONAL
)

install(DIRECTORY
  rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
  OPTIONAL
)

install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(rclcpp std_msgs sensor_msgs geometry_msgs tf2 tf2_geometry_msgs)

ament_package()