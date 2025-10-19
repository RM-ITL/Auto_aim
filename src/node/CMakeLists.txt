cmake_minimum_required(VERSION 3.8)
project(pipeline)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(hikvision REQUIRED)
find_package(utils REQUIRED)
find_package(serial REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(dm_imu REQUIRED)
find_package(detected REQUIRED)
find_package(solver REQUIRED)
find_package(tracker REQUIRED)
find_package(planner REQUIRED)
find_package(autoaim_msgs REQUIRED)

add_executable(pipeline_node
  src/test_node.cpp
)

target_include_directories(pipeline_node PRIVATE
  include
)

ament_target_dependencies(pipeline_node
  rclcpp
  Eigen3
  hikvision
  utils
  serial
  dm_imu
  detected
  solver
  tracker
  planner
  autoaim_msgs
)

target_link_libraries(pipeline_node
  ${OpenCV_LIBS}
)

install(TARGETS
  pipeline_node
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
