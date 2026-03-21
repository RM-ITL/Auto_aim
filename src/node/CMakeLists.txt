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
find_package(utils REQUIRED)
find_package(serial REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(autoaim_msgs REQUIRED)
find_package(deveice REQUIRED)
find_package(auto_aim REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(sentry_node
  src/sentry.cpp
)

add_executable(standard3_node
  src/standard3.cpp
)

target_include_directories(sentry_node PRIVATE
  include
)

target_include_directories(standard3_node PRIVATE
  include
)

ament_target_dependencies(sentry_node
  rclcpp
  Eigen3
  deveice
  auto_aim
  utils
  serial
  autoaim_msgs
  sensor_msgs
  geometry_msgs
  std_msgs
)

ament_target_dependencies(standard3_node
  Eigen3
  deveice
  auto_aim
  utils
  serial
)

target_link_libraries(sentry_node
  ${OpenCV_LIBS}
  deveice::deveice
  auto_aim::auto_aim
)

target_link_libraries(standard3_node
  ${OpenCV_LIBS}
  deveice::deveice
  auto_aim::auto_aim
)

install(TARGETS
  sentry_node
  standard3_node
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
