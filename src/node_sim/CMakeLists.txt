cmake_minimum_required(VERSION 3.10)
project(sim_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json 3.2 REQUIRED)

find_package(utils REQUIRED)
find_package(app_config REQUIRED)
find_package(deveice REQUIRED)
find_package(auto_aim REQUIRED)

add_executable(sim_node
  src/sim_node.cpp
  src/sim_client.cpp
)

target_include_directories(sim_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(sim_node
  rclcpp
  Eigen3
  utils
  app_config
  deveice
  auto_aim
)

target_link_libraries(sim_node
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
  deveice::deveice
  auto_aim::auto_aim
  app_config::app_config
)

install(TARGETS sim_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
