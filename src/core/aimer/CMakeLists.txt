cmake_minimum_required(VERSION 3.8)
project(aimer)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

find_package(utils REQUIRED)
find_package(cboard REQUIRED)
find_package(detected REQUIRED)
find_package(predictor REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/aimer.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  # BUILD_INTERFACE：编译时使用的路径
  # 指向当前源码目录的include文件夹
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  
  ${EIGEN3_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
  ${fmt_INCLUDE_DIRS}
)

if(DEFINED EIGEN3_DEFINITIONS)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ${EIGEN3_DEFINITIONS})
endif()

target_link_libraries(${PROJECT_NAME}
  utils::utils
  yaml-cpp
  fmt
  cboard::cboard
  # Eigen是纯头文件库，通常不需要链接
  # 但如果使用了某些需要链接的Eigen模块，可以添加：
  # ${EIGEN3_LIBRARIES}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  detected
  predictor
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

ament_export_include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

ament_export_libraries(
  ${PROJECT_NAME}
  yaml-cpp
  fmt
  cboard
  detected
  predictor
)

ament_export_dependencies(
  rclcpp
  Eigen3
  yaml-cpp
  fmt
  utils  # 重要：导出utils依赖
  detected
  predictor
  cboard
)

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib    # 静态库位置（虽然我们用SHARED）
  LIBRARY DESTINATION lib    # 动态库位置（我们的库会安装在这里）
  RUNTIME DESTINATION bin    # Windows DLL位置
  INCLUDES DESTINATION include
)

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)


ament_package()