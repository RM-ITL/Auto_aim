cmake_minimum_required(VERSION 3.8)
project(filter_lib)

# 设置C++标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# 创建静态库（因为现在有源文件了）
add_library(${PROJECT_NAME} STATIC
  src/track_queue_v3.cpp
  src/track_queue_v4.cpp
  src/rotating_target_tracker.cpp
  src/decoupled_tracker.cpp
)

# 设置目标属性
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

# 设置包含目录
target_include_directories(${PROJECT_NAME} 
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 链接依赖库
target_link_libraries(${PROJECT_NAME} 
  PUBLIC
    Ceres::ceres
    Eigen3::Eigen
)

# 导出依赖信息
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  Eigen3
  Ceres
)

# 安装头文件
install(
  DIRECTORY include/
  DESTINATION include
)

# 安装库文件
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)


ament_package()