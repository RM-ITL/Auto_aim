cmake_minimum_required(VERSION 3.5)
project(motion_model)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -fPIC)
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(autoaim_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
# 假设filter_lib是另一个内部包

# 包含目录
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# 创建静态库
add_library(${PROJECT_NAME} SHARED
  src/full_model.cpp
)

# 设置库的包含目录（用于其他包使用此库时）
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 链接依赖
ament_target_dependencies(${PROJECT_NAME}
  OpenCV
  rclcpp
  autoaim_msgs
)

# 链接yaml-cpp（非ament包，需要手动链接）
target_link_libraries(${PROJECT_NAME}
  ${YAML_CPP_LIBRARIES}
)

# 导出库的依赖信息
ament_export_targets(${PROJECT_NAME}_targets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  OpenCV
  rclcpp
  autoaim_msgs
  Eigen3
  yaml-cpp
)

# 安装库文件
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}_targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# 安装头文件
install(
  DIRECTORY include/
  DESTINATION include
)

# 测试配置（可选）
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  # 添加单元测试
  find_package(ament_cmake_gtest REQUIRED)
  # ament_add_gtest(test_common_model test/test_common_model.cpp)
  # if(TARGET test_common_model)
  #   target_link_libraries(test_common_model ${PROJECT_NAME})
  #   ament_target_dependencies(test_common_model rclcpp)
  # endif()
endif()

# 导出包信息
ament_package()
