cmake_minimum_required(VERSION 3.8)
project(serial_sender)  # 您可以根据需要修改项目名称

# 设置 C++ 标准
# 使用 C++17 是因为它提供了更好的性能和更现代的特性
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 添加编译选项
# 这些选项帮助我们在开发阶段发现潜在问题
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖项
# 这是 ROS2 构建系统的核心部分
find_package(ament_cmake REQUIRED)           # ROS2 的构建系统
find_package(rclcpp REQUIRED)                # ROS2 的 C++ 客户端库
find_package(std_msgs REQUIRED)              # 标准消息类型
find_package(autoaim_msgs REQUIRED)          # 您的自定义消息包

# 包含目录设置
# 告诉编译器在哪里找头文件
include_directories(
  include  # 假设头文件在 include/ 目录下
)

# 创建可执行文件
# 这里定义了我们要编译的程序
add_executable(serial_node
  src/serial_node.cpp
)

# 指定目标的依赖项
# 这告诉构建系统我们的程序需要链接哪些库
ament_target_dependencies(serial_node
  rclcpp
  std_msgs
  autoaim_msgs
)

# 安装可执行文件
# 这样 ros2 run 命令才能找到我们的程序
install(TARGETS
  serial_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装头文件（如果您希望其他包能够使用这些头文件）
install(DIRECTORY include/
  DESTINATION include
)


# 测试相关配置
# 良好的软件工程实践包括编写测试
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # 下面这行会自动运行所有的 linter
  # 包括代码风格检查、静态分析等
  ament_lint_auto_find_test_dependencies()
  
  # 如果您编写了单元测试，可以在这里添加
  # find_package(ament_cmake_gtest REQUIRED)
  # ament_add_gtest(${PROJECT_NAME}_test test/test_serial_sender.cpp)
  # target_link_libraries(${PROJECT_NAME}_test your_library_name)
endif()

# 导出包信息
# 这是 ament 构建系统的要求
ament_package()