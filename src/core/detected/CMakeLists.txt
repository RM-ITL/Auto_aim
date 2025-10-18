cmake_minimum_required(VERSION 3.8)
project(detected)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 优化编译选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# ============================================================================
# 查找依赖项
# ============================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(autoaim_msgs REQUIRED)
find_package(armor_pointer REQUIRED)
find_package(utils REQUIRED)  # 添加utils包依赖

# 设置相机SDK路径（如果需要的话）
set(MV_CAMERA_SDK_PATH "/opt/MVS" CACHE PATH "Path to Hikvision MVS SDK")
set(MV_CAMERA_INCLUDE_DIR ${MV_CAMERA_SDK_PATH}/include)
set(MV_CAMERA_LIB_DIR ${MV_CAMERA_SDK_PATH}/lib/64)

# 添加头文件目录
include_directories(
  include
  include/detected  # 添加项目特定的头文件目录
  ${OpenCV_INCLUDE_DIRS}
  ${yaml-cpp_INCLUDE_DIRS}
  ${OpenVINO_INCLUDE_DIRS}  # 添加OpenVINO头文件路径
)

# ============================================================================
# 定义源文件
# ============================================================================

# 数据结构源文件
set(DATA_STRUCTURE_SOURCES
  src/common/armor.cpp  # 装甲板数据结构实现
)

# 检测器模块源文件
set(DETECTOR_SOURCES
  src/detector/armor_detector.cpp
  src/detector/yolo11.cpp  # 新增YOLO11检测器
)

# 分类器模块源文件
set(CLASSIFIER_SOURCES
  src/classifier/armor_classifier.cpp
)

# 节点源文件
set(NODE_SOURCES
  src/node/armor_detector_node.cpp
)

# ============================================================================
# 创建库
# ============================================================================

# 创建数据结构库（如果armor.cpp存在的话）
add_library(armor_data_lib SHARED
  ${DATA_STRUCTURE_SOURCES}
)

# 设置数据结构库的依赖
target_link_libraries(armor_data_lib
  ${OpenCV_LIBS}
)

# 添加ROS2依赖到数据结构库
ament_target_dependencies(armor_data_lib
  rclcpp
)

# 创建检测器库
add_library(armor_detector_lib SHARED
  ${DETECTOR_SOURCES}
)

# 设置检测器库的包含目录
target_include_directories(armor_detector_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenVINO_INCLUDE_DIRS}
)

# 设置检测器库的依赖
target_link_libraries(armor_detector_lib
  armor_data_lib  # 链接数据结构库
  ${OpenCV_LIBS}
  yaml-cpp
  openvino::runtime  # OpenVINO运行时库
)

# 添加ROS2依赖到检测器库（包括utils）
ament_target_dependencies(armor_detector_lib
  rclcpp
  utils  # 添加utils依赖，用于detector_tools等工具函数
)

# 创建分类器库
add_library(armor_classifier_lib SHARED
  ${CLASSIFIER_SOURCES}
)

# 设置分类器库的依赖
target_link_libraries(armor_classifier_lib
  armor_data_lib  # 链接数据结构库
  ${OpenCV_LIBS}
  yaml-cpp
  openvino::runtime
)

# 添加ROS2依赖到分类器（包括utils）
ament_target_dependencies(armor_classifier_lib
  rclcpp
  utils  # 添加utils依赖，用于performance_monitor等工具
)

# ============================================================================
# 创建可执行文件
# ============================================================================

# 主节点可执行文件
add_executable(armor_detector_node
  ${NODE_SOURCES}
)

# 设置包含目录
target_include_directories(armor_detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 链接库到可执行文件
target_link_libraries(armor_detector_node
  armor_data_lib  # 链接数据结构库
  armor_detector_lib
  armor_classifier_lib
  yaml-cpp
  armor_pointer::armor_pointer
  pthread
)

# 设置ROS2依赖（包括utils）
ament_target_dependencies(armor_detector_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  autoaim_msgs
  image_transport
  armor_pointer
  utils  # 添加utils依赖，确保节点能访问所有工具函数
)

# ============================================================================
# 为了向后兼容，保留原来的节点名称
# ============================================================================
# 创建一个符号链接或别名
add_executable(detector_classifer_node
  ${NODE_SOURCES}
)

target_include_directories(detector_classifer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(detector_classifer_node
  armor_data_lib  # 链接数据结构库
  armor_detector_lib
  armor_classifier_lib
  yaml-cpp
  armor_pointer::armor_pointer
  pthread
)

# 设置兼容节点的ROS2依赖（包括utils）
ament_target_dependencies(detector_classifer_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  autoaim_msgs
  image_transport
  armor_pointer
  utils  # 添加utils依赖
)


# ============================================================================
# 安装
# ============================================================================

# 安装库文件
install(TARGETS
  armor_data_lib  # 安装数据结构库
  armor_detector_lib
  armor_classifier_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 安装可执行文件
install(TARGETS
  armor_detector_node
  detector_classifer_node  # 保留原名称以保持兼容性
  DESTINATION lib/${PROJECT_NAME}
)

# 安装头文件
install(DIRECTORY include/
  DESTINATION include
)

# 安装配置文件（如果有的话）
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  OPTIONAL  # 如果config目录不存在也不会报错
)



# ============================================================================
# 测试
# ============================================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  
  # 可以在这里添加单元测试
  # add_executable(unit_test_yolo11 test/unit_test_yolo11.cpp)
  # target_link_libraries(unit_test_yolo11 
  #   armor_data_lib
  #   armor_detector_lib
  # )
  # ament_target_dependencies(unit_test_yolo11 utils)
endif()

# ============================================================================
# 导出依赖项
# ============================================================================
ament_export_dependencies(
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  autoaim_msgs
  OpenCV
  yaml-cpp
  image_transport
  OpenVINO
  armor_pointer
  utils  # 导出utils依赖，让依赖detected包的其他包也能使用
)

# 导出库，这样其他包可以使用这些库
ament_export_libraries(
  armor_data_lib
  armor_detector_lib
  armor_classifier_lib
)

# 导出包含目录
ament_export_include_directories(include)

ament_package()