#include "rclcpp/rclcpp.hpp"
#include "imu_driver.h"
#include <iostream>
#include <thread>
#include <condition_variable>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    auto imu_node = std::make_shared<dmbot_serial::DmImu>();
    
    // 使用rclcpp::spin来处理回调
    rclcpp::spin(imu_node);
    
    // 清理资源
    rclcpp::shutdown();
    
    return 0;
}
