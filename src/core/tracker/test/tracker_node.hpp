#ifndef TRACKER_NODE_HPP
#define TRACKER_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <autoaim_msgs/msg/armor.hpp>
#include <autoaim_msgs/msg/target.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <list>
#include <mutex>

#include "tracker.hpp"
#include "solver_node.hpp"
#include "draw_tools.hpp"
#include "module/coord_converter.hpp"
#include "logger.hpp"

namespace tracker {

class TrackerNode : public rclcpp::Node {
public:
    explicit TrackerNode();
    ~TrackerNode() = default;

private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void armorCallback(const autoaim_msgs::msg::Armor::SharedPtr msg);
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void processArmors();
    
    // 订阅者
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<autoaim_msgs::msg::Armor>::SharedPtr armor_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    
    // 发布者
    rclcpp::Publisher<autoaim_msgs::msg::Target>::SharedPtr target_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr tracking_image_pub_;
    
    // 核心组件
    std::unique_ptr<solver::Solver> solver_;
    solver::CoordConverter* coord_converter_;  
    solver::YawOptimizer* yaw_optimizer_;
    std::unique_ptr<tracker::Tracker> tracker_;
    
    // 数据缓冲
    std::list<autoaim_msgs::msg::Armor> armor_buffer_;
    std::mutex armor_mutex_;
    
    // 图像数据（保存最新图像供你自己使用）
    cv::Mat latest_image_;
    std::mutex image_mutex_;
    std_msgs::msg::Header latest_image_header_;
    
    std::string config_path_;
};

}

#endif