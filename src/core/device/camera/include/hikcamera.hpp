#ifndef HIKCAMERA_HPP
#define HIKCAMERA_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <thread>
#include <atomic>
#include <chrono>
#include "MvCameraControl.h"
#include "thread_safe_queue.hpp"
#include "performance_monitor.hpp"

namespace camera
{

struct CameraData {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
};

class HikRobotNode : public rclcpp::Node
{
public:
    HikRobotNode(const std::string& config_path, const std::string& name = "hikrobot_node");
    ~HikRobotNode();

private:
    void daemonThread();
    void captureThread();
    void processThread();
    bool startCapture();
    void stopCapture();
    void setCameraParameters();
    cv::Mat convertBayer(const cv::Mat& raw, unsigned int type);
    void resetUSB() const;
    bool loadConfig(const std::string& config_path);

private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    
    void* camera_handle_ = nullptr;
    unsigned int payload_size_ = 0;
    std::unique_ptr<unsigned char[]> raw_buffer_;
    
    std::thread daemon_thread_, capture_thread_, process_thread_;
    std::atomic<bool> daemon_quit_{false};
    std::atomic<bool> capture_quit_{false};
    std::atomic<bool> capturing_{false};
    std::atomic<bool> node_shutdown_{false};
    
    utils::ThreadSafeQueue<CameraData> queue_;
    utils::PerformanceMonitor perf_monitor_;
    
    double exposure_us_, gain_, fps_;
    cv::Size target_size_;
    std::string image_topic_;
    int vid_ = 0x2bdf;
    int pid_ = 0x0299;
};

}

#endif