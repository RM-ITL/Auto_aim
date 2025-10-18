#ifndef ARMOR_DETECTOR_NODE_HPP
#define ARMOR_DETECTOR_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <autoaim_msgs/msg/armor.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include <queue>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>
#include <memory>
#include <chrono>
#include <condition_variable>

#include "detector/yolo11.hpp"
#include "common/armor.hpp"
#include "detector_tools.hpp"
#include "performance_monitor.hpp"
#include "draw_tools.hpp"

namespace armor_auto_aim {

class ArmorDetectorNode : public rclcpp::Node {
public:
    ArmorDetectorNode();
    ~ArmorDetectorNode();

private:
    // 节点配置结构体，管理所有ROS相关的配置参数
    struct NodeConfig {
        // ROS话题配置
        std::string image_sub_topic = "/image_raw";
        std::string armor_pub_topic = "detected_data";
        std::string annotated_image_pub_topic = "yolo_detections_image";
        
        // 性能相关配置
        int max_pending_tasks = 8;              // 最大待处理任务数
        int processing_thread_count = 1;        // 处理线程数量
        
        // 发布控制
        bool publish_annotated_image = true;    // 是否发布标注图像
        
        // 可视化配置
        struct VisualizationConfig {
            cv::Point center_point{640, 384};   // 图像中心参考点
        } visualization;
        
        // 从YAML节点中提取配置
        static NodeConfig from_yaml(const YAML::Node& node);
    };
    
    // 处理任务结构体，封装一个检测任务的所有数据
    struct ProcessingTask {
        cv::Mat original_image;                          // 原始图像
        sensor_msgs::msg::Image::SharedPtr original_msg; // 原始ROS消息
        std::future<std::vector<Armor>> armor_future;    // 异步检测结果
        
        ProcessingTask() {}
    };
    
    // 发布任务结构体，封装待发布的结果
    struct PublishingTask {
        cv::Mat annotated_image;                         // 标注后的图像
        std::vector<Armor> armors;                       // 检测到的装甲板
        sensor_msgs::msg::Image::SharedPtr original_msg; // 原始消息（用于时间戳）
    };
    
    // 核心回调和处理函数
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void processing_thread_func();
    void result_publishing_thread_func();
    void process_detection_task(std::unique_ptr<ProcessingTask> task);
    
    // 可视化和发布函数
    cv::Mat visualize_results(const cv::Mat& image, const std::vector<Armor>& armors);
    void publish_results(
        const cv::Mat& annotated_image,
        const std::vector<Armor>& armors,
        const sensor_msgs::msg::Image::SharedPtr& original_msg);
    
    // ROS2接口
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<autoaim_msgs::msg::Armor>::SharedPtr armor_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_image_pub_;
    
    // 核心检测模块
    std::unique_ptr<YOLO11Detector> detector_;
    std::unique_ptr<utils::PerformanceMonitor> fps_monitor_;
    
    // 配置管理
    NodeConfig config_;
    std::string config_file_path_;
    
    // 多线程处理相关
    std::queue<std::unique_ptr<ProcessingTask>> pending_tasks_;
    std::mutex tasks_mutex_;
    std::condition_variable tasks_cv_;
    
    std::queue<PublishingTask> publishing_queue_;
    std::mutex publishing_mutex_;
    std::condition_variable publishing_cv_;
    
    std::vector<std::thread> processing_threads_;
    std::thread publishing_thread_;
    std::atomic<bool> stop_processing_{false};
};

} // namespace armor_auto_aim

#endif // ARMOR_DETECTOR_NODE_HPP