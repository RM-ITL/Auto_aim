#ifndef ARMOR_DETECTOR_NODE_HPP
#define ARMOR_DETECTOR_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <autoaim_msgs/msg/armor.hpp>
#include <autoaim_msgs/msg/armorresult.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include <queue>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>
#include <memory>
#include <chrono>
#include <condition_variable>

#include "detector/armor_detector.hpp"
#include "classifier/armor_classifier.hpp"
#include "common/armor.hpp"  // 使用统一的Armor定义
#include "detector_tools.hpp"
#include "performance_monitor.hpp"
#include "lightbar_detector.hpp"
#include "draw_tools.hpp"
#include "types.hpp"

namespace armor_auto_aim {

class ArmorDetectorNode : public rclcpp::Node {
public:
    ArmorDetectorNode();
    ~ArmorDetectorNode();

private:
    struct NodeConfig {
        // ROS话题配置
        std::string image_sub_topic = "/image_raw";
        std::string armorresult_pub_topic = "detected_data";
        std::string annotated_image_pub_topic = "yolo_detections_image";
        std::string lightbar_roi_pub_topic = "/armor_detector/lightbar_roi";

        float large_armor_ratio = 1.84f;      
        float small_armor_ratio = 1.08f;      
        float ratio_threshold = 10.0f;        
        
        // 性能相关配置
        int max_pending_tasks = 8;
        int processing_thread_count = 1;
        
        // 发布控制
        bool publish_annotated_image = true;
        bool publish_lightbar_roi = true;
        
        // 装甲板尺寸因子
        float armor_width_factor = 2.0f;
        float armor_height_factor = 2.0f;
        
        // Pointer精确定位配置
        bool enable_pointer_refinement = true;
        armor_pointer::ArmorColor enemy_color = armor_pointer::ARMOR_COLOR_BLUE;
        double roi_expansion_ratio = 1.5;
        
        // 可视化配置
        struct VisualizationConfig {
            cv::Point center_point{640, 384};      
            bool show_lightbar_corners = true;     
        } visualization;
        
        // 配置文件路径
        std::string config_file_path = "/home/guo/ITL_sentry_auto/src/config/robomaster_vision_config.yaml";
        
        static NodeConfig from_yaml(const YAML::Node& node);
    };
    
    // 处理任务结构体 - 保留original_image和original_msg，只替换Detection为Armor
    struct ProcessingTask {
        cv::Mat original_image;                           // 保留：用于可视化和ROI提取
        sensor_msgs::msg::Image::SharedPtr original_msg;  // 保留：用于时间戳和header信息
        std::future<std::vector<Armor>> armor_future;     // 修改：直接使用Armor
        
        ProcessingTask() {}
    };
    
    // 发布任务结构体 - 直接使用Armor，去掉EnhancedDetection
    struct PublishingTask {
        cv::Mat annotated_image;
        std::vector<Armor> armors;  // 直接使用Armor替代EnhancedDetection
        sensor_msgs::msg::Image::SharedPtr original_msg;
        std::vector<cv::Mat> lightbar_rois;
    };
    
    // 回调和处理函数
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void processing_thread_func();
    void result_publishing_thread_func();
    void process_detection_task(std::unique_ptr<ProcessingTask> task);
    
    // 简化后的处理函数 - 直接操作Armor对象
    void classify_and_refine_armors(
        const cv::Mat& image, 
        std::vector<Armor>& armors);
    
    std::pair<std::vector<Armor>, std::vector<cv::Mat>> 
    refine_armors_with_pointer(
        const cv::Mat& image,
        std::vector<Armor>& armors);
    
    // 可视化和发布
    cv::Mat visualize_results(
        const cv::Mat& image,
        const std::vector<Armor>& armors);
    
    void publish_results(
        const cv::Mat& annotated_image,
        const std::vector<Armor>& armors,
        const sensor_msgs::msg::Image::SharedPtr& original_msg,
        const std::vector<cv::Mat>& lightbar_rois);
    
    cv::Rect expand_roi_for_lightbar(const cv::Rect& original_roi, 
        const cv::Size& image_size, double expansion_ratio);
    
    // ROS2接口
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<autoaim_msgs::msg::Armorresult>::SharedPtr armorresult_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr lightbar_roi_pub_;
    
    // 核心检测模块
    std::unique_ptr<ArmorDetector> detector_;
    std::unique_ptr<ArmorClassifier> classifier_;
    std::unique_ptr<armor_pointer::LightbarDetector> lightbar_detector_;
    std::unique_ptr<utils::PerformanceMonitor> fps_monitor_;
    
    // 配置
    NodeConfig config_;
    YAML::Node yaml_config_;
    
    // 多线程处理队列
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