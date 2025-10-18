#include "node/armor_detector_node.hpp"
#include <filesystem>
#include <yaml-cpp/yaml.h>

namespace armor_auto_aim {

ArmorDetectorNode::NodeConfig ArmorDetectorNode::NodeConfig::from_yaml(const YAML::Node& node) {
    NodeConfig config;
    
    if (node["yolo"] && node["yolo"]["topics"]) {
        const auto& topics = node["yolo"]["topics"];
        config.image_sub_topic = topics["image_sub"].as<std::string>("image_topic");
        config.armor_pub_topic = topics["armor_pub"].as<std::string>("detected_data");
        config.annotated_image_pub_topic = topics["annotated_image_pub"].as<std::string>("yolo_detections_image");
    }
    
    if (node["yolo"] && node["yolo"]["model"]) {
        const auto& model = node["yolo"]["model"];
        config.max_pending_tasks = model["max_pending_inferences"].as<int>(8);
    }
    
    if (node["yolo"] && node["yolo"]["data_inspection"]) {
        const auto& inspection = node["yolo"]["data_inspection"];
        config.publish_annotated_image = inspection["publish_annotated_image"].as<bool>(true);
    }
    
    if (node["yolo"] && node["yolo"]["visualization"] && node["yolo"]["visualization"]["center_point"]) {
        const auto& center = node["yolo"]["visualization"]["center_point"];
        config.visualization.center_point.x = center["x"].as<int>(640);
        config.visualization.center_point.y = center["y"].as<int>(512);
    }
    
    return config;
}

ArmorDetectorNode::ArmorDetectorNode() 
    : Node("armor_detector_node") {
    
    RCLCPP_INFO(this->get_logger(), "初始化YOLO11装甲板检测节点...");
    
    try {
        config_file_path_ = "/home/guo/ITL_sentry_auto_new/src/config/robomaster_vision_config.yaml";
        
        if (!std::filesystem::exists(config_file_path_)) {
            throw std::runtime_error("配置文件不存在: " + config_file_path_);
        }
        
        YAML::Node yaml_config = YAML::LoadFile(config_file_path_);
        config_ = NodeConfig::from_yaml(yaml_config);
        
        RCLCPP_INFO(this->get_logger(), "配置文件加载成功");

        utils::PerformanceMonitor::Config fps_config;
        fps_config.enable_logging = false;
        fps_monitor_ = std::make_unique<utils::PerformanceMonitor>(fps_config);
        fps_monitor_->register_metric("frame");

        detector_ = std::make_unique<YOLO11Detector>(config_file_path_, false);
        RCLCPP_INFO(this->get_logger(), "YOLO11检测器初始化成功");
        
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1))
            .best_effort()
            .durability_volatile();
        
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            config_.image_sub_topic, qos,
            std::bind(&ArmorDetectorNode::image_callback, this, std::placeholders::_1));
        
        armor_pub_ = this->create_publisher<autoaim_msgs::msg::Armor>(
            config_.armor_pub_topic, 10);
        
        if (config_.publish_annotated_image) {
            annotated_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                config_.annotated_image_pub_topic, qos);
        }
        
        for (int i = 0; i < config_.processing_thread_count; ++i) {
            processing_threads_.emplace_back(&ArmorDetectorNode::processing_thread_func, this);
        }
        
        publishing_thread_ = std::thread(&ArmorDetectorNode::result_publishing_thread_func, this);
        
        RCLCPP_INFO(this->get_logger(), "节点初始化完成");
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "初始化失败: %s", e.what());
        throw;
    }
}

ArmorDetectorNode::~ArmorDetectorNode() {
    stop_processing_.store(true);
    tasks_cv_.notify_all();
    publishing_cv_.notify_all();
    
    for (auto& thread : processing_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    if (publishing_thread_.joinable()) {
        publishing_thread_.join();
    }
}

void ArmorDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        if (pending_tasks_.size() >= static_cast<size_t>(config_.max_pending_tasks)) {
            return;
        }
    }
    
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        
        auto task = std::make_unique<ProcessingTask>();
        task->original_image = cv_ptr->image.clone();
        task->original_msg = msg;
        
        task->armor_future = std::async(std::launch::async, 
            [this, image = cv_ptr->image]() {
                return detector_->detect(image);
            });
        
        {
            std::lock_guard<std::mutex> lock(tasks_mutex_);
            pending_tasks_.push(std::move(task));
        }
        tasks_cv_.notify_one();
        
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "图像转换失败: %s", e.what());
    }
}

void ArmorDetectorNode::processing_thread_func() {
    while (!stop_processing_.load()) {
        std::unique_ptr<ProcessingTask> task;
        
        {
            std::unique_lock<std::mutex> lock(tasks_mutex_);
            tasks_cv_.wait(lock, [this] {
                return !pending_tasks_.empty() || stop_processing_.load();
            });
            
            if (stop_processing_.load()) break;
            
            if (!pending_tasks_.empty()) {
                auto& front_task = pending_tasks_.front();
                if (front_task->armor_future.wait_for(std::chrono::milliseconds(0)) 
                    == std::future_status::ready) {
                    task = std::move(pending_tasks_.front());
                    pending_tasks_.pop();
                } else {
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
            }
        }
        
        if (task) {
            process_detection_task(std::move(task));
        }
    }
}

void ArmorDetectorNode::process_detection_task(std::unique_ptr<ProcessingTask> task) {
    try {
        auto armors = task->armor_future.get();
        
        auto it = armors.begin();
        while (it != armors.end()) {
            if (it->points.size() != 4) {
                it = armors.erase(it);
                continue;
            }
            
            if (!it->is_valid) {
                it = armors.erase(it);
                continue;
            }
            
            ++it;
        }
        
        cv::Mat annotated_image;
        if (config_.publish_annotated_image && !task->original_image.empty()) {
            annotated_image = visualize_results(task->original_image, armors);
        }
        
        {
            std::lock_guard<std::mutex> lock(publishing_mutex_);
            publishing_queue_.push({annotated_image, armors, task->original_msg});
        }
        publishing_cv_.notify_one();
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "处理任务时出错: %s", e.what());
    }
}

cv::Mat ArmorDetectorNode::visualize_results(
    const cv::Mat& image,
    const std::vector<Armor>& armors) {
    
    cv::Mat display = image.clone();
    
    // 帧计数器
    static int frame_count = 0;
    frame_count++;
    
    // 绘制所有检测到的装甲板
    for (const auto& armor : armors) {
        if (armor.points.size() == 4) {
            // 使用专业的绿色四边形绘制
            utils::draw_quadrangle_with_corners(display, armor.points, 
                utils::colors::GREEN, 2, 3);
            
            // 计算标签位置（四边形顶部中心）
            cv::Point2f label_pos;
            label_pos.x = (armor.points[0].x + armor.points[1].x) / 2;
            label_pos.y = std::min(armor.points[0].y, armor.points[1].y);
            
            // 构建专业格式的标签
            std::stringstream label;
            label << std::fixed << std::setprecision(2);
            label << armor.detection_confidence << " ";
            
            // 颜色字符串
            switch (armor.color) {
                case blue: label << "blue"; break;
                case red: label << "red"; break;
                default: label << "unknown"; break;
            }
            
            label << ", " << armor.getNameString();
            label << " ," << (armor.type == small ? "small" : "big");
            
            // 使用专业的标签绘制
            utils::draw_detection_label(display, label.str(), label_pos, 
                utils::colors::GREEN, 0.7, 2);
        }
    }
    
    // 绘制帧号
    utils::draw_frame_number(display, frame_count);
    
    // 绘制FPS信息
    auto metrics = fps_monitor_->get_metrics("frame");
    utils::draw_fps(display, metrics.fps, armors.size());
    
    return display;
}

void ArmorDetectorNode::result_publishing_thread_func() {
    while (!stop_processing_.load()) {
        PublishingTask task;
        bool has_task = false;
        
        {
            std::unique_lock<std::mutex> lock(publishing_mutex_);
            publishing_cv_.wait(lock, [this] {
                return !publishing_queue_.empty() || stop_processing_.load();
            });
            
            if (stop_processing_.load()) break;
            
            if (!publishing_queue_.empty()) {
                task = std::move(publishing_queue_.front());
                publishing_queue_.pop();
                has_task = true;
            }
        }
        
        if (has_task) {
            publish_results(task.annotated_image, task.armors, task.original_msg);
        }
    }
}

void ArmorDetectorNode::publish_results(
    const cv::Mat& annotated_image,
    const std::vector<Armor>& armors,
    const sensor_msgs::msg::Image::SharedPtr& original_msg) {

    // 更新性能监控
    fps_monitor_->record("frame", 1, true);
    
    // 遍历所有装甲板，逐个发布
    for (const auto& armor : armors) {
        // 只发布有效的装甲板
        if (!armor.is_valid) {
            continue;
        }
        
        // 创建单个armor消息
        auto armor_msg = std::make_unique<autoaim_msgs::msg::Armor>();
        armor_msg->header = original_msg->header;
        
        // 设置装甲板颜色
        switch (armor.color) {
            case red:       armor_msg->color = "red"; break;
            case blue:      armor_msg->color = "blue"; break;
            case purple:    armor_msg->color = "purple"; break;
            case extinguish:armor_msg->color = "extinguish"; break;
            default:        armor_msg->color = "unknown"; break;
        }
        
        // 设置装甲板类型（大/小）
        armor_msg->type = (armor.type == big) ? "big" : "small";
        armor_msg->number = armor.getClassNumber();
        armor_msg->confidence = armor.detection_confidence;
        armor_msg->conf_color = armor.detection_confidence;
        armor_msg->conf_number = armor.classify_confidence > 0 ? 
            armor.classify_confidence : armor.detection_confidence;
        
        // ==================== 新增部分开始 ====================
        // 添加优先级信息（rank值）
        // rank 值越小，优先级越高（1最高，5最低）
        armor_msg->rank = armor.rank;
        
        // 添加装甲板中心点坐标
        // 如果已经计算过中心点，直接使用
        if (armor.center.x > 0 && armor.center.y > 0) {
            armor_msg->center.x = armor.center.x;
            armor_msg->center.y = armor.center.y;
            armor_msg->center.z = 0.0;  // 2D图像中z坐标为0
        } else if (armor.points.size() == 4) {
            // 如果中心点未计算，从四个角点计算中心
            float center_x = 0.0, center_y = 0.0;
            for (const auto& point : armor.points) {
                center_x += point.x;
                center_y += point.y;
            }
            armor_msg->center.x = center_x / 4.0;
            armor_msg->center.y = center_y / 4.0;
            armor_msg->center.z = 0.0;
        }
        // ==================== 新增部分结束 ====================
        
        // 设置四个角点坐标（保持原有逻辑）
        if (armor.points.size() == 4) {
            for (size_t i = 0; i < 4; ++i) {
                armor_msg->light_corners[i].x = armor.points[i].x;
                armor_msg->light_corners[i].y = armor.points[i].y;
                armor_msg->light_corners[i].z = 0.0;
                armor_msg->corners[i] = armor_msg->light_corners[i];
            }
        }
        
        // 直接发布单个armor消息
        armor_pub_->publish(*armor_msg);
    }
    
    // 如果需要，发布标注后的图像（这部分保持不变）
    if (annotated_image_pub_ && !annotated_image.empty()) {
        try {
            auto annotated_msg = cv_bridge::CvImage(
                original_msg->header, "bgr8", annotated_image).toImageMsg();
            annotated_image_pub_->publish(*annotated_msg);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "标注图像转换失败: %s", e.what());
        }
    }
}

} // namespace armor_auto_aim

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<armor_auto_aim::ArmorDetectorNode>();  
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "节点运行错误: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}