#include "node/armor_detector_node.hpp"
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace armor_auto_aim {

ArmorDetectorNode::NodeConfig ArmorDetectorNode::NodeConfig::from_yaml(const YAML::Node& node) {
    NodeConfig config;

        // 添加配置文件路径的读取
    if (node["config_file_path"]) {
        config.config_file_path = node["config_file_path"].as<std::string>();
    }
    
    if (node["yolo"] && node["yolo"]["topics"]) {
        const auto& topics = node["yolo"]["topics"];
        config.image_sub_topic = topics["image_sub"].as<std::string>("/image_raw");
        config.armorresult_pub_topic = topics["armorresult_pub"].as<std::string>("/armor_detector/result");
        config.annotated_image_pub_topic = topics["annotated_image_pub"].as<std::string>("/armor_detector/annotated_image");
        config.lightbar_roi_pub_topic = topics["lightbar_roi_pub"].as<std::string>("/armor_detector/lightbar_roi");
    }
    
    if (node["yolo"] && node["yolo"]["model"]) {
        const auto& model = node["yolo"]["model"];
        config.max_pending_tasks = model["max_pending_inferences"].as<int>(8);
    }
    
    if (node["yolo"] && node["yolo"]["data_inspection"]) {
        const auto& inspection = node["yolo"]["data_inspection"];
        config.publish_lightbar_roi = inspection["publish_lightbar_roi"].as<bool>(true);
        config.publish_annotated_image = inspection["publish_annotated_image"].as<bool>(true);
    }
    
    if (node["yolo"] && node["yolo"]["armor"]) {
        const auto& armor = node["yolo"]["armor"];
        config.armor_width_factor = armor["width_factor"].as<float>(2.0f);
        config.armor_height_factor = armor["height_factor"].as<float>(2.0f);
    }
    
    if (node["pointer"]) {
        const auto& pointer = node["pointer"];
        config.enable_pointer_refinement = pointer["enable"].as<bool>(true);
        config.roi_expansion_ratio = pointer["roi_expansion_ratio"].as<double>(1.5);
        
        std::string enemy_color_str = pointer["enemy_color"].as<std::string>("blue");
        if (enemy_color_str == "red") {
            config.enemy_color = armor_pointer::ARMOR_COLOR_RED;
        } else if (enemy_color_str == "blue") {
            config.enemy_color = armor_pointer::ARMOR_COLOR_BLUE;
        }
    }
    
    // 简化可视化配置的读取部分
    if (node["yolo"] && node["yolo"]["visualization"]) {
        if (node["yolo"]["visualization"]["center_point"]) {
            const auto& center = node["yolo"]["visualization"]["center_point"];
            config.visualization.center_point.x = center["x"].as<int>(640);
            config.visualization.center_point.y = center["y"].as<int>(512);
        }
        
        if (node["yolo"]["visualization"]["corners"]) {
            const auto& corners = node["yolo"]["visualization"]["corners"];
            config.visualization.show_lightbar_corners = corners["show_lightbar"].as<bool>(true);
        }
    }

        // 读取装甲板尺寸分类配置
    if (node["armor_size_classification"]) {
        const auto& size = node["armor_size_classification"];
        config.large_armor_ratio = size["large_armor_ratio"].as<float>(1.84f);
        config.small_armor_ratio = size["small_armor_ratio"].as<float>(1.08f);
        config.ratio_threshold = size["ratio_threshold"].as<float>(1.46f);
    }
    
    return config;
}

ArmorDetectorNode::ArmorDetectorNode() 
    : Node("armor_detector_node") {
    
    RCLCPP_INFO(this->get_logger(), "初始化装甲板检测节点...");
    
    try {
        // 加载配置文件
        yaml_config_ = utils::load_config();
        config_ = NodeConfig::from_yaml(yaml_config_);

                // 初始化FPS监控器（静默模式，仅用于计算）
        utils::PerformanceMonitor::Config fps_config;
        fps_config.enable_logging = false;  // 静默模式
        fps_monitor_ = std::make_unique<utils::PerformanceMonitor>(fps_config);
        fps_monitor_->register_metric("frame");

        // 修改点1：使用配置文件路径初始化检测器和分类器
        // 这两个类的构造函数只接受文件路径，不接受Config对象
        detector_ = std::make_unique<ArmorDetector>(config_.config_file_path);
        classifier_ = std::make_unique<ArmorClassifier>(config_.config_file_path);

        // 初始化pointer模块
        if (config_.enable_pointer_refinement) {
            armor_pointer::LightbarDetector::Config pointer_config;
            std::string config_path = "/home/guo/ITL_sentry_auto/src/config/robomaster_vision_config.yaml";
            pointer_config.load_from_yaml(config_path);
            lightbar_detector_ = std::make_unique<armor_pointer::LightbarDetector>(pointer_config);
            RCLCPP_INFO(this->get_logger(), "Pointer精确定位模块已启用");
        }
        
        // 创建ROS2接口
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            config_.image_sub_topic, qos,
            std::bind(&ArmorDetectorNode::image_callback, this, std::placeholders::_1));
        
        armorresult_pub_ = this->create_publisher<autoaim_msgs::msg::Armorresult>(
            config_.armorresult_pub_topic, 10);
        
        if (config_.publish_annotated_image) {
            annotated_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                config_.annotated_image_pub_topic, qos);
            RCLCPP_INFO(this->get_logger(), "标注图像发布话题: %s", config_.annotated_image_pub_topic.c_str());
        }
        
        if (config_.publish_lightbar_roi) {
            lightbar_roi_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
                config_.lightbar_roi_pub_topic, qos);
            RCLCPP_INFO(this->get_logger(), "灯条ROI发布话题: %s", config_.lightbar_roi_pub_topic.c_str());
        }
        
        // 启动处理线程
        for (int i = 0; i < config_.processing_thread_count; ++i) {
            processing_threads_.emplace_back(&ArmorDetectorNode::processing_thread_func, this);
        }
        publishing_thread_ = std::thread(&ArmorDetectorNode::result_publishing_thread_func, this);
        
        RCLCPP_INFO(this->get_logger(), "装甲板检测节点初始化完成");
        
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

// image_callback函数 - 只需要改变future的类型
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
        task->original_image = cv_ptr->image;
        task->original_msg = msg;
        task->armor_future = detector_->detect_async(cv_ptr->image);  // 改为armor_future
        
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

// 简化后的处理函数
void ArmorDetectorNode::process_detection_task(std::unique_ptr<ProcessingTask> task) {
    try {
        // 获取检测结果 - 现在直接是Armor对象
        auto armors = task->armor_future.get();
        
        // 分类和优化 - 直接在Armor对象上操作
        if (!armors.empty()) {
            classify_and_refine_armors(task->original_image, armors);
        }
        
        // 使用pointer进行精确定位（如果启用）
        std::vector<cv::Mat> lightbar_rois;
        if (config_.enable_pointer_refinement && lightbar_detector_ && !armors.empty()) {
            auto [refined_armors, rois] = refine_armors_with_pointer(
                task->original_image, armors);
            armors = refined_armors;
            lightbar_rois = rois;
        }
        
        // 可视化
        cv::Mat annotated_image;
        if (config_.publish_annotated_image) {
            annotated_image = visualize_results(task->original_image, armors);
        }
        
        // 发布结果
        {
            std::lock_guard<std::mutex> lock(publishing_mutex_);
            publishing_queue_.push({annotated_image, armors, task->original_msg, lightbar_rois});
        }
        publishing_cv_.notify_one();
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "处理任务时出错: %s", e.what());
    }
}

// 新的分类和优化函数 - 直接操作Armor对象
void ArmorDetectorNode::classify_and_refine_armors(
    const cv::Mat& image, 
    std::vector<Armor>& armors) {
    
    // 批量分类 - 分类器会直接修改Armor对象
    classifier_->classify_batch(armors);
    
    // 根据宽高比调整分类结果（特殊逻辑保留）
    for (auto& armor : armors) {
        float aspect_ratio = static_cast<float>(armor.box.width) / static_cast<float>(armor.box.height);
        
        // 大装甲板特殊处理
        if (aspect_ratio > config_.ratio_threshold) {
            // 如果宽高比表明是大装甲板，强制分类为1号
            if (armor.name != one) {
                armor.name = one;
                armor.classify_confidence *= 0.8f;  // 稍微降低置信度表示这是推断的
                
                RCLCPP_DEBUG(this->get_logger(), 
                    "大装甲板检测 - 宽高比:%.2f, 强制分类为1号", aspect_ratio);
            }
        } else {
            // 小装甲板不应该是1号
            if (armor.name == one) {
                armor.is_valid = false;  // 标记为无效
                armor.classify_confidence *= 0.5f;  // 大幅降低置信度
                
                RCLCPP_DEBUG(this->get_logger(), 
                    "尺寸不匹配 - 宽高比:%.2f表明是小装甲板，但分类为1号", aspect_ratio);
            }
        }
    }
}

// Pointer精确定位 - 适配Armor结构
std::pair<std::vector<Armor>, std::vector<cv::Mat>>
ArmorDetectorNode::refine_armors_with_pointer(
    const cv::Mat& image,
    std::vector<Armor>& armors) {
    
    std::vector<cv::Mat> roi_images;
    roi_images.reserve(armors.size());
    
    for (auto& armor : armors) {
        try {
            cv::Rect roi_rect = expand_roi_for_lightbar(
                armor.box, image.size(), config_.roi_expansion_ratio);
            
            // 使用pointer进行精确定位
            auto pointer_result = lightbar_detector_->process(
                image, roi_rect, config_.enemy_color);
            
            // 如果成功，更新Armor的points字段
            if (pointer_result.success && pointer_result.four_points.size() == 4) {
                armor.points = pointer_result.four_points;  // 直接使用Armor的points字段
                // 可以添加一个标记表示已经精确定位（但不是必须的）
            }
            
            // 保存用于显示的二值化图像
            cv::Mat display_image;
            if (!pointer_result.binary_image.empty()) {
                cv::cvtColor(pointer_result.binary_image, display_image, cv::COLOR_GRAY2BGR);
            } else {
                display_image = cv::Mat::zeros(roi_rect.height, roi_rect.width, CV_8UC3);
            }
            roi_images.push_back(display_image);
            
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Pointer处理异常: %s", e.what());
            roi_images.push_back(cv::Mat::zeros(100, 200, CV_8UC3));
        }
    }
    
    return {armors, roi_images};
}

cv::Rect ArmorDetectorNode::expand_roi_for_lightbar(
    const cv::Rect& original_roi, 
    const cv::Size& image_size, 
    double expansion_ratio) {
    
    int expanded_width = static_cast<int>(original_roi.width * expansion_ratio);
    int expanded_height = static_cast<int>(original_roi.height * expansion_ratio);
    
    int x = original_roi.x - (expanded_width - original_roi.width) / 2;
    int y = original_roi.y - (expanded_height - original_roi.height) / 2;
    
    x = std::max(0, x);
    y = std::max(0, y);
    expanded_width = std::min(expanded_width, image_size.width - x);
    expanded_height = std::min(expanded_height, image_size.height - y);
    
    return cv::Rect(x, y, expanded_width, expanded_height);
}


// 可视化函数 - 简化版本
cv::Mat ArmorDetectorNode::visualize_results(
    const cv::Mat& image,
    const std::vector<Armor>& armors) {
    
    cv::Mat display = image.clone();
    
    for (const auto& armor : armors) {
        // 获取颜色（从color枚举）
        cv::Scalar draw_color;
        if (armor.color == blue) {
            draw_color = cv::Scalar(255, 0, 0);
        } else if (armor.color == red) {
            draw_color = cv::Scalar(0, 0, 255);
        } else {
            draw_color = cv::Scalar(0, 255, 0);
        }
        
        // 构建标签
        std::string label = armor.getNameString() + 
            " (" + std::to_string(static_cast<int>(armor.getCombinedConfidence() * 100)) + "%)";
        
        // 绘制装甲板
        if (!armor.points.empty() && config_.visualization.show_lightbar_corners) {
            utils::draw_armor_with_corners(display, armor.box, label, 
                                          armor.points, draw_color);
        } else {
            utils::draw_armor(display, armor.box, label, draw_color);
        }
    }
    
    // 绘制中心参考点和FPS
    utils::draw_crosshair(display, config_.visualization.center_point);
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
            publish_results(task.annotated_image, task.armors, 
                          task.original_msg, task.lightbar_rois);
        }
    }
}

void ArmorDetectorNode::publish_results(
    const cv::Mat& annotated_image,
    const std::vector<Armor>& armors,
    const sensor_msgs::msg::Image::SharedPtr& original_msg,
    const std::vector<cv::Mat>& lightbar_rois) {

    fps_monitor_->record("frame", 1, true);
    
    // 发布装甲板检测结果
    if (!armors.empty()) {
        // 按装甲板编号分组
        std::map<int, std::vector<Armor>> grouped_armors;
        for (const auto& armor : armors) {
            if (armor.is_valid) {
                int armor_number = static_cast<int>(armor.name);
                grouped_armors[armor_number].push_back(armor);
            }
        }
        
        // 优先级顺序
        std::vector<int> priority_order = {3, 4, 1, -1, 2, 0};
        
        for (int armor_id : priority_order) {
            auto it = grouped_armors.find(armor_id);
            if (it == grouped_armors.end()) continue;
            
            auto armorresult_msg = std::make_unique<autoaim_msgs::msg::Armorresult>();
            armorresult_msg->header = original_msg->header;
            armorresult_msg->target_id = armor_id;
            armorresult_msg->armor_count = static_cast<int32_t>(it->second.size());
            
            for (const auto& armor : it->second) {
                autoaim_msgs::msg::Armor armor_msg;
                armor_msg.header = original_msg->header;
                
                // 使用Armor中已有的数据
                armor_msg.color = armor.color == red ? "red" : "blue";
                armor_msg.conf_color = armor.detection_confidence;
                armor_msg.number = static_cast<int>(armor.name);
                armor_msg.conf_number = armor.classify_confidence;
                
                // 填充角点 - 优先使用精确定位的points，否则用box计算
                if (armor.points.size() == 4) {
                    for (size_t i = 0; i < 4; ++i) {
                        armor_msg.light_corners[i].x = armor.points[i].x;
                        armor_msg.light_corners[i].y = armor.points[i].y;
                        armor_msg.light_corners[i].z = 0.0;
                    }
                } else {
                    // 使用box计算默认角点
                    float center_x = armor.center.x;
                    float center_y = armor.center.y;
                    float half_width = armor.box.width / config_.armor_width_factor;
                    float half_height = armor.box.height / config_.armor_height_factor;
                    
                    armor_msg.corners[0].x = center_x - half_width;
                    armor_msg.corners[0].y = center_y - half_height;
                    armor_msg.corners[1].x = center_x + half_width;
                    armor_msg.corners[1].y = center_y - half_height;
                    armor_msg.corners[2].x = center_x + half_width;
                    armor_msg.corners[2].y = center_y + half_height;
                    armor_msg.corners[3].x = center_x - half_width;
                    armor_msg.corners[3].y = center_y + half_height;
                    
                    for (size_t i = 0; i < 4; ++i) {
                        armor_msg.light_corners[i] = armor_msg.corners[i];
                    }
                }
                
                armorresult_msg->armor.push_back(armor_msg);
            }
            
            armorresult_pub_->publish(*armorresult_msg);
        }
    }
    
    // 发布标注图像和ROI（保持原有逻辑）
    if (annotated_image_pub_ && !annotated_image.empty()) {
        try {
            auto annotated_msg = cv_bridge::CvImage(
                original_msg->header, "bgr8", annotated_image).toImageMsg();
            annotated_image_pub_->publish(*annotated_msg);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "标注图像转换失败: %s", e.what());
        }
    }
    
    if (lightbar_roi_pub_ && !lightbar_rois.empty()) {
        try {
            cv::Mat roi_display = utils::create_grid(lightbar_rois, 4, 2);
            auto roi_msg = cv_bridge::CvImage(
                original_msg->header, "bgr8", roi_display).toImageMsg();
            lightbar_roi_pub_->publish(*roi_msg);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "灯条ROI图像转换失败: %s", e.what());
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
    }
    
    rclcpp::shutdown();
    return 0;
}