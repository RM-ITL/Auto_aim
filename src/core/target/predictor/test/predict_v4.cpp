#include "prediction_node.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace aimer {

// ArmorTracker 实现 - 使用TrackQueueV4和XYZ坐标
ArmorTracker::ArmorTracker(int armor_id,
                           const std::string& armor_type,
                           const Eigen::Vector3d& initial_position_mm,
                           double initial_time)
    : armor_id_(armor_id),
      armor_type_(armor_type),
      filter_initialized_(false),
      last_update_time_(initial_time),
      update_count_(0) {
    
    // 将初始观测添加到缓冲区
    Eigen::Vector4d init_pose;
    init_pose << initial_position_mm.x(), initial_position_mm.y(), initial_position_mm.z(), initial_time;
    init_buffer_.push_back(init_pose);
    
    RCLCPP_INFO(rclcpp::get_logger("ArmorTracker"),
                "初始化装甲板跟踪器 - ID: %d, 类型: %s, 初始位置: (%.3fm, %.3fm, %.3fm)",
                armor_id_, armor_type_.c_str(), 
                initial_position_mm.x() * MM_TO_M, 
                initial_position_mm.y() * MM_TO_M, 
                initial_position_mm.z() * MM_TO_M);
}

void ArmorTracker::update(const Eigen::Vector3d& position_mm, double time) {
    // 如果滤波器还未初始化
    if (!filter_initialized_) {
        // 添加到初始化缓冲区
        Eigen::Vector4d pose;
        pose << position_mm.x(), position_mm.y(), position_mm.z(), time;
        init_buffer_.push_back(pose);
        
        // 限制缓冲区大小，保留最新的观测
        while (init_buffer_.size() > 10) {
            init_buffer_.pop_front();
        }
        
        // 尝试初始化V4滤波器（需要至少3个点）
        if (position_filter_.init(init_buffer_)) {
            filter_initialized_ = true;
            RCLCPP_INFO(rclcpp::get_logger("ArmorTracker"),
                       "装甲板 %d 滤波器初始化成功，使用了 %zu 个观测点",
                       armor_id_, init_buffer_.size());
        }
    } else {
        // 滤波器已初始化，执行正常更新
        double dt = time - last_update_time_;
        if (dt > 0) {
            Eigen::Vector3d measurement = position_mm;
            position_filter_.update(measurement, dt);
        }
    }
    
    last_update_time_ = time;
    update_count_++;
    
    RCLCPP_DEBUG(rclcpp::get_logger("ArmorTracker"),
                 "更新装甲板 %d - 时间: %.3f, 更新次数: %d, 已初始化: %s",
                 armor_id_, time, update_count_, filter_initialized_ ? "是" : "否");
}

Eigen::Vector3d ArmorTracker::predictPosition(double predict_time) const {
    if (!filter_initialized_) {
        // 如果滤波器未初始化，返回最后的观测位置
        if (!init_buffer_.empty()) {
            const auto& last_pose = init_buffer_.back();
            return Eigen::Vector3d(last_pose(0), last_pose(1), last_pose(2));
        }
        return Eigen::Vector3d::Zero();
    }
    
    // 使用V4滤波器预测
    auto predicted_state = position_filter_.predict(predict_time);
    
    // 提取位置（前3个状态）
    return Eigen::Vector3d(predicted_state(0), predicted_state(1), predicted_state(2));
}

Eigen::Vector3d ArmorTracker::predictVelocity(double predict_time) const {
    if (!filter_initialized_) {
        return Eigen::Vector3d::Zero();
    }
    
    // 预测未来状态
    auto predicted_state = position_filter_.predict(predict_time);
    
    // V4状态向量：[x, y, z, v, vz, angle, w, a]
    // 从极坐标速度转换到笛卡尔速度
    double v_horizontal = predicted_state(3);  // 水平速度大小
    double vz = predicted_state(4);           // 垂直速度
    double angle = predicted_state(5);         // 运动方向角
    
    // 计算笛卡尔速度分量
    double vx = v_horizontal * std::cos(angle);
    double vy = v_horizontal * std::sin(angle);
    
    return Eigen::Vector3d(vx, vy, vz);
}

// TargetPredictorNode 实现
TargetPredictorNode::TargetPredictorNode()
    : Node("target_predictor_node"),
      current_prediction_time_(0.03) {  // 初始预测时间设为30ms
    
    // 设置默认配置文件路径
    config_.config_file_path = "/home/guo/ITL_sentry_auto/src/config/robomaster_vision_config.yaml";
    
    // 检查环境变量是否指定了配置文件路径
    const char* config_env = std::getenv("TARGET_PREDICTOR_CONFIG_PATH");
    if (config_env) {
        config_.config_file_path = std::string(config_env);
    }
    
    // 加载配置文件
    if (!loadConfigFromYAML(config_.config_file_path)) {
        RCLCPP_WARN(this->get_logger(), 
                    "无法加载配置文件 %s，使用默认参数", 
                    config_.config_file_path.c_str());
    }
    
    // 初始化延迟历史队列
    latency_history_.clear();
    
    // 创建ROS2接口 - 使用QoS配置确保与解算节点匹配
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
    
    // 订阅解算节点发布的目标数据
    targets_sub_ = this->create_subscription<autoaim_msgs::msg::Targets>(
        "/pnp_solver/targets", qos,
        std::bind(&TargetPredictorNode::targetsCallback, this, std::placeholders::_1));
    
    // 发布预测结果
    predictions_pub_ = this->create_publisher<autoaim_msgs::msg::Predictions>(
        "/predictor/armor_predictions", 10);
    
    // 创建定时器，按照配置的频率发布预测
    prediction_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(config_.prediction_interval * 1000)),
        std::bind(&TargetPredictorNode::predictionTimerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), 
                "目标预测节点初始化完成 - 使用TrackQueueV4滤波器");
    RCLCPP_INFO(this->get_logger(),
                "预测配置 - 基础补偿时间: %.2fms, 发布频率: %.0fHz",
                config_.base_prediction_time * 1000, 1.0 / config_.prediction_interval);
    RCLCPP_INFO(this->get_logger(),
                "V4滤波器特点：极坐标运动模型，适合曲线运动目标");
}

bool TargetPredictorNode::loadConfigFromYAML(const std::string& config_path) {
    try {
        // 检查配置文件是否存在
        if (!std::filesystem::exists(config_path)) {
            RCLCPP_ERROR(this->get_logger(), 
                        "配置文件不存在: %s", config_path.c_str());
            return false;
        }
        
        YAML::Node config = YAML::LoadFile(config_path);
        
        // 加载预测器配置
        if (config["predictor"]) {
            const auto& predictor = config["predictor"];
            
            // 加载预测参数
            if (predictor["prediction"]) {
                config_.base_prediction_time = predictor["prediction"]["base_time"].as<double>(0.01);
                config_.prediction_interval = predictor["prediction"]["interval"].as<double>(0.01);
                
                // 延迟相关参数
                config_.latency_filter_alpha = predictor["prediction"]["latency_filter_alpha"].as<double>(0.3);
                config_.latency_history_size = predictor["prediction"]["latency_history_size"].as<int>(20);
                config_.max_prediction_time = predictor["prediction"]["max_time"].as<double>(0.1);
                config_.min_prediction_time = predictor["prediction"]["min_time"].as<double>(0.01);
            }
            
            // 加载跟踪器管理参数
            if (predictor["tracker"]) {
                config_.tracker_timeout = predictor["tracker"]["timeout"].as<double>(1.0);
                config_.min_update_count = predictor["tracker"]["min_update_count"].as<int>(3);
                config_.min_init_count = predictor["tracker"]["min_init_count"].as<int>(3);
            }
            
            // 注意：V4版本不需要加载EKF噪声参数，因为它们是内置的
        }
        
        RCLCPP_INFO(this->get_logger(), "成功加载配置文件");
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), 
                    "加载配置文件时发生错误: %s", e.what());
        return false;
    }
}

void TargetPredictorNode::updateLatencyEstimate(double new_latency) {
    std::lock_guard<std::mutex> lock(latency_mutex_);
    
    // 添加新的延迟测量到历史记录
    latency_history_.push_back(new_latency);
    
    // 保持历史记录在指定大小内
    while (latency_history_.size() > static_cast<size_t>(config_.latency_history_size)) {
        latency_history_.pop_front();
    }
    
    // 使用指数移动平均更新预测时间
    if (latency_history_.size() == 1) {
        // 第一个样本，直接使用
        current_prediction_time_ = new_latency + config_.base_prediction_time;
    } else {
        // 指数移动平均
        current_prediction_time_ = config_.latency_filter_alpha * (new_latency + config_.base_prediction_time) + 
                                  (1.0 - config_.latency_filter_alpha) * current_prediction_time_;
    }
    
    // 限制预测时间在合理范围内
    current_prediction_time_ = std::max(config_.min_prediction_time, 
                                       std::min(config_.max_prediction_time, current_prediction_time_));
}

void TargetPredictorNode::targetsCallback(const autoaim_msgs::msg::Targets::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(trackers_mutex_);
    
    // 获取当前时间戳并计算延迟
    double current_time = rclcpp::Time(this->now()).seconds();
    double message_time = rclcpp::Time(msg->header.stamp).seconds();
    double latency = current_time - message_time;
    
    // 更新延迟估计
    updateLatencyEstimate(latency);
    
    // 输出当前延迟信息
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,  // 每秒最多输出一次
                        "系统延迟: %.2fms, 当前预测时间: %.2fms",
                        latency * 1000, current_prediction_time_ * 1000);
    
    // 处理每个检测到的目标
    for (const auto& target : msg->targets) {
        // 解析装甲板ID（从字符串转换为整数）
        int armor_id = 0;
        try {
            armor_id = std::stoi(target.id);
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), 
                       "无法解析装甲板ID: %s", target.id.c_str());
            continue;
        }
        
        // ========== 核心改动：直接使用解算端的XYZ坐标 ==========
        // 从消息中获取位置，转换单位从米到毫米
        Eigen::Vector3d position_mm(
            target.position.x * ArmorTracker::M_TO_MM,
            target.position.y * ArmorTracker::M_TO_MM,
            target.position.z * ArmorTracker::M_TO_MM
        );
        
        // 添加调试信息
        RCLCPP_DEBUG(this->get_logger(),
                    "使用XYZ坐标 - ID: %d, 位置: (%.3fm, %.3fm, %.3fm)",
                    armor_id,
                    target.position.x,
                    target.position.y,
                    target.position.z);
        
        // 查找或创建对应的跟踪器
        auto it = trackers_.find(armor_id);
        if (it == trackers_.end()) {
            // 新发现的装甲板，创建新的跟踪器
            trackers_[armor_id] = std::make_unique<ArmorTracker>(
                armor_id, target.armor_type, position_mm, message_time
            );
            
            RCLCPP_INFO(this->get_logger(), 
                       "发现新装甲板 - ID: %d, 类型: %s, 位置: (%.2fm, %.2fm, %.2fm)",
                       armor_id, target.armor_type.c_str(), 
                       target.position.x, target.position.y, target.position.z);
        } else {
            // 已存在的装甲板，更新跟踪器
            it->second->update(position_mm, message_time);
        }
    }
    
    // 清理超时的跟踪器
    cleanupOldTrackers(message_time);
}

void TargetPredictorNode::predictionTimerCallback() {
    std::lock_guard<std::mutex> lock(trackers_mutex_);
    
    // 如果没有跟踪的目标，直接返回
    if (trackers_.empty()) {
        return;
    }
    
    // 获取当前的预测时间
    double prediction_time;
    {
        std::lock_guard<std::mutex> latency_lock(latency_mutex_);
        prediction_time = current_prediction_time_;
    }
    
    // 创建预测消息
    auto predictions_msg = std::make_unique<autoaim_msgs::msg::Predictions>();
    predictions_msg->header.stamp = this->now();
    predictions_msg->header.frame_id = "world";
    
    // 对每个跟踪器进行预测
    for (const auto& [armor_id, tracker] : trackers_) {
        // 检查跟踪器是否已初始化且更新次数足够
        if (!tracker->isInitialized() || tracker->getUpdateCount() < config_.min_update_count) {
            continue;
        }
        
        // 使用动态的预测时间进行预测
        Eigen::Vector3d predicted_pos_mm = tracker->predictPosition(prediction_time);
        Eigen::Vector3d predicted_vel_mm = tracker->predictVelocity(prediction_time);
        
        // 创建单个装甲板的预测消息
        autoaim_msgs::msg::Prediction prediction;
        prediction.armor_id = armor_id;
        prediction.armor_type = tracker->getArmorType();
        
        // 转换为米单位并填充消息
        prediction.position.x = predicted_pos_mm.x() * ArmorTracker::MM_TO_M;
        prediction.position.y = predicted_pos_mm.y() * ArmorTracker::MM_TO_M;
        prediction.position.z = predicted_pos_mm.z() * ArmorTracker::MM_TO_M;
        
        prediction.velocity.x = predicted_vel_mm.x() * ArmorTracker::MM_TO_M;
        prediction.velocity.y = predicted_vel_mm.y() * ArmorTracker::MM_TO_M;
        prediction.velocity.z = predicted_vel_mm.z() * ArmorTracker::MM_TO_M;
        
        // 计算预测位置的YPD坐标（用于输出）
        filter_lib::math::YpdCoord predicted_ypd = millimetersXyzToYpd(predicted_pos_mm);
        prediction.yaw = predicted_ypd.yaw;
        prediction.pitch = predicted_ypd.pitch;
        prediction.distance = predicted_ypd.dis * ArmorTracker::MM_TO_M;
        
        predictions_msg->predictions.push_back(prediction);
    }
    
    // 发布预测结果
    if (!predictions_msg->predictions.empty()) {
        predictions_pub_->publish(*predictions_msg);
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,  // 每5秒输出一次
                           "发布预测 - 目标数: %zu, 预测时间: %.2fms",
                           predictions_msg->predictions.size(), 
                           prediction_time * 1000);
    }
}

void TargetPredictorNode::cleanupOldTrackers(double current_time) {
    // 遍历所有跟踪器，移除超时的
    for (auto it = trackers_.begin(); it != trackers_.end(); ) {
        double time_since_update = current_time - it->second->getLastUpdateTime();
        
        if (time_since_update > config_.tracker_timeout) {
            RCLCPP_INFO(this->get_logger(), 
                       "移除超时装甲板 %d - 超时时间: %.2fs",
                       it->first, time_since_update);
            it = trackers_.erase(it);
        } else {
            ++it;
        }
    }
}

// 辅助函数：将笛卡尔坐标转换为YPD（用于输出）
filter_lib::math::YpdCoord TargetPredictorNode::millimetersXyzToYpd(
    const Eigen::Vector3d& position_mm) {
    double pos_array[3] = {position_mm.x(), position_mm.y(), position_mm.z()};
    double ypd_array[3];
    
    filter_lib::math::ceres_xyz_to_ypd(pos_array, ypd_array);
    
    return filter_lib::math::YpdCoord(ypd_array[0], ypd_array[1], ypd_array[2]);
}

} // namespace aimer

// 主函数
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<aimer::TargetPredictorNode>();
        RCLCPP_INFO(rclcpp::get_logger("main"), "装甲板预测节点启动成功");
        RCLCPP_INFO(rclcpp::get_logger("main"), 
                   "使用TrackQueueV4滤波器 - 极坐标运动模型");
        RCLCPP_INFO(rclcpp::get_logger("main"), 
                   "直接使用解算端的XYZ坐标，避免坐标转换误差");
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), 
                    "节点运行异常: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}