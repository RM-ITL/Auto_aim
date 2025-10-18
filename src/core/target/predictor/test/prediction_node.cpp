/**
 * @file single_model_prediction_node.cpp
 * @brief 使用SingleTargetMotionModel的预测节点实现
 */

#include "prediction_node.hpp"
#include <chrono>

namespace predict {

SingleModelPredictionNode::SingleModelPredictionNode() : Node("single_model_prediction_node") {
    // 创建SingleTargetMotionModel实例
    motion_model_ = std::make_unique<motion_model::SingleTargetMotionModel>();
    
    // 设置滤波器噪声参数
    motion_model_->set_process_noise(
        config_.q_pos, config_.q_vel, config_.q_acc,
        config_.q_angle, config_.q_omega, config_.q_alpha
    );
    motion_model_->set_measurement_noise(config_.r_pos, config_.r_angle);
    
    // 创建ROS2接口
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10))
        .best_effort()
        .durability_volatile();
    
    // 订阅目标话题
    targets_sub_ = this->create_subscription<autoaim_msgs::msg::Targets>(
        "/pnp_solver/targets", qos,
        std::bind(&SingleModelPredictionNode::targetsCallback, this, std::placeholders::_1));
    
    // 发布预测结果
    predictions_pub_ = this->create_publisher<autoaim_msgs::msg::Predictions>(
        "/predictor/armor_predictions", 10);
    
    // 创建定时器
    prediction_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(config_.prediction_interval * 1000)),
        std::bind(&SingleModelPredictionNode::predictionTimerCallback, this));
    
    status_timer_ = this->create_wall_timer(
        std::chrono::seconds(5),
        std::bind(&SingleModelPredictionNode::statusTimerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), 
                "单目标预测节点已启动（动态时间补偿%s）",
                config_.enable_dynamic_compensation ? "已启用" : "已禁用");
}

void SingleModelPredictionNode::targetsCallback(const autoaim_msgs::msg::Targets::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    // 记录当前时间（接收时刻）
    double current_time = getCurrentTime();
    tracking_state_.last_observation_time = current_time;
    
    // 如果没有检测到目标
    if (msg->targets.empty()) {
        if (tracking_state_.is_tracking) {
            RCLCPP_WARN(this->get_logger(), "目标丢失");
            tracking_state_.is_tracking = false;
            motion_model_->reset();
        }
        return;
    }
    
    // 选择第一个有效的目标
    const autoaim_msgs::msg::Target* selected_target = nullptr;
    for (const auto& target : msg->targets) {
        if (isOrientationValid(target.yaw_self)) {
            selected_target = &target;
            break;
        }
    }
    
    if (!selected_target) {
        return;
    }
    
    // 获取目标信息
    Eigen::Vector3d armor_position(
        selected_target->position.x,
        selected_target->position.y,
        selected_target->position.z
    );
    double yaw_self_rad = selected_target->yaw_self * M_PI / 180.0;
    
    // 提取消息中的原始时间戳（图像采集时刻）
    double message_time = rclcpp::Time(msg->header.stamp).seconds();
    tracking_state_.observation_timestamp = message_time;
    
    // 计算系统延迟
    stats_.last_latency = current_time - message_time;
    
    if (stats_.total_messages % 50 == 0) {
        RCLCPP_INFO(this->get_logger(), 
                   "系统延迟: %.1fms", 
                   stats_.last_latency * 1000);
    }
    
    // 检查是否需要重置跟踪器
    if (tracking_state_.is_tracking) {
        double time_gap = current_time - tracking_state_.last_update_time;
        double position_change = (armor_position - tracking_state_.last_position).norm();
        
        if (time_gap > config_.tracker_timeout || 
            (position_change > config_.position_continuity_threshold && time_gap < 0.1)) {
            RCLCPP_WARN(this->get_logger(), "目标切换或超时，重置跟踪器");
            motion_model_->reset();
        }
    } else {
        tracking_state_.is_tracking = true;
        RCLCPP_INFO(this->get_logger(), "开始跟踪目标: %s", selected_target->armor_type.c_str());
    }
    
    // 更新跟踪状态
    tracking_state_.last_position = armor_position;
    tracking_state_.last_update_time = current_time;
    tracking_state_.target_type = selected_target->armor_type;
    
    // 创建观测数据 - 使用SingleTargetMotionModel的Observation结构
    motion_model::Observation observation(
        armor_position,
        yaw_self_rad,
        message_time  // 使用原始时间戳
    );
    
    // 更新运动模型
    motion_model_->update(observation);
    stats_.total_messages++;
}

void SingleModelPredictionNode::predictionTimerCallback() {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    // 检查模型是否已初始化
    if (!motion_model_->is_initialized() || !tracking_state_.is_tracking) {
        return;
    }
    
    // 检查观测是否超时
    double current_time = getCurrentTime();
    double time_since_last_observation = current_time - tracking_state_.last_observation_time;
    
    if (time_since_last_observation > config_.observation_timeout) {
        if (tracking_state_.is_tracking) {
            RCLCPP_WARN(this->get_logger(), 
                       "观测数据超时 %.3fs，停止预测", 
                       time_since_last_observation);
            tracking_state_.is_tracking = false;
            motion_model_->reset();
        }
        return;
    }
    
    // 动态计算预测时间
    double prediction_time;
    
    if (config_.enable_dynamic_compensation) {
        // 计算当前延迟
        double current_delay = current_time - tracking_state_.observation_timestamp;
        
        // 组合延迟补偿和额外预测
        prediction_time = current_delay + config_.base_prediction_time;
        
        // 限制预测时间
        if (prediction_time > config_.max_prediction_time) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), 
                                *this->get_clock(), 
                                1000,
                                "预测时间过长: %.3fs，限制为%.3fs",
                                prediction_time, config_.max_prediction_time);
            prediction_time = config_.max_prediction_time;
        }
        
        stats_.last_prediction_time = prediction_time;
        
        if (stats_.total_predictions % 100 == 0) {
            RCLCPP_INFO(this->get_logger(),
                       "动态预测: 延迟=%.1fms, 预测时间=%.1fms",
                       current_delay * 1000,
                       prediction_time * 1000);
        }
    } else {
        prediction_time = config_.base_prediction_time;
        stats_.last_prediction_time = prediction_time;
    }
    
    // 获取预测结果 - 使用SingleTargetMotionModel的predict方法
    auto prediction = motion_model_->predict(prediction_time);
    
    if (!prediction.valid) {
        return;
    }
    
    // 创建预测消息
    auto predictions_msg = std::make_unique<autoaim_msgs::msg::Predictions>();
    predictions_msg->header.stamp = this->now();
    predictions_msg->header.frame_id = "world";
    
    autoaim_msgs::msg::Prediction pred_msg;
    pred_msg.armor_id = 0;
    pred_msg.armor_type = tracking_state_.target_type;
    
    // 填充预测位置和速度
    pred_msg.position.x = prediction.position.x();
    pred_msg.position.y = prediction.position.y();
    pred_msg.position.z = prediction.position.z();
    
    pred_msg.velocity.x = prediction.velocity.x();
    pred_msg.velocity.y = prediction.velocity.y();
    pred_msg.velocity.z = prediction.velocity.z();
    
    // 朝向角预测
    pred_msg.has_orientation_prediction = true;
    pred_msg.orientation = prediction.orientation;
    pred_msg.angular_velocity = prediction.angular_velocity;
    
    // 不确定性估计
    auto covariance = motion_model_->get_covariance();
    double position_uncertainty = std::sqrt(covariance(0,0) + covariance(1,1) + covariance(2,2));
    double orientation_uncertainty = std::sqrt(covariance(3,3));
    
    double uncertainty_factor = 1.0 + prediction_time * 2.0;
    pred_msg.position_uncertainty = position_uncertainty * uncertainty_factor;
    pred_msg.orientation_uncertainty = orientation_uncertainty * uncertainty_factor;
    
    // 转换为YPD坐标
    Eigen::Vector3d pos_for_ypd(pred_msg.position.x, pred_msg.position.y, pred_msg.position.z);
    filter_lib::math::YpdCoord predicted_ypd = metersXyzToYpd(pos_for_ypd);
    pred_msg.yaw = predicted_ypd.yaw;
    pred_msg.pitch = predicted_ypd.pitch;
    pred_msg.distance = predicted_ypd.dis;
    
    predictions_msg->predictions.push_back(pred_msg);
    predictions_pub_->publish(*predictions_msg);
    
    stats_.total_predictions++;
}

void SingleModelPredictionNode::statusTimerCallback() {
    auto status = motion_model_->get_status();
    
    RCLCPP_INFO(this->get_logger(), 
                "状态统计:\n"
                "  接收消息: %zu\n"
                "  发布预测: %zu\n"
                "  系统延迟: %.1fms\n"
                "  预测时间: %.1fms\n"
                "  跟踪状态: %s\n"
                "  模型初始化: %s\n"
                "  更新次数: %d\n"
                "  位置误差: %.3fm\n"
                "  速度: %.2fm/s\n"
                "  补偿模式: %s",
                stats_.total_messages,
                stats_.total_predictions,
                stats_.last_latency * 1000,
                stats_.last_prediction_time * 1000,
                tracking_state_.is_tracking ? "跟踪中" : "未跟踪",
                status.initialized ? "是" : "否",
                status.update_count,
                status.position_error,
                status.speed,
                config_.enable_dynamic_compensation ? "动态补偿" : "固定预测");
}

bool SingleModelPredictionNode::isOrientationValid(double yaw_self_deg) const {
    if (std::abs(yaw_self_deg - config_.yaw_self_invalid_value) < 0.001) {
        return false;
    }
    
    if (yaw_self_deg < config_.yaw_self_min_valid || 
        yaw_self_deg > config_.yaw_self_max_valid) {
        return false;
    }
    
    if (std::isnan(yaw_self_deg) || std::isinf(yaw_self_deg)) {
        return false;
    }
    
    return true;
}

filter_lib::math::YpdCoord SingleModelPredictionNode::metersXyzToYpd(const Eigen::Vector3d& position_m) {
    filter_lib::math::YpdCoord ypd;
    
    ypd.dis = position_m.norm();
    
    if (ypd.dis < 1e-6) {
        ypd.yaw = 0.0;
        ypd.pitch = 0.0;
        return ypd;
    }
    
    ypd.yaw = std::atan2(position_m.y(), position_m.x());
    double horizontal_distance = std::sqrt(position_m.x() * position_m.x() + 
                                         position_m.y() * position_m.y());
    ypd.pitch = std::atan2(position_m.z(), horizontal_distance);
    
    return ypd;
}

double SingleModelPredictionNode::getCurrentTime() const {
    return this->now().seconds();
}

} // namespace aimer

// 主函数
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<aimer::SingleModelPredictionNode>();
    
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}