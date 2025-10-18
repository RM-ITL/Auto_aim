/**
 * @file single_model_prediction_node.hpp
 * @brief 使用SingleTargetMotionModel的预测节点
 */

#ifndef SINGLE_MODEL_PREDICTION_NODE_HPP_
#define SINGLE_MODEL_PREDICTION_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <autoaim_msgs/msg/targets.hpp>
#include <autoaim_msgs/msg/predictions.hpp>
#include <autoaim_msgs/msg/prediction.hpp>

#include "common_model_v2.hpp"
#include "base/math.hpp"

#include <memory>
#include <chrono>
#include <mutex>

namespace predict {

class SingleModelPredictionNode : public rclcpp::Node {
public:
    SingleModelPredictionNode();

private:
    // 回调函数
    void targetsCallback(const autoaim_msgs::msg::Targets::SharedPtr msg);
    void predictionTimerCallback();
    void statusTimerCallback();
    
    // 辅助函数
    bool isOrientationValid(double yaw_self_deg) const;
    filter_lib::math::YpdCoord metersXyzToYpd(const Eigen::Vector3d& position_m);
    double getCurrentTime() const;

private:
    // ROS2接口
    rclcpp::Subscription<autoaim_msgs::msg::Targets>::SharedPtr targets_sub_;
    rclcpp::Publisher<autoaim_msgs::msg::Predictions>::SharedPtr predictions_pub_;
    rclcpp::TimerBase::SharedPtr prediction_timer_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    
    // 运动模型 - 使用SingleTargetMotionModel替代TestModel
    std::unique_ptr<motion_model::SingleTargetMotionModel> motion_model_;
    std::mutex model_mutex_;
    
    // 跟踪状态
    struct TrackingState {
        bool is_tracking = false;
        Eigen::Vector3d last_position = Eigen::Vector3d::Zero();
        double last_update_time = 0.0;
        std::string target_type;
        double last_observation_time = 0.0;
        double observation_timestamp = 0.0;  // 原始观测时间戳
    } tracking_state_;
    
    // 配置参数
    struct Config {
        // 预测参数
        double base_prediction_time = 0.0;       // 基础预测时间
        double prediction_interval = 0.01;       // 预测发布间隔（秒）
        double max_prediction_time = 0.5;        // 最大预测时间（秒）
        
        // 跟踪参数
        double tracker_timeout = 1.0;
        double position_continuity_threshold = 0.5;
        
        // 朝向角验证
        double yaw_self_invalid_value = -114.514;
        double yaw_self_min_valid = -180.0;
        double yaw_self_max_valid = 180.0;
        
        double observation_timeout = 0.1;
        
        // 是否启用动态时间补偿
        bool enable_dynamic_compensation = true;
        
        // 滤波器参数（用于SingleTargetMotionModel）
        double q_pos = 0.1;
        double q_vel = 0.1;
        double q_acc = 0.1;
        double q_angle = 0.1;
        double q_omega = 0.1;
        double q_alpha = 0.1;
        double r_pos = 0.1;
        double r_angle = 0.1;
    } config_;
    
    // 统计信息
    struct Statistics {
        size_t total_messages = 0;
        size_t total_predictions = 0;
        double last_latency = 0.0;
        double last_prediction_time = 0.0;
    } stats_;
};

} // namespace aimer

#endif // SINGLE_MODEL_PREDICTION_NODE_HPP_