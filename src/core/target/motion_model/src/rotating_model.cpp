/**
 * @file rotating_motion_model.cpp
 * @brief 旋转目标运动模型实现
 */

#include "rotating_model.hpp"
#include <sstream>
#include <cmath>

namespace motion_model {

RotatingMotionModel::RotatingMotionModel(const Params& params)
    : params_(params), initialized_(false), update_count_(0), last_timestamp_(0.0) {
    
    // 创建滤波器参数
    filter_lib::RotatingTargetTracker::Params tracker_params;
    tracker_params.radius_min = params.radius_min;
    tracker_params.radius_max = params.radius_max;
    tracker_params.armor_count = params.armor_count;
    tracker_params.enable_height_filter = params.enable_height_filter;
    tracker_params.min_update_count = params.min_fire_updates;
    tracker_params.max_fire_delay = params.fire_delay;
    tracker_params.armor_fire_angle = params.armor_angle_thr;
    tracker_params.center_fire_angle = params.center_angle_thr;
    
    tracker_ = std::make_unique<filter_lib::RotatingTargetTracker>(tracker_params);
}

void RotatingMotionModel::update(const RotatingObservation& obs) {
    // 构建滤波器观测
    filter_lib::RotatingTargetTracker::Observation filter_obs(
        obs.position.x(),
        obs.position.y(), 
        obs.position.z(),
        obs.orientation,
        obs.timestamp
    );
    
    if (!initialized_) {
        // 收集初始化数据
        init_buffer_.push_back(filter_obs);
        
        // 清理过旧的数据 - 现在 init_buffer_ 是 deque，可以使用 pop_front()
        while (!init_buffer_.empty() && 
               (obs.timestamp - init_buffer_.front().timestamp) > params_.timeout) {
            init_buffer_.pop_front();
        }
        
        // 尝试初始化
        if (init_buffer_.size() >= 3) {
            if (tracker_->init(init_buffer_)) {
                initialized_ = true;
                update_count_ = init_buffer_.size();
                init_buffer_.clear();
            }
        }
    } else {
        // 检查超时
        if ((obs.timestamp - last_timestamp_) > params_.timeout) {
            reset();
            update(obs);
            return;
        }
        
        // 正常更新
        tracker_->update(filter_obs);
        update_count_++;
    }
    
    last_timestamp_ = obs.timestamp;
}


RotatingPrediction RotatingMotionModel::predict(double dt) const {
    RotatingPrediction pred;
    
    if (!initialized_ || !tracker_->initialized()) {
        pred.valid = false;
        return pred;
    }
    
    // 获取装甲板预测
    auto armor_pred = tracker_->predict_armor(dt);
    pred.armor_position = Eigen::Vector3d(armor_pred(0), armor_pred(1), armor_pred(2));
    pred.armor_orientation = armor_pred(3);
    
    // 获取中心预测
    auto center_pred = tracker_->predict_center(dt);
    pred.center_position = Eigen::Vector3d(center_pred(0), center_pred(1), center_pred(2));
    
    // 获取状态信息
    auto status = tracker_->status();
    pred.center_velocity = Eigen::Vector3d(status.center_vel.x(), status.center_vel.y(), 0);
    pred.angular_velocity = status.omega;
    pred.radius = status.radius;
    pred.current_armor = status.current_armor;
    
    // 检查开火条件
    pred.armor_fire_ready = tracker_->check_fire_armor(armor_pred);
    pred.center_fire_ready = tracker_->check_fire_center(center_pred);
    
    pred.valid = true;
    
    return pred;
}

std::vector<std::string> RotatingMotionModel::status() const {
    std::vector<std::string> info;
    std::stringstream ss;
    
    // 基本信息
    info.push_back("Model: Rotating Target Tracker");
    
    ss << "Initialized: " << (initialized_ ? "Yes" : "No");
    info.push_back(ss.str());
    ss.str("");
    
    ss << "Update Count: " << update_count_;
    info.push_back(ss.str());
    ss.str("");
    
    ss << "Armor Count: " << params_.armor_count;
    info.push_back(ss.str());
    
    // 如果已初始化，添加详细状态
    if (initialized_ && tracker_) {
        auto status = tracker_->status();
        
        info.push_back("--- Tracker Status ---");
        
        ss.str("");
        ss << "Center: [" << status.center.x() << ", " 
           << status.center.y() << ", " << status.center.z() << "]";
        info.push_back(ss.str());
        
        ss.str("");
        ss << "Radius: " << status.radius << " m";
        info.push_back(ss.str());
        
        ss.str("");
        ss << "Angular Velocity: " << status.omega << " rad/s";
        info.push_back(ss.str());
        
        ss.str("");
        ss << "Current Armor: " << status.current_armor;
        info.push_back(ss.str());
        
        ss.str("");
        ss << "Ready to Fire: " << (status.ready_to_fire ? "Yes" : "No");
        info.push_back(ss.str());
    }
    
    return info;
}

bool RotatingMotionModel::is_timeout(double current_time) const {
    if (!initialized_ || update_count_ == 0) {
        return false;
    }
    
    return (current_time - last_timestamp_) > params_.timeout;
}

void RotatingMotionModel::reset() {
    tracker_->reset();
    init_buffer_.clear();
    initialized_ = false;
    update_count_ = 0;
    last_timestamp_ = 0.0;
}

void RotatingMotionModel::set_params(const Params& params) {
    params_ = params;
    
    // 更新滤波器参数
    filter_lib::RotatingTargetTracker::Params tracker_params;
    tracker_params.radius_min = params.radius_min;
    tracker_params.radius_max = params.radius_max;
    tracker_params.armor_count = params.armor_count;
    tracker_params.enable_height_filter = params.enable_height_filter;
    tracker_params.min_update_count = params.min_fire_updates;
    tracker_params.max_fire_delay = params.fire_delay;
    tracker_params.armor_fire_angle = params.armor_angle_thr;
    tracker_params.center_fire_angle = params.center_angle_thr;
    
    if (tracker_) {
        tracker_->set_params(tracker_params);
    }
}

void RotatingMotionModel::set_process_noise(double pos, double vel, double angle, double omega) {
    if (tracker_) {
        // 构建主滤波器过程噪声矩阵
        Eigen::Matrix<double, 9, 9> Q = Eigen::Matrix<double, 9, 9>::Zero();
        
        // 位置噪声
        Q(0,0) = Q(1,1) = Q(2,2) = pos * pos;
        
        // 角度噪声
        Q(3,3) = angle * angle;
        
        // 速度噪声
        Q(4,4) = Q(5,5) = Q(6,6) = vel * vel;
        
        // 角速度噪声
        Q(7,7) = omega * omega;
        
        // 半径噪声（较小）
        Q(8,8) = 0.001;
        
        tracker_->set_main_process_noise(Q);
    }
}

void RotatingMotionModel::set_measurement_noise(double pos, double angle) {
    if (tracker_) {
        // 构建测量噪声矩阵
        Eigen::Matrix<double, 4, 4> R = Eigen::Matrix<double, 4, 4>::Zero();
        
        // 位置测量噪声
        R(0,0) = R(1,1) = R(2,2) = pos * pos;
        
        // 角度测量噪声
        R(3,3) = angle * angle;
        
        tracker_->set_main_measurement_noise(R);
    }
}

} // namespace motion_model