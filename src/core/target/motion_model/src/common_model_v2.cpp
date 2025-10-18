/**
 * @file single_target_motion_model.cpp
 * @brief 单目标运动模型实现
 */

#include "common_model_v2.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace motion_model {

SingleTargetMotionModel::SingleTargetMotionModel()
    : filter_params_(), filter_(filter_params_), 
      initialized_(false), update_count_(0), last_timestamp_(0.0) {
}

SingleTargetMotionModel::SingleTargetMotionModel(const filter_lib::TrackQueueV3::Parameters& params)
    : filter_params_(params), filter_(params), 
      initialized_(false), update_count_(0), last_timestamp_(0.0) {
}

bool SingleTargetMotionModel::initialize(const std::deque<Observation>& observations) {
    if (observations.size() < 2) {
        std::cerr << "[MotionModel] Need at least 2 observations for initialization" << std::endl;
        return false;
    }
    
    // 转换观测格式
    std::deque<filter_lib::TrackQueueV3::Observation> filter_obs;
    for (const auto& obs : observations) {
        filter_obs.push_back(convert_observation(obs));
    }
    
    // 初始化滤波器
    if (filter_.init(filter_obs)) {
        initialized_ = true;
        update_count_ = observations.size();
        last_timestamp_ = observations.back().timestamp;
        init_buffer_.clear();  // 清空缓冲
        
        std::cout << "[MotionModel] Successfully initialized with " 
                  << observations.size() << " observations" << std::endl;
        return true;
    }
    
    std::cerr << "[MotionModel] Filter initialization failed" << std::endl;
    return false;
}

void SingleTargetMotionModel::update(const Observation& obs) {
    auto filter_obs = convert_observation(obs);
    
    if (!initialized_) {
        // 收集初始化数据
        init_buffer_.push_back(filter_obs);
        
        if (init_buffer_.size() >= 2) {
            if (filter_.init(init_buffer_)) {
                initialized_ = true;
                update_count_ = init_buffer_.size();
                init_buffer_.clear();
                std::cout << "[MotionModel] Auto-initialized with buffered observations" << std::endl;
            } else if (init_buffer_.size() > 5) {
                // 防止缓冲区过大
                init_buffer_.pop_front();
            }
        }
    } else {
        // 正常更新
        double dt = obs.timestamp - last_timestamp_;
        
        if (dt > 0 && dt < 5.0) {
            filter_lib::TrackQueueV3::Meas meas;
            meas << obs.position.x(), obs.position.y(), obs.position.z(), obs.orientation;
            filter_.update(meas, dt);
            update_count_++;
            
            if (dt > 1.0) {
                std::cout << "[MotionModel] Large time gap: " << dt << " seconds" << std::endl;
            }
        } else if (dt <= 0) {
            std::cerr << "[MotionModel] Invalid time interval: " << dt << std::endl;
            return;
        } else {
            std::cerr << "[MotionModel] Time interval too large: " << dt 
                     << " seconds, skipping update" << std::endl;
            return;
        }
    }
    
    last_timestamp_ = obs.timestamp;
}

Prediction SingleTargetMotionModel::predict(double dt) const {
    Prediction pred;
    
    if (!initialized_ || dt < 0) {
        pred.valid = false;
        return pred;
    }
    
    if (dt > 10.0) {
        std::cout << "[MotionModel] Warning: large prediction interval " << dt << " seconds" << std::endl;
    }
    
    // 获取预测状态
    auto state = filter_.predict(dt);
    
    // 填充预测结果
    pred.position = Eigen::Vector3d(state(0), state(1), state(2));
    pred.velocity = Eigen::Vector3d(state(4), state(5), state(6));
    pred.acceleration = Eigen::Vector3d(state(8), state(9), 0.0);
    pred.orientation = state(3);
    pred.angular_velocity = state(7);
    pred.angular_acceleration = state(10);
    pred.valid = true;
    
    return pred;
}

Prediction SingleTargetMotionModel::get_current_state() const {
    Prediction current;
    
    if (!initialized_) {
        current.valid = false;
        return current;
    }
    
    auto state = filter_.state();
    
    current.position = filter_.position();
    current.velocity = filter_.velocity();
    current.acceleration = filter_.acceleration();
    current.orientation = filter_.orientation();
    current.angular_velocity = filter_.angular_velocity();
    current.angular_acceleration = filter_.angular_acceleration();
    current.valid = true;
    
    return current;
}

TrackerStatus SingleTargetMotionModel::get_status() const {
    TrackerStatus status;
    status.initialized = initialized_;
    status.update_count = update_count_;
    status.last_timestamp = last_timestamp_;
    
    if (initialized_) {
        // 从协方差矩阵估计误差
        auto covariance = filter_.covariance();
        status.position_error = std::sqrt(covariance(0,0) + covariance(1,1) + covariance(2,2));
        status.orientation_error = std::sqrt(covariance(3,3));
        
        // 获取跟踪信息
        auto tracking_info = filter_.getTrackingInfo();
        status.speed = tracking_info.speed;
        status.heading_rate = tracking_info.heading_rate;
    } else {
        status.position_error = 1e6;
        status.orientation_error = 1e6;
        status.speed = 0.0;
        status.heading_rate = 0.0;
    }
    
    return status;
}

void SingleTargetMotionModel::reset() {
    filter_.reset();
    init_buffer_.clear();
    initialized_ = false;
    update_count_ = 0;
    last_timestamp_ = 0.0;
}

void SingleTargetMotionModel::set_parameters(const filter_lib::TrackQueueV3::Parameters& params) {
    filter_params_ = params;
    filter_.setParameters(params);
}

Eigen::Matrix<double, 11, 11> SingleTargetMotionModel::get_covariance() const {
    if (!initialized_) {
        return Eigen::Matrix<double, 11, 11>::Identity() * 1e6;
    }
    return filter_.covariance();
}

void SingleTargetMotionModel::set_process_noise(double pos, double vel, double acc, 
                                                 double angle, double omega, double alpha) {
    filter_params_.q_pos = pos;
    filter_params_.q_vel = vel;
    filter_params_.q_acc = acc;
    filter_params_.q_angle = angle;
    filter_params_.q_omega = omega;
    filter_params_.q_alpha = alpha;
    filter_.setParameters(filter_params_);
}

void SingleTargetMotionModel::set_measurement_noise(double pos, double angle) {
    filter_params_.r_pos = pos;
    filter_params_.r_angle = angle;
    filter_.setParameters(filter_params_);
}

filter_lib::TrackQueueV3::Observation 
SingleTargetMotionModel::convert_observation(const Observation& obs) const {
    return filter_lib::TrackQueueV3::Observation(
        obs.position.x(),
        obs.position.y(),
        obs.position.z(),
        obs.orientation,
        obs.timestamp
    );
}

} // namespace motion_model