/**
 * @file rotating_target_tracker.cpp
 * @brief 旋转目标跟踪滤波器实现
 */

#include "filter/rotating_target_tracker.hpp"
#include <cmath>
#include <algorithm>

namespace filter_lib {

// 主EKF状态转移函数实现
template<typename T>
void RotatingTargetTracker::MainTransitionModel::operator()(
    const T x_prev[MAIN_DIM_STATE], T x_curr[MAIN_DIM_STATE]) const {
    
    // 位置更新: p = p + v * dt
    x_curr[0] = x_prev[0] + x_prev[4] * dt;  // x
    x_curr[1] = x_prev[1] + x_prev[5] * dt;  // y
    x_curr[2] = x_prev[2] + x_prev[6] * dt;  // z
    
    // 角度更新: theta = theta + omega * dt
    x_curr[3] = x_prev[3] + x_prev[7] * dt;  // theta
    
    // 速度和角速度保持不变（恒速模型）
    x_curr[4] = x_prev[4];  // vx
    x_curr[5] = x_prev[5];  // vy
    x_curr[6] = x_prev[6];  // vz
    x_curr[7] = x_prev[7];  // omega
    
    // 半径保持不变
    x_curr[8] = x_prev[8];  // r
}

// 主EKF观测函数实现
template<typename T>
void RotatingTargetTracker::MainMeasurementModel::operator()(
    const T x[MAIN_DIM_STATE], T y[MAIN_DIM_MEAS]) const {
    
    // 从中心状态计算装甲板位置
    T cos_theta = ceres::cos(x[3]);
    T sin_theta = ceres::sin(x[3]);
    
    // 装甲板位置 = 中心位置 + 半径 * 方向向量
    y[0] = x[0] + x[8] * cos_theta;  // armor_x
    y[1] = x[1] + x[8] * sin_theta;  // armor_y
    y[2] = x[2];                      // armor_z
    y[3] = x[3];                      // armor_theta
}

// 构造函数
RotatingTargetTracker::RotatingTargetTracker(const Params& params)
    : params_(params) {
    
    // 初始化主EKF
    main_ekf_ = std::make_unique<AdaptiveEkf<MAIN_DIM_STATE, MAIN_DIM_MEAS>>();
    
    // 设置默认噪声矩阵
    Q_main_ = Eigen::Matrix<double, MAIN_DIM_STATE, MAIN_DIM_STATE>::Identity();
    Q_main_ *= 0.01;
    
    R_main_ = Eigen::Matrix<double, MAIN_DIM_MEAS, MAIN_DIM_MEAS>::Identity();
    R_main_ *= 0.1;
    
    // 初始化中心位置滤波器
    std::vector<double> center_q = {0.01, 0.01, 0.1, 0.1};
    Eigen::Matrix<double, CENTER_DIM_MEAS, CENTER_DIM_MEAS> center_R;
    center_R << 0.1, 0,
                0, 0.1;
    center_filter_ = std::make_unique<Kalman<CENTER_DIM_MEAS, CENTER_DIM_STATE>>(
        center_q, center_R, 2);
    
    // 初始化角速度滤波器
    std::vector<double> omega_q = {0.01, 0.1, 0.01};
    omega_filter_ = std::make_unique<Kalman<OMEGA_DIM_MEAS, OMEGA_DIM_STATE>>(
        omega_q, 0.1, 2);
    
    // 初始化装甲板半径和高度
    for (int i = 0; i < 2; ++i) {
        armor_r_[i] = (params_.radius_min + params_.radius_max) / 2.0;
        armor_z_[i] = 0.0;
        
        if (params_.enable_height_filter) {
            weighted_z_[i] = SlideWeightedAvg<double>(20);
        }
    }
}

// 初始化函数
bool RotatingTargetTracker::init(const std::deque<Observation>& observations) {
    if (observations.size() < 3) {
        return false;
    }
    
    if (!estimate_initial_state(observations)) {
        return false;
    }
    
    initialized_ = true;
    update_count_ = observations.size();
    last_timestamp_ = observations.back().timestamp;
    
    return true;
}

// 更新函数
void RotatingTargetTracker::update(const Observation& obs) {
    if (!initialized_) {
        // 需要先初始化
        std::deque<Observation> init_obs = {obs};
        init(init_obs);
        return;
    }
    
    // 计算时间间隔
    double dt = obs.timestamp - last_timestamp_;
    
    // 处理异常时间间隔
    if (dt <= 0) {
        return;
    }
    
    if (dt > 1.0) {
        // 目标丢失超过1秒，重新初始化
        reset();
        std::deque<Observation> init_obs = {obs};
        init(init_obs);
        return;
    }
    
    // 检测装甲板切换
    MainState current_state = main_ekf_->get_x();
    double expected_theta = current_state(3) + current_state(7) * dt;
    int new_armor = detect_armor_switch(obs.theta, expected_theta);
    
    if (new_armor != current_armor_) {
        current_armor_ = new_armor;
    }
    
    // 主EKF更新
    MainTransitionModel transition(dt);
    MainMeasurementModel measurement;
    main_ekf_->update(measurement, transition, obs.vec(), Q_main_, R_main_);
    
    // 更新当前状态
    current_state = main_ekf_->get_x();
    
    // 计算装甲板中心位置
    double center_x = obs.x - current_state(8) * cos(obs.theta);
    double center_y = obs.y - current_state(8) * sin(obs.theta);
    
    // 更新中心位置滤波器
    center_filter_->update(center_x, center_y, obs.timestamp);
    
    // 更新角速度滤波器
    omega_filter_->update(obs.theta, obs.timestamp);
    
    // 更新装甲板特定参数
    update_armor_params(obs);
    
    last_timestamp_ = obs.timestamp;
    update_count_++;
}

// 预测装甲板位置
RotatingTargetTracker::Meas RotatingTargetTracker::predict_armor(double dt) const {
    if (!initialized_) {
        return Meas::Zero();
    }
    
    // 获取主EKF预测
    MainState predicted_state = main_ekf_->get_x();
    if (dt > 0) {
        MainTransitionModel transition(dt);
        auto predict_result = main_ekf_->predict(transition);
        predicted_state = predict_result.x_p;
    }
    
    // 计算装甲板位置
    Meas result;
    result(0) = predicted_state(0) + predicted_state(8) * cos(predicted_state(3));
    result(1) = predicted_state(1) + predicted_state(8) * sin(predicted_state(3));
    result(2) = armor_z_[current_armor_];
    result(3) = normalize_angle(predicted_state(3));
    
    return result;
}

// 预测中心位置
RotatingTargetTracker::Meas RotatingTargetTracker::predict_center(double dt) const {
    if (!initialized_) {
        return Meas::Zero();
    }
    
    double predict_time = last_timestamp_ + dt;
    
    // 从中心滤波器获取预测
    CenterState center_pred = center_filter_->predict(predict_time);
    
    // 从角速度滤波器获取角度预测
    OmegaState omega_pred = omega_filter_->predict(predict_time);
    
    Meas result;
    result(0) = center_pred(0);
    result(1) = center_pred(1);
    result(2) = (armor_z_[0] + armor_z_[1]) / 2.0;  // 使用平均高度
    result(3) = normalize_angle(omega_pred(0));
    
    return result;
}

// 获取详细状态
RotatingTargetTracker::Status RotatingTargetTracker::status() const {
    Status status;
    status.update_count = update_count_;
    status.current_armor = current_armor_;
    status.ready_to_fire = (update_count_ >= params_.min_update_count);
    
    if (main_ekf_) {
        auto state = main_ekf_->get_x();
        status.center = Eigen::Vector3d(state(0), state(1), state(2));
        status.center_vel = Eigen::Vector2d(state(4), state(5));
        status.radius = state(8);
    } else {
        status.center = Eigen::Vector3d::Zero();
        status.center_vel = Eigen::Vector2d::Zero();
        status.radius = (params_.radius_min + params_.radius_max) / 2.0;
    }
    
    if (omega_filter_) {
        status.omega = omega_filter_->get_x_k1()(1);
    } else {
        status.omega = 0.0;
    }
    
    return status;
}

// 检查是否适合开火（装甲板模式）
bool RotatingTargetTracker::check_fire_armor(const Meas& predicted_pose) const {
    if (update_count_ < params_.min_update_count) {
        return false;
    }
    
    Meas current_pred = predict_armor(params_.max_fire_delay);
    double angle_difference = std::abs(angle_diff(current_pred(3), predicted_pose(3)));
    
    return angle_difference < params_.armor_fire_angle;
}

// 检查是否适合开火（中心模式）
bool RotatingTargetTracker::check_fire_center(const Meas& predicted_pose) const {
    if (update_count_ < params_.min_update_count) {
        return false;
    }
    
    Meas current_pred = predict_center(params_.max_fire_delay);
    double angle_difference = std::abs(angle_diff(current_pred(3), predicted_pose(3)));
    
    return angle_difference < params_.center_fire_angle;
}

// 重置滤波器
void RotatingTargetTracker::reset() {
    main_ekf_ = std::make_unique<AdaptiveEkf<MAIN_DIM_STATE, MAIN_DIM_MEAS>>();
    
    // 重新初始化辅助滤波器
    std::vector<double> center_q = {0.01, 0.01, 0.1, 0.1};
    Eigen::Matrix<double, CENTER_DIM_MEAS, CENTER_DIM_MEAS> center_R;
    center_R << 0.1, 0,
                0, 0.1;
    center_filter_ = std::make_unique<Kalman<CENTER_DIM_MEAS, CENTER_DIM_STATE>>(
        center_q, center_R, 2);
    
    std::vector<double> omega_q = {0.01, 0.1, 0.01};
    omega_filter_ = std::make_unique<Kalman<OMEGA_DIM_MEAS, OMEGA_DIM_STATE>>(
        omega_q, 0.1, 2);
    
    initialized_ = false;
    update_count_ = 0;
    current_armor_ = 0;
    
    // 重置高度滤波器
    if (params_.enable_height_filter) {
        for (int i = 0; i < 2; ++i) {
            weighted_z_[i] = SlideWeightedAvg<double>(20);
        }
    }
}

// 设置主滤波器过程噪声
void RotatingTargetTracker::set_main_process_noise(
    const Eigen::Matrix<double, 9, 9>& Q) {
    Q_main_ = Q;
}

// 设置主滤波器测量噪声
void RotatingTargetTracker::set_main_measurement_noise(
    const Eigen::Matrix<double, 4, 4>& R) {
    R_main_ = R;
}

// 角度归一化
double RotatingTargetTracker::normalize_angle(double angle) const {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

// 计算角度差
double RotatingTargetTracker::angle_diff(double a1, double a2) const {
    double diff = a1 - a2;
    return normalize_angle(diff);
}

// 检测装甲板切换
int RotatingTargetTracker::detect_armor_switch(double new_angle, double old_angle) const {
    double diff = std::abs(angle_diff(new_angle, old_angle));
    
    if (params_.armor_count == 2) {
        // 两块装甲板，相差180度
        return (diff > M_PI / 2) ? 1 - current_armor_ : current_armor_;
    } else {
        // 四块装甲板，相差90度
        if (diff > M_PI / 4) {
            // 发生切换，计算切换到哪个装甲板
            int switches = std::round(diff / (M_PI / 2));
            return (current_armor_ + switches) % params_.armor_count;
        }
        return current_armor_;
    }
}

// 计算切换权重
double RotatingTargetTracker::calc_switch_weight(double theta) const {
    // 角度越接近正面，权重越大
    double normalized_theta = fmod(theta, M_PI);
    if (normalized_theta < 0) normalized_theta += M_PI;
    
    return 0.5 + 0.5 * cos(2 * normalized_theta);
}

// 估计初始状态
bool RotatingTargetTracker::estimate_initial_state(
    const std::deque<Observation>& observations) {
    
    if (observations.size() < 3) {
        return false;
    }
    
    const auto& last_obs = observations.back();
    
    // 初始化主EKF状态
    MainState init_state;
    init_state << last_obs.x, last_obs.y, last_obs.z, last_obs.theta, 
                  0, 0, 0, 0, (params_.radius_min + params_.radius_max) / 2.0;
    
    // 如果有足够的观测，估计速度和角速度
    if (observations.size() >= 3) {
        size_t n = observations.size();
        double dt1 = observations[n-1].timestamp - observations[n-2].timestamp;
        double dt2 = observations[n-2].timestamp - observations[n-3].timestamp;
        
        if (dt1 > 0 && dt2 > 0) {
            // 估计速度
            double vx = (observations[n-1].x - observations[n-2].x) / dt1;
            double vy = (observations[n-1].y - observations[n-2].y) / dt1;
            double vz = (observations[n-1].z - observations[n-2].z) / dt1;
            
            init_state(4) = vx;
            init_state(5) = vy;
            init_state(6) = vz;
            
            // 估计角速度
            double omega = angle_diff(observations[n-1].theta, observations[n-2].theta) / dt1;
            init_state(7) = omega;
        }
    }
    
    main_ekf_->init_x(init_state);
    
    // 设置初始协方差
    Eigen::Matrix<double, MAIN_DIM_STATE, MAIN_DIM_STATE> P0;
    P0 = Eigen::Matrix<double, MAIN_DIM_STATE, MAIN_DIM_STATE>::Identity();
    P0.block<3,3>(0,0) *= 0.01;  // 位置
    P0(3,3) = 0.01;              // 角度
    P0.block<3,3>(4,4) *= 0.1;   // 速度
    P0(7,7) = 0.1;               // 角速度
    P0(8,8) = 0.01;              // 半径
    main_ekf_->set_P(P0);
    
    // 初始化中心滤波器
    double center_x = last_obs.x - init_state(8) * cos(last_obs.theta);
    double center_y = last_obs.y - init_state(8) * sin(last_obs.theta);
    
    CenterState center_state;
    center_state << center_x, center_y, init_state(4), init_state(5);
    center_filter_->set_x(center_state);
    center_filter_->set_t(last_obs.timestamp);
    
    // 初始化角速度滤波器
    OmegaState omega_state;
    omega_state << last_obs.theta, init_state(7), 0;
    omega_filter_->set_x(omega_state);
    omega_filter_->set_t(last_obs.timestamp);
    
    // 初始化高度
    armor_z_[0] = armor_z_[1] = last_obs.z;
    
    return true;
}

// 更新装甲板参数
void RotatingTargetTracker::update_armor_params(const Observation& obs) {
    // 更新高度
    if (params_.enable_height_filter) {
        double weight = calc_switch_weight(obs.theta);
        weighted_z_[current_armor_].push(obs.z, weight);
        armor_z_[current_armor_] = weighted_z_[current_armor_].getAvg();
    } else {
        armor_z_[current_armor_] = obs.z;
    }
    
    // 更新半径（限制在合理范围内）
    double current_radius = main_ekf_->get_x()(8);
    armor_r_[current_armor_] = std::max(params_.radius_min, 
                                       std::min(params_.radius_max, current_radius));
}

} // namespace filter_lib