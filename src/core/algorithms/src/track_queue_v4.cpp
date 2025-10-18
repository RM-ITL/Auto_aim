/**
 * @file track_queue_v4.cpp
 * @brief 简化版极坐标系恒定加速度跟踪滤波器实现
 */

#include "filter/track_queue_v4.hpp"
#include <cmath>
#include <algorithm>

namespace filter_lib {

// 状态转移函数实现
template<typename T>
void TrackQueueV4::TransitionModel::operator()(
    const T x_prev[DIM_STATE], T x_curr[DIM_STATE]) const {
    
    // 提取前一时刻状态
    T x = x_prev[0];         // x位置
    T y = x_prev[1];         // y位置  
    T z = x_prev[2];         // z位置
    T v = x_prev[3];         // 水平速度大小
    T vz = x_prev[4];        // 垂直速度
    T angle = x_prev[5];     // 运动方向角
    T w = x_prev[6];         // 角速度
    T a = x_prev[7];         // 切向加速度
    
    // 恒定加速度运动模型更新
    T new_angle = angle + w * dt;
    T new_v = v + a * dt;
    
    // 使用平均速度和角度提高位置预测精度
    T avg_v = (v + new_v) / T(2.0);
    T avg_angle = (angle + new_angle) / T(2.0);
    
    // 更新状态
    x_curr[0] = x + avg_v * ceres::cos(avg_angle) * dt;
    x_curr[1] = y + avg_v * ceres::sin(avg_angle) * dt;
    x_curr[2] = z + vz * dt;
    x_curr[3] = new_v;
    x_curr[4] = vz;
    x_curr[5] = new_angle;
    x_curr[6] = w;
    x_curr[7] = a;
}

// 构造函数
TrackQueueV4::TrackQueueV4() {
    // 使用默认参数值，保持简洁
    set_process_noise(0.01, 0.1, 1.0);
    set_measurement_noise(0.05);
}

// 初始化函数
bool TrackQueueV4::init(const std::deque<Observation>& observations) {
    // 至少需要3个点来可靠估计运动参数
    if (observations.size() < 3) {
        return false;
    }
    
    State initial_state;
    if (!estimate_initial_state(observations, initial_state)) {
        return false;
    }
    
    ekf_.init_x(initial_state);
    
    // 设置初始协方差矩阵
    StateCov P0 = StateCov::Identity();
    P0.block<3,3>(0,0) *= 0.01;  // 位置不确定性较小
    P0.block<2,2>(3,3) *= 0.1;   // 速度不确定性
    P0.block<3,3>(5,5) *= 0.1;   // 角度相关不确定性
    ekf_.set_P(P0);
    
    initialized_ = true;
    update_count_ = 0;
    
    return true;
}

// 更新函数
void TrackQueueV4::update(const Meas& meas, double dt) {
    if (!initialized_ || dt <= 0) {
        return;
    }
    
    // 创建模型对象
    TransitionModel transition(dt);
    MeasurementModel measurement;
    
    // 构建噪声矩阵
    StateCov Q = build_process_noise(dt);
    MeasCov R = MeasCov::Identity() * (r_pos_ * r_pos_);
    
    // 执行EKF更新
    ekf_.update(measurement, transition, meas, Q, R);
    
    update_count_++;
}

// 预测函数
TrackQueueV4::State TrackQueueV4::predict(double dt) const {
    if (!initialized_) {
        return State::Zero();
    }
    
    TransitionModel transition(dt);
    auto pred_result = ekf_.predict(transition);
    
    // 归一化角度
    State result = pred_result.x_p;
    result(5) = math::reduced_angle(result(5));
    
    return result;
}

// 重置函数
void TrackQueueV4::reset() {
    ekf_ = AdaptiveEkf<DIM_STATE, DIM_MEAS>();
    initialized_ = false;
    update_count_ = 0;
}

// 设置过程噪声参数
void TrackQueueV4::set_process_noise(double q_pos, double q_vel, double q_acc) {
    q_pos_ = q_pos;
    q_vel_ = q_vel;
    q_acc_ = q_acc;
}

// 设置测量噪声参数
void TrackQueueV4::set_measurement_noise(double r_pos) {
    r_pos_ = r_pos;
}

// 估计初始状态
bool TrackQueueV4::estimate_initial_state(
    const std::deque<Observation>& observations, 
    State& state) {
    
    if (observations.size() < 3) {
        return false;
    }
    
    state.setZero();
    const auto& last_obs = observations.back();
    
    // 设置位置为最新观测
    state(0) = last_obs.x;
    state(1) = last_obs.y;
    state(2) = last_obs.z;
    
    // 使用最小二乘法估计速度，提高鲁棒性
    if (observations.size() >= 5) {
        double sum_vx = 0, sum_vy = 0, sum_vz = 0;
        int valid_pairs = 0;
        
        for (size_t i = 1; i < observations.size(); ++i) {
            double dt = observations[i].timestamp - observations[i-1].timestamp;
            if (dt > 0.01 && dt < 1.0) {  // 合理的时间间隔
                sum_vx += (observations[i].x - observations[i-1].x) / dt;
                sum_vy += (observations[i].y - observations[i-1].y) / dt;
                sum_vz += (observations[i].z - observations[i-1].z) / dt;
                valid_pairs++;
            }
        }
        
        if (valid_pairs > 0) {
            double avg_vx = sum_vx / valid_pairs;
            double avg_vy = sum_vy / valid_pairs;
            double avg_vz = sum_vz / valid_pairs;
            
            // 计算速度大小和方向
            state(3) = std::sqrt(avg_vx * avg_vx + avg_vy * avg_vy);
            state(4) = avg_vz;
            state(5) = std::atan2(avg_vy, avg_vx);
        }
    } else {
        // 观测点较少时，使用最后两个点估计
        size_t n = observations.size();
        double dt = observations[n-1].timestamp - observations[n-2].timestamp;
        
        if (dt > 0) {
            double vx = (observations[n-1].x - observations[n-2].x) / dt;
            double vy = (observations[n-1].y - observations[n-2].y) / dt;
            double vz = (observations[n-1].z - observations[n-2].z) / dt;
            
            state(3) = std::sqrt(vx * vx + vy * vy);
            state(4) = vz;
            state(5) = std::atan2(vy, vx);
        }
    }
    
    // 初始角速度和加速度设为0，让滤波器逐渐估计
    state(6) = 0.0;  // omega
    state(7) = 0.0;  // a
    
    return true;
}

// 构建过程噪声矩阵
TrackQueueV4::StateCov TrackQueueV4::build_process_noise(double dt) const {
    StateCov Q = StateCov::Zero();
    
    // 基于物理模型的噪声设计
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    
    // 位置噪声（受速度和加速度影响）
    Q(0, 0) = q_pos_ * q_pos_ * dt2;
    Q(1, 1) = q_pos_ * q_pos_ * dt2;
    Q(2, 2) = q_pos_ * q_pos_ * dt2;
    
    // 速度噪声
    Q(3, 3) = q_vel_ * q_vel_ * dt;
    Q(4, 4) = q_vel_ * q_vel_ * dt;
    
    // 角度和角速度噪声
    Q(5, 5) = q_vel_ * q_vel_ * dt3 / 3.0;  // 角度受角速度积分影响
    Q(6, 6) = q_vel_ * q_vel_ * dt;
    
    // 加速度噪声
    Q(7, 7) = q_acc_ * q_acc_ * dt;
    
    return Q;
}

} // namespace filter_lib