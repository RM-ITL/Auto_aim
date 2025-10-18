/**
 * @file track_queue_v3.cpp
 * @brief 4D跟踪滤波器实现 - 位置和朝向角联合估计
 * 
 * 物理模型基础：
 * ===============
 * 
 * 一、状态空间模型（11维）
 * 状态向量：x = [px, py, pz, θ, vx, vy, vz, ω, ax, ay, α]ᵀ
 * 
 * 二、运动学方程
 * 平移运动（恒定加速度模型）：
 *   p(t+Δt) = p(t) + v(t)·Δt + 0.5·a(t)·Δt²
 *   v(t+Δt) = v(t) + a(t)·Δt  
 *   a(t+Δt) = a(t) + w_a(t)
 * 
 * 旋转运动（恒定角加速度模型）：
 *   θ(t+Δt) = θ(t) + ω(t)·Δt + 0.5·α(t)·Δt²
 *   ω(t+Δt) = ω(t) + α(t)·Δt
 *   α(t+Δt) = α(t) + w_α(t)
 * 
 * 三、噪声模型
 * 采用连续白噪声的精确离散化，考虑噪声通过积分对状态的影响：
 * - 加速度噪声对位置的影响：经过双重积分，方差 ∝ Δt⁵/20
 * - 加速度噪声对速度的影响：经过单次积分，方差 ∝ Δt³/3
 * - 加速度噪声直接影响：方差 ∝ Δt
 */

#include "filter/track_queue_v3.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace filter_lib {

// 状态转移模型实现
template<typename T>
void TrackQueueV3::TransitionModel::operator()(
    const T x_prev[DIM_STATE], T x_curr[DIM_STATE]) const {
    
    // 位置更新（恒加速运动）
    x_curr[0] = x_prev[0] + x_prev[4] * dt + T(0.5) * x_prev[8] * dt * dt;
    x_curr[1] = x_prev[1] + x_prev[5] * dt + T(0.5) * x_prev[9] * dt * dt;
    x_curr[2] = x_prev[2] + x_prev[6] * dt;  // z方向简化为匀速
    
    // 角度更新（恒角加速运动）
    T theta_new = x_prev[3] + x_prev[7] * dt + T(0.5) * x_prev[10] * dt * dt;
    // 角度归一化到[-π, π]
    while (theta_new > T(M_PI)) theta_new -= T(2*M_PI);
    while (theta_new < T(-M_PI)) theta_new += T(2*M_PI);
    x_curr[3] = theta_new;
    
    // 速度更新
    x_curr[4] = x_prev[4] + x_prev[8] * dt;
    x_curr[5] = x_prev[5] + x_prev[9] * dt;
    x_curr[6] = x_prev[6];  // vz保持恒定
    
    // 角速度更新
    x_curr[7] = x_prev[7] + x_prev[10] * dt;
    
    // 加速度保持恒定（随机游走模型）
    x_curr[8] = x_prev[8];
    x_curr[9] = x_prev[9];
    x_curr[10] = x_prev[10];
}

// 无参构造函数
TrackQueueV3::TrackQueueV3() 
    : TrackQueueV3(Parameters()) {
}

// 带参数构造函数
TrackQueueV3::TrackQueueV3(const Parameters& params) 
    : params_(params) {
    ekf_ = AdaptiveEkf<DIM_STATE, DIM_MEAS>();
}

// 初始化函数
bool TrackQueueV3::init(const std::deque<Observation>& observations) {
    if (observations.size() < 2) {
        std::cerr << "[TrackQueueV3] Need at least 2 observations for initialization" << std::endl;
        return false;
    }
    
    State initial_state;
    if (!estimateInitialState(observations, initial_state)) {
        return false;
    }
    
    ekf_.init_x(initial_state);
    
    // 设置初始协方差矩阵
    StateCov P0 = StateCov::Identity();
    P0.block<3,3>(0,0) *= params_.init_pos_cov;
    P0(3,3) = params_.init_angle_cov;
    P0.block<3,3>(4,4) *= params_.init_vel_cov;
    P0(7,7) = params_.init_omega_cov;
    P0.block<2,2>(8,8) *= params_.init_acc_cov;
    P0(10,10) = params_.init_alpha_cov;
    
    ekf_.set_P(P0);
    
    initialized_ = true;
    update_count_ = 0;
    
    // 记录最后的时间戳
    double last_time = observations.back().timestamp;
    last_update_time_ = TimePoint(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(last_time)));
    
    updateTrackingInfo();
    
    std::cout << "[TrackQueueV3] Successfully initialized with " << observations.size() << " observations" << std::endl;
    std::cout << "  Initial speed: " << tracking_info_.speed << " m/s" << std::endl;
    std::cout << "  Initial heading: " << initial_state(3) * 180/M_PI << " degrees" << std::endl;
    
    return true;
}

// 更新函数（时间间隔版本）
void TrackQueueV3::update(const Meas& meas, double dt) {
    if (!initialized_ || dt <= 0) {
        std::cerr << "[TrackQueueV3] Invalid update: initialized=" << initialized_ 
                  << ", dt=" << dt << std::endl;
        return;
    }
    
    // 大时间间隔警告
    if (dt > 1.0) {
        std::cout << "[TrackQueueV3] Large time gap (" << dt 
                  << "s), prediction uncertainty increased" << std::endl;
    }
    
    // 构建模型和噪声矩阵
    TransitionModel transition(dt);
    MeasurementModel measurement;
    
    StateCov Q = buildProcessNoise(dt);
    MeasCov R = buildMeasurementNoise();
    
    // 执行EKF更新
    ekf_.update(measurement, transition, meas, Q, R);
    
    // 角度归一化
    State x = ekf_.get_x();
    x(3) = normalizeAngle(x(3));
    ekf_.set_x(x);
    
    update_count_++;
    updateTrackingInfo();
}

// 更新函数（时间戳版本）
void TrackQueueV3::update(const Meas& meas, TimePoint timestamp) {
    if (!initialized_) {
        std::cerr << "[TrackQueueV3] Not initialized!" << std::endl;
        return;
    }
    
    double dt = std::chrono::duration<double>(timestamp - last_update_time_).count();
    update(meas, dt);
    last_update_time_ = timestamp;
}

// 预测函数
TrackQueueV3::State TrackQueueV3::predict(double dt) const {
    if (!initialized_) {
        return State::Zero();
    }
    
    TransitionModel transition(dt);
    auto pred_result = ekf_.predict(transition);
    
    State result = pred_result.x_p;
    result(3) = normalizeAngle(result(3));
    
    return result;
}

// 重置滤波器
void TrackQueueV3::reset() {
    ekf_ = AdaptiveEkf<DIM_STATE, DIM_MEAS>();
    initialized_ = false;
    update_count_ = 0;
    tracking_info_ = TrackingInfo();
}

// 获取状态和协方差
std::pair<TrackQueueV3::State, TrackQueueV3::StateCov> 
TrackQueueV3::getStateWithCovariance() const {
    if (!initialized_) {
        return {State::Zero(), StateCov::Identity() * 1e6};
    }
    
    return {ekf_.get_x(), ekf_.get_P()};
}

// 获取跟踪信息
TrackQueueV3::TrackingInfo TrackQueueV3::getTrackingInfo() const {
    return tracking_info_;
}

// 更新跟踪信息
void TrackQueueV3::updateTrackingInfo() {
    if (!initialized_) {
        return;
    }
    
    State x = ekf_.get_x();
    
    // 计算速度大小
    Eigen::Vector3d vel(x(4), x(5), x(6));
    tracking_info_.speed = vel.norm();
    
    // 角速度和角加速度
    tracking_info_.heading_rate = x(7);
    tracking_info_.angular_acceleration = x(10);
    
    // 加速度
    tracking_info_.acceleration = Eigen::Vector3d(x(8), x(9), 0.0);
    
    // 观测计数
    tracking_info_.observation_count = update_count_;
    
    // 估计质量（基于协方差矩阵的迹）
    StateCov P = ekf_.get_P();
    double position_uncertainty = P.block<3,3>(0,0).trace();
    double velocity_uncertainty = P.block<3,3>(4,4).trace();
    
    // 质量评分：不确定性越小，质量越高
    tracking_info_.estimation_quality = 1.0 / (1.0 + position_uncertainty + 0.1 * velocity_uncertainty);
}

// 角度归一化
double TrackQueueV3::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

// 估计初始状态
bool TrackQueueV3::estimateInitialState(
    const std::deque<Observation>& observations,
    State& state) {
    
    if (observations.size() < 2) {
        return false;
    }
    
    state.setZero();
    size_t n = observations.size();
    
    if (n >= 3) {
        // 使用最后三个点估计完整状态
        const auto& p1 = observations[n - 3];
        const auto& p2 = observations[n - 2];
        const auto& p3 = observations[n - 1];
        
        // 位置和角度使用最新观测
        state(0) = p3.x;
        state(1) = p3.y;
        state(2) = p3.z;
        state(3) = p3.theta;
        
        double dt1 = p2.timestamp - p1.timestamp;
        double dt2 = p3.timestamp - p2.timestamp;
        
        if (dt1 > 0 && dt2 > 0) {
            // 速度估计（使用最新段）
            state(4) = (p3.x - p2.x) / dt2;
            state(5) = (p3.y - p2.y) / dt2;
            state(6) = (p3.z - p2.z) / dt2;
            
            // 使用两段速度估计加速度
            double vx1 = (p2.x - p1.x) / dt1;
            double vy1 = (p2.y - p1.y) / dt1;
            double vx2 = state(4);
            double vy2 = state(5);
            
            double dt_avg = (dt1 + dt2) / 2.0;
            state(8) = (vx2 - vx1) / dt_avg;
            state(9) = (vy2 - vy1) / dt_avg;
            
            // 角速度和角加速度估计
            double omega1 = normalizeAngle(p2.theta - p1.theta) / dt1;
            double omega2 = normalizeAngle(p3.theta - p2.theta) / dt2;
            
            state(7) = omega2;
            state(10) = (omega2 - omega1) / dt_avg;
        }
    } else {
        // 只有两个点，仅估计位置和速度
        const auto& p1 = observations[0];
        const auto& p2 = observations[1];
        
        state(0) = p2.x;
        state(1) = p2.y;
        state(2) = p2.z;
        state(3) = p2.theta;
        
        double dt = p2.timestamp - p1.timestamp;
        if (dt > 0) {
            state(4) = (p2.x - p1.x) / dt;
            state(5) = (p2.y - p1.y) / dt;
            state(6) = (p2.z - p1.z) / dt;
            state(7) = normalizeAngle(p2.theta - p1.theta) / dt;
        }
    }
    
    return true;
}

// 构建过程噪声矩阵
TrackQueueV3::StateCov TrackQueueV3::buildProcessNoise(double dt) const {
    StateCov Q = StateCov::Zero();
    
    // 基于连续白噪声积分的精确离散化
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    double dt5 = dt4 * dt;
    
    // X-Y平面运动（完整的加速度噪声模型）
    double q_acc = params_.q_acc;
    
    // 位置方差
    Q(0,0) = q_acc * q_acc * dt5 / 20.0;
    Q(1,1) = q_acc * q_acc * dt5 / 20.0;
    
    // 速度方差
    Q(4,4) = q_acc * q_acc * dt3 / 3.0;
    Q(5,5) = q_acc * q_acc * dt3 / 3.0;
    
    // 加速度方差
    Q(8,8) = q_acc * q_acc * dt;
    Q(9,9) = q_acc * q_acc * dt;
    
    // 位置-速度协方差
    Q(0,4) = q_acc * q_acc * dt4 / 8.0;
    Q(4,0) = Q(0,4);
    Q(1,5) = Q(0,4);
    Q(5,1) = Q(1,5);
    
    // 位置-加速度协方差
    Q(0,8) = q_acc * q_acc * dt3 / 6.0;
    Q(8,0) = Q(0,8);
    Q(1,9) = Q(0,8);
    Q(9,1) = Q(1,9);
    
    // 速度-加速度协方差
    Q(4,8) = q_acc * q_acc * dt2 / 2.0;
    Q(8,4) = Q(4,8);
    Q(5,9) = Q(4,8);
    Q(9,5) = Q(5,9);
    
    // Z方向（简化模型）
    Q(2,2) = params_.q_pos * params_.q_pos * dt2;
    Q(6,6) = params_.q_vel * params_.q_vel * dt;
    
    // 角度运动（类似于X-Y的处理）
    double q_alpha = params_.q_alpha;
    
    Q(3,3) = q_alpha * q_alpha * dt5 / 20.0;  // 角度方差
    Q(7,7) = q_alpha * q_alpha * dt3 / 3.0;   // 角速度方差
    Q(10,10) = q_alpha * q_alpha * dt;        // 角加速度方差
    
    // 角度相关协方差
    Q(3,7) = q_alpha * q_alpha * dt4 / 8.0;
    Q(7,3) = Q(3,7);
    Q(3,10) = q_alpha * q_alpha * dt3 / 6.0;
    Q(10,3) = Q(3,10);
    Q(7,10) = q_alpha * q_alpha * dt2 / 2.0;
    Q(10,7) = Q(7,10);
    
    return Q;
}

// 构建测量噪声矩阵
TrackQueueV3::MeasCov TrackQueueV3::buildMeasurementNoise() const {
    MeasCov R = MeasCov::Zero();
    
    // 位置测量噪声
    R.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * (params_.r_pos * params_.r_pos);
    
    // 角度测量噪声
    R(3,3) = params_.r_angle * params_.r_angle;
    
    return R;
}

} // namespace filter_lib