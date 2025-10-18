/**
 * @file decoupled_tracker.cpp
 * @brief 基于EKF的解耦式跟踪滤波器实现
 * 
 * 详细物理模型推导：
 * 
 * 一、平移运动模型（匀加速运动）
 * ================================
 * 
 * 连续时间状态方程：
 * d/dt [x]   [0 0 0 1 0 0 0 0 0]   [x]   [0]
 * d/dt [y]   [0 0 0 0 1 0 0 0 0]   [y]   [0]
 * d/dt [z] = [0 0 0 0 0 1 0 0 0] * [z] + [0]
 * d/dt [vx]  [0 0 0 0 0 0 1 0 0]   [vx]  [0]
 * d/dt [vy]  [0 0 0 0 0 0 0 1 0]   [vy]  [0]
 * d/dt [vz]  [0 0 0 0 0 0 0 0 1]   [vz]  [0]
 * d/dt [ax]  [0 0 0 0 0 0 0 0 0]   [ax]  [wx]
 * d/dt [ay]  [0 0 0 0 0 0 0 0 0]   [ay]  [wy]
 * d/dt [az]  [0 0 0 0 0 0 0 0 0]   [az]  [wz]
 * 
 * 其中 wx, wy, wz 是加速度的过程噪声（白噪声）
 * 
 * 离散化（使用精确离散化）：
 * x(k+1) = F(dt) * x(k) + w(k)
 * 
 * 状态转移矩阵 F(dt) = exp(A*dt)，对于这个特殊的A矩阵：
 * F(dt) = [1 0 0 dt 0  0  dt²/2 0     0    ]
 *         [0 1 0 0  dt 0  0     dt²/2 0    ]
 *         [0 0 1 0  0  dt 0     0     dt²/2]
 *         [0 0 0 1  0  0  dt    0     0    ]
 *         [0 0 0 0  1  0  0     dt    0    ]
 *         [0 0 0 0  0  1  0     0     dt   ]
 *         [0 0 0 0  0  0  1     0     0    ]
 *         [0 0 0 0  0  0  0     1     0    ]
 *         [0 0 0 0  0  0  0     0     1    ]
 * 
 * 二、姿态运动模型
 * ================
 * 
 * 状态方程：
 * θ(k+1) = θ(k) + ω(k)*dt + 0.5*α(k)*dt²
 * ω(k+1) = ω(k) + α(k)*dt
 * α(k+1) = α(k) + wα(k)
 * 
 * 注意：由于角度的周期性，这是一个非线性系统！
 * 
 * 三、耦合机制的数学基础
 * =====================
 * 
 * 核心观察：当目标做正常前向运动时，速度方向应该与朝向大致一致。
 * 
 * 定义虚拟观测：
 * h(x_trans, x_pose) = atan2(vy, vx) - θ
 * 
 * 在理想的前向运动中，h ≈ 0
 * 
 * 耦合强度的计算基于速度向量与朝向向量的一致性：
 * ρ = exp(-|v × e_θ|² / σ²)
 * 
 * 其中：
 * - v = [vx, vy, 0] 是速度向量
 * - e_θ = [cos(θ), sin(θ), 0] 是朝向单位向量
 * - |v × e_θ| = |v| * sin(angle_between_v_and_θ)
 * 
 * 当 v 与 e_θ 平行时，叉积为0，ρ→1（强耦合）
 * 当 v 与 e_θ 垂直时，叉积最大，ρ→0（弱耦合）
 */

#include "filter/decoupled_tracker.hpp"
#include <iostream>
#include <algorithm>

namespace filter_lib {

// 添加无参构造函数实现
DecoupledTracker::DecoupledTracker() 
    : DecoupledTracker(Parameters()) {
}

// 带参数的构造函数
DecoupledTracker::DecoupledTracker(const Parameters& params) 
    : params_(params) {
    translation_filter_ = std::make_unique<AdaptiveEkf<TRANS_DIM, TRANS_OBS>>();
    pose_filter_ = std::make_unique<AdaptiveEkf<POSE_DIM, POSE_OBS>>();
}



// 构建平移系统的过程噪声协方差矩阵
Eigen::Matrix<double, DecoupledTracker::TRANS_DIM, DecoupledTracker::TRANS_DIM> 
DecoupledTracker::buildTransQ(double dt) const {
    // 基于连续白噪声的离散化
    // 对于加速度的白噪声，其对位置、速度和加速度的影响不同
    
    Eigen::Matrix<double, TRANS_DIM, TRANS_DIM> Q = 
        Eigen::Matrix<double, TRANS_DIM, TRANS_DIM>::Zero();
    
    // 由于加速度噪声的积分效应：
    // - 对位置的影响：∫∫w(τ)dτdτ → 方差 ∝ q²*dt⁵/20
    // - 对速度的影响：∫w(τ)dτ → 方差 ∝ q²*dt³/3
    // - 对加速度的影响：w(t) → 方差 ∝ q²*dt
    
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    double dt5 = dt4 * dt;
    
    // X方向
    double qx = params_.trans_q_acc;
    Q(0,0) = qx*qx * dt5/20.0;  // 位置方差
    Q(0,3) = qx*qx * dt4/8.0;   // 位置-速度协方差
    Q(0,6) = qx*qx * dt3/6.0;   // 位置-加速度协方差
    Q(3,0) = Q(0,3);            // 对称
    Q(3,3) = qx*qx * dt3/3.0;   // 速度方差
    Q(3,6) = qx*qx * dt2/2.0;   // 速度-加速度协方差
    Q(6,0) = Q(0,6);            // 对称
    Q(6,3) = Q(3,6);            // 对称
    Q(6,6) = qx*qx * dt;        // 加速度方差
    
    // Y和Z方向类似
    double qy = params_.trans_q_acc;
    Q(1,1) = qy*qy * dt5/20.0;
    Q(1,4) = qy*qy * dt4/8.0;
    Q(1,7) = qy*qy * dt3/6.0;
    Q(4,1) = Q(1,4);
    Q(4,4) = qy*qy * dt3/3.0;
    Q(4,7) = qy*qy * dt2/2.0;
    Q(7,1) = Q(1,7);
    Q(7,4) = Q(4,7);
    Q(7,7) = qy*qy * dt;
    
    double qz = params_.trans_q_acc;
    Q(2,2) = qz*qz * dt5/20.0;
    Q(2,5) = qz*qz * dt4/8.0;
    Q(2,8) = qz*qz * dt3/6.0;
    Q(5,2) = Q(2,5);
    Q(5,5) = qz*qz * dt3/3.0;
    Q(5,8) = qz*qz * dt2/2.0;
    Q(8,2) = Q(2,8);
    Q(8,5) = Q(5,8);
    Q(8,8) = qz*qz * dt;
    
    return Q;
}

// 构建姿态系统的过程噪声协方差矩阵
Eigen::Matrix<double, DecoupledTracker::POSE_DIM, DecoupledTracker::POSE_DIM> 
DecoupledTracker::buildPoseQ(double dt) const {
    Eigen::Matrix<double, POSE_DIM, POSE_DIM> Q = 
        Eigen::Matrix<double, POSE_DIM, POSE_DIM>::Zero();
    
    double dt2 = dt * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;
    double dt5 = dt4 * dt;
    
    double q_alpha = params_.pose_q_alpha;
    
    // 类似于平移系统
    Q(0,0) = q_alpha*q_alpha * dt5/20.0;  // θ方差
    Q(0,1) = q_alpha*q_alpha * dt4/8.0;   // θ-ω协方差
    Q(0,2) = q_alpha*q_alpha * dt3/6.0;   // θ-α协方差
    Q(1,0) = Q(0,1);
    Q(1,1) = q_alpha*q_alpha * dt3/3.0;   // ω方差
    Q(1,2) = q_alpha*q_alpha * dt2/2.0;   // ω-α协方差
    Q(2,0) = Q(0,2);
    Q(2,1) = Q(1,2);
    Q(2,2) = q_alpha*q_alpha * dt;        // α方差
    
    return Q;
}

// 初始化函数
bool DecoupledTracker::init(const std::deque<Eigen::Matrix<double, 5, 1>>& poses) {
    if (poses.size() < 3) {
        std::cerr << "[DecoupledTracker] Need at least 3 poses for robust initialization" << std::endl;
        return false;
    }
    
    if (!estimateInitialStates(poses)) {
        return false;
    }
    
    initialized_ = true;
    update_count_ = poses.size();
    
    // 记录最后的时间戳
    double last_time = poses.back()(4);
    last_update_time_ = TimePoint(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(last_time)));
    
    std::cout << "[DecoupledTracker] Successfully initialized with " << poses.size() << " poses" << std::endl;
    std::cout << "  Initial speed: " << coupling_info_.speed << " m/s" << std::endl;
    std::cout << "  Initial heading: " << poses.back()(3) * 180/M_PI << " degrees" << std::endl;
    
    return true;
}

// 更新函数
void DecoupledTracker::update(const MeasVec& measurement, TimePoint timestamp) {
    if (!initialized_) {
        std::cerr << "[DecoupledTracker] Not initialized!" << std::endl;
        return;
    }
    
    // 计算时间间隔
    double dt = std::chrono::duration<double>(timestamp - last_update_time_).count();
    
    if (dt <= 0) {
        std::cerr << "[DecoupledTracker] Invalid time interval: " << dt << std::endl;
        return;
    }
    
    if (dt > 1.0) {
        std::cout << "[DecoupledTracker] Large time gap (" << dt 
                  << "s), reducing coupling strength" << std::endl;
        coupling_info_.rho *= 0.5;  // 衰减耦合强度
    }
    
    // 步骤1：分别更新两个EKF
    // 平移系统更新
    TransObsVec trans_obs;
    trans_obs << measurement(0), measurement(1), measurement(2);
    
    TransitionModelTrans trans_model(dt);
    MeasurementModelTrans meas_model_trans;
    
    Eigen::Matrix<double, TRANS_OBS, TRANS_OBS> R_trans = 
        Eigen::Matrix<double, TRANS_OBS, TRANS_OBS>::Identity() * 
        (params_.trans_r * params_.trans_r);
    
    translation_filter_->update(
        meas_model_trans, 
        trans_model, 
        trans_obs, 
        buildTransQ(dt), 
        R_trans
    );
    
    // 姿态系统更新
    PoseObsVec pose_obs;
    pose_obs << measurement(3);
    
    TransitionModelPose pose_model(dt);
    MeasurementModelPose meas_model_pose;
    
    Eigen::Matrix<double, POSE_OBS, POSE_OBS> R_pose;
    R_pose(0,0) = params_.pose_r * params_.pose_r;
    
    pose_filter_->update(
        meas_model_pose,
        pose_model,
        pose_obs,
        buildPoseQ(dt),
        R_pose
    );
    
    // 步骤2：更新耦合强度
    updateCouplingStrength();
    
    // 步骤3：如果耦合强度足够高，应用耦合校正
    if (coupling_info_.rho > params_.rho_threshold && coupling_info_.speed > params_.min_speed) {
        applyCouplingCorrection(dt);
    }
    
    // 更新状态
    last_update_time_ = timestamp;
    update_count_++;
    
    // 维护历史记录
    TransStateVec trans_state = translation_filter_->get_x();
    Eigen::Vector3d velocity(trans_state(3), trans_state(4), trans_state(5));
    velocity_history_.push_back(velocity);
    
    PoseStateVec pose_state = pose_filter_->get_x();
    heading_history_.push_back(pose_state(0));
    omega_history_.push_back(pose_state(1));
    
    // 限制历史记录大小
    while (velocity_history_.size() > HISTORY_SIZE) {
        velocity_history_.pop_front();
    }
    while (heading_history_.size() > HISTORY_SIZE) {
        heading_history_.pop_front();
    }
    while (omega_history_.size() > HISTORY_SIZE) {
        omega_history_.pop_front();
    }
}

// 预测函数
DecoupledTracker::MeasVec DecoupledTracker::predict(double dt) const {
    if (!initialized_) {
        return MeasVec::Zero();
    }
    
    // 使用EKF的预测功能
    TransitionModelTrans trans_model(dt);
    auto trans_pred = translation_filter_->predict(trans_model);
    
    TransitionModelPose pose_model(dt);
    auto pose_pred = pose_filter_->predict(pose_model);
    
    // 组合结果
    MeasVec result;
    result << trans_pred.x_p(0), trans_pred.x_p(1), trans_pred.x_p(2),  // x,y,z
              normalizeAngle(pose_pred.x_p(0));                          // theta
    
    return result;
}

// 更新耦合强度
void DecoupledTracker::updateCouplingStrength() {
    if (velocity_history_.size() < 3 || heading_history_.size() < 3) {
        coupling_info_.rho = params_.min_rho;
        coupling_info_.mode_description = "Insufficient data";
        return;
    }
    
    // 获取当前状态
    TransStateVec trans_state = translation_filter_->get_x();
    PoseStateVec pose_state = pose_filter_->get_x();
    
    // 计算当前速度
    Eigen::Vector3d velocity(trans_state(3), trans_state(4), trans_state(5));
    double speed = velocity.norm();
    coupling_info_.speed = speed;
    
    // 如果速度太小，降低耦合
    if (speed < params_.min_speed) {
        coupling_info_.rho = params_.min_rho;
        coupling_info_.mode_description = "Near stationary (v=" + 
            std::to_string(speed) + " m/s)";
        return;
    }
    
    // 计算速度方向
    coupling_info_.velocity_dir = velocity / speed;
    coupling_info_.velocity_angle = atan2(velocity(1), velocity(0));
    
    // 计算朝向方向
    double theta = pose_state(0);
    coupling_info_.heading_dir << cos(theta), sin(theta);
    coupling_info_.heading_rate = pose_state(1);
    
    // 计算速度方向与朝向的差值
    coupling_info_.angle_diff = angleDiff(coupling_info_.velocity_angle, theta);
    
    // 计算一致性（点积）
    Eigen::Vector2d velocity_xy = velocity.head<2>().normalized();
    coupling_info_.consistency = velocity_xy.dot(coupling_info_.heading_dir);
    
    // 计算叉积（用于耦合强度）
    Eigen::Vector3d velocity_3d = velocity;
    Eigen::Vector3d heading_3d(cos(theta) * speed, sin(theta) * speed, 0);
    Eigen::Vector3d cross = velocity_3d.cross(heading_3d);
    double cross_norm = cross.norm() / (speed * speed);  // 归一化
    
    // 计算耦合强度
    double raw_rho = exp(-cross_norm * cross_norm / 
        (params_.coupling_sigma * params_.coupling_sigma));
    
    // 基于角速度调整耦合强度
    // 如果目标在快速旋转，降低耦合
    double omega_factor = 1.0 / (1.0 + std::abs(pose_state(1)) * 2.0);
    raw_rho *= omega_factor;
    
    // 基于历史一致性调整
    // 如果历史速度方向变化很大，说明可能在做复杂机动
    if (velocity_history_.size() >= 5) {
        double vel_variance = 0;
        Eigen::Vector3d mean_vel = Eigen::Vector3d::Zero();
        
        for (size_t i = velocity_history_.size() - 5; i < velocity_history_.size(); ++i) {
            mean_vel += velocity_history_[i];
        }
        mean_vel /= 5.0;
        
        for (size_t i = velocity_history_.size() - 5; i < velocity_history_.size(); ++i) {
            Eigen::Vector3d diff = velocity_history_[i] - mean_vel;
            vel_variance += diff.squaredNorm();
        }
        vel_variance /= 5.0;
        
        // 高方差意味着不稳定的运动
        double stability_factor = 1.0 / (1.0 + vel_variance * 0.1);
        raw_rho *= stability_factor;
    }
    
    // 限制范围
    coupling_info_.rho = std::max(params_.min_rho, std::min(params_.max_rho, raw_rho));
    
    // 更新运动模式描述
    if (coupling_info_.rho > 0.7) {
        coupling_info_.mode_description = "Forward motion (ρ=" + 
            std::to_string(coupling_info_.rho) + ")";
    } else if (coupling_info_.rho < 0.3) {
        coupling_info_.mode_description = "Decoupled motion (ρ=" + 
            std::to_string(coupling_info_.rho) + ", Δθ=" + 
            std::to_string(coupling_info_.angle_diff * 180/M_PI) + "°)";
    } else {
        coupling_info_.mode_description = "Mixed motion (ρ=" + 
            std::to_string(coupling_info_.rho) + ")";
    }
}

void DecoupledTracker::applyCouplingCorrection(double dt) {
    // 获取当前状态和协方差
    TransStateVec trans_state = translation_filter_->get_x();
    auto P_trans = translation_filter_->get_P();
    PoseStateVec pose_state = pose_filter_->get_x();
    
    // 检查速度
    double vx = trans_state(3);
    double vy = trans_state(4);
    double speed = std::sqrt(vx*vx + vy*vy);
    
    if (speed < params_.min_speed) {
        return;
    }
    
    // 计算角度差
    double velocity_angle = std::atan2(vy, vx);
    double theta = pose_state(0);
    double angle_diff = normalizeAngle(velocity_angle - theta);
    
    // 如果角度差很小，不需要校正
    if (std::abs(angle_diff) < 0.1) {  // 约6度
        return;
    }
    
    // 基于耦合强度计算校正量
    double correction_strength = coupling_info_.rho * 0.1;  // 限制校正强度
    
    // 直接修改状态（软约束方式）
    double corrected_angle = velocity_angle - correction_strength * angle_diff;
    
    // 更新速度，保持大小不变
    trans_state(3) = speed * std::cos(corrected_angle);
    trans_state(4) = speed * std::sin(corrected_angle);
    
    // 增加速度的不确定性（因为我们进行了人为调整）
    P_trans(3, 3) *= (1.0 + 0.1 * (1.0 - coupling_info_.rho));
    P_trans(4, 4) *= (1.0 + 0.1 * (1.0 - coupling_info_.rho));
    
    // 更新滤波器状态
    translation_filter_->set_x(trans_state);
    translation_filter_->set_P(P_trans);
    
    // 记录校正
    coupling_info_.coupling_correction = correction_strength;
}

// 获取状态及协方差
std::pair<DecoupledTracker::TransStateVec, Eigen::Matrix<double, DecoupledTracker::TRANS_DIM, DecoupledTracker::TRANS_DIM>> 
DecoupledTracker::getTransStateWithCovariance() const {
    if (!initialized_) {
        return {TransStateVec::Zero(), 
                Eigen::Matrix<double, TRANS_DIM, TRANS_DIM>::Identity() * 1e6};
    }
    
    return {translation_filter_->get_x(), translation_filter_->get_P()};
}

std::pair<DecoupledTracker::PoseStateVec, Eigen::Matrix<double, DecoupledTracker::POSE_DIM, DecoupledTracker::POSE_DIM>> 
DecoupledTracker::getPoseStateWithCovariance() const {
    if (!initialized_) {
        return {PoseStateVec::Zero(), 
                Eigen::Matrix<double, POSE_DIM, POSE_DIM>::Identity() * 1e6};
    }
    
    return {pose_filter_->get_x(), pose_filter_->get_P()};
}

// 角度处理函数
double DecoupledTracker::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

double DecoupledTracker::angleDiff(double a1, double a2) {
    double diff = a1 - a2;
    return normalizeAngle(diff);
}

// 重置滤波器
void DecoupledTracker::reset() {
    translation_filter_ = std::make_unique<AdaptiveEkf<TRANS_DIM, TRANS_OBS>>();
    pose_filter_ = std::make_unique<AdaptiveEkf<POSE_DIM, POSE_OBS>>();
    
    initialized_ = false;
    update_count_ = 0;
    
    velocity_history_.clear();
    heading_history_.clear();
    omega_history_.clear();
    
    coupling_info_ = CouplingInfo();
}

// 从历史数据估计初始状态
bool DecoupledTracker::estimateInitialStates(
    const std::deque<Eigen::Matrix<double, 5, 1>>& poses) {
    
    if (poses.size() < 3) {
        return false;
    }
    
    // 使用最小二乘法估计速度和加速度
    // 对于位置 p(t) = p0 + v0*t + 0.5*a0*t²
    
    // 构建时间序列
    std::vector<double> times;
    std::vector<Eigen::Vector3d> positions;
    std::vector<double> thetas;
    
    double t0 = poses[0](4);
    for (const auto& pose : poses) {
        times.push_back(pose(4) - t0);
        positions.push_back(pose.head<3>());
        thetas.push_back(pose(3));
    }
    
    // 估计最后时刻的位置、速度和加速度
    size_t n = poses.size();
    Eigen::Vector3d final_pos = positions.back();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    
    if (n >= 2) {
        // 使用最后两个点估计速度
        double dt = times[n-1] - times[n-2];
        if (dt > 0) {
            velocity = (positions[n-1] - positions[n-2]) / dt;
        }
    }
    
    if (n >= 3) {
        // 使用最后三个点估计加速度
        double dt1 = times[n-1] - times[n-2];
        double dt2 = times[n-2] - times[n-3];
        
        if (dt1 > 0 && dt2 > 0) {
            Eigen::Vector3d v1 = (positions[n-1] - positions[n-2]) / dt1;
            Eigen::Vector3d v2 = (positions[n-2] - positions[n-3]) / dt2;
            acceleration = (v1 - v2) / ((dt1 + dt2) / 2.0);
        }
    }
    
    // 初始化平移滤波器
    TransStateVec trans_init;
    trans_init << final_pos, velocity, acceleration;
    translation_filter_->init_x(trans_init);
    
    // 设置初始协方差
    Eigen::Matrix<double, TRANS_DIM, TRANS_DIM> P_trans = 
        Eigen::Matrix<double, TRANS_DIM, TRANS_DIM>::Identity();
    
    // 位置的初始不确定性较小
    P_trans.block<3,3>(0,0) *= 0.01;
    
    // 速度的初始不确定性
    P_trans.block<3,3>(3,3) *= params_.init_vel_cov;
    
    // 加速度的初始不确定性较大
    P_trans.block<3,3>(6,6) *= params_.init_acc_cov;
    
    translation_filter_->set_P(P_trans);
    
    // 估计角度、角速度
    double final_theta = thetas.back();
    double omega = 0;
    double alpha = 0;
    
    if (n >= 2) {
        double dt = times[n-1] - times[n-2];
        if (dt > 0) {
            omega = angleDiff(thetas[n-1], thetas[n-2]) / dt;
        }
    }
    
    if (n >= 3) {
        double dt1 = times[n-1] - times[n-2];
        double dt2 = times[n-2] - times[n-3];
        
        if (dt1 > 0 && dt2 > 0) {
            double omega1 = angleDiff(thetas[n-1], thetas[n-2]) / dt1;
            double omega2 = angleDiff(thetas[n-2], thetas[n-3]) / dt2;
            alpha = (omega1 - omega2) / ((dt1 + dt2) / 2.0);
        }
    }
    
    // 初始化姿态滤波器
    PoseStateVec pose_init;
    pose_init << final_theta, omega, alpha;
    pose_filter_->init_x(pose_init);
    
    // 设置初始协方差
    Eigen::Matrix<double, POSE_DIM, POSE_DIM> P_pose = 
        Eigen::Matrix<double, POSE_DIM, POSE_DIM>::Identity();
    
    P_pose(0,0) = 0.01;  // 角度
    P_pose(1,1) = params_.init_omega_cov;  // 角速度
    P_pose(2,2) = params_.init_alpha_cov;  // 角加速度
    
    pose_filter_->set_P(P_pose);
    
    // 初始化历史记录
    velocity_history_.push_back(velocity);
    heading_history_.push_back(final_theta);
    omega_history_.push_back(omega);
    
    // 计算初始耦合信息
    coupling_info_.speed = velocity.norm();
    coupling_info_.heading_rate = omega;
    
    return true;
}

// 获取融合状态
Eigen::Matrix<double, 11, 1> DecoupledTracker::getFusedState() const {
    Eigen::Matrix<double, 11, 1> fused_state;
    fused_state.setZero();
    
    if (!initialized_) {
        return fused_state;
    }
    
    // 获取平移状态：位置、速度、加速度
    TransStateVec trans_state = translation_filter_->get_x();
    fused_state(0) = trans_state(0);  // x
    fused_state(1) = trans_state(1);  // y
    fused_state(2) = trans_state(2);  // z
    fused_state(4) = trans_state(3);  // vx
    fused_state(5) = trans_state(4);  // vy
    fused_state(6) = trans_state(5);  // vz
    fused_state(8) = trans_state(6);  // ax
    fused_state(9) = trans_state(7);  // ay
    // fused_state(10) 预留给 az，但我们的模型中 z 方向加速度为 0
    
    // 获取姿态状态：角度、角速度、角加速度
    PoseStateVec pose_state = pose_filter_->get_x();
    fused_state(3) = pose_state(0);   // theta
    fused_state(7) = pose_state(1);   // omega
    fused_state(10) = pose_state(2);  // alpha
    
    return fused_state;
}

} // namespace filter_lib