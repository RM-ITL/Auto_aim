/**
 * @file decoupled_tracker.hpp
 * @brief 解耦式目标跟踪滤波器 - 修复版本
 */

#ifndef FILTER_LIB__DECOUPLED_TRACKER_HPP_
#define FILTER_LIB__DECOUPLED_TRACKER_HPP_

#include "base/ekf.hpp"
#include <Eigen/Dense>
#include <memory>
#include <deque>
#include <chrono>
#include <cmath>

namespace filter_lib {

class DecoupledTracker {
public:
    // 维度定义
    static constexpr int TRANS_DIM = 9;
    static constexpr int TRANS_OBS = 3;
    static constexpr int POSE_DIM = 3;
    static constexpr int POSE_OBS = 1;
    
    // 类型别名
    using TransStateVec = Eigen::Matrix<double, TRANS_DIM, 1>;      // 状态转移矩阵与观测矩阵
    using TransObsVec = Eigen::Matrix<double, TRANS_OBS, 1>;
    using PoseStateVec = Eigen::Matrix<double, POSE_DIM, 1>;
    using PoseObsVec = Eigen::Matrix<double, POSE_OBS, 1>;
    using MeasVec = Eigen::Matrix<double, 4, 1>;                    //观测向量
    using TimePoint = std::chrono::steady_clock::time_point;
    
    struct CouplingInfo {                                           // 耦合信息的结构体
        double rho = 0.0;
        double consistency = 0.0;
        double speed = 0.0;
        double heading_rate = 0.0;
        Eigen::Vector3d velocity_dir;
        Eigen::Vector2d heading_dir;
        double velocity_angle = 0.0;
        double angle_diff = 0.0;
        std::string mode_description;
        double coupling_correction = 0.0;
    };
    
    /**
     * @brief 滤波器参数结构体
     * 注意：不使用默认成员初始化器以避免编译器兼容性问题
     */
    struct Parameters {
        double trans_q_pos;
        double trans_q_vel;
        double trans_q_acc;
        double trans_r;
        double pose_q_theta;
        double pose_q_omega;
        double pose_q_alpha;
        double pose_r;
        double coupling_sigma;
        double min_speed;
        double min_rho;
        double max_rho;
        double rho_threshold;
        double virtual_obs_base_sigma;
        double virtual_obs_scaling;
        double init_vel_cov;
        double init_acc_cov;
        double init_omega_cov;
        double init_alpha_cov;
        
        // 提供构造函数来设置默认值
        Parameters() 
            : trans_q_pos(0.01),
              trans_q_vel(0.1),
              trans_q_acc(5.0),
              trans_r(0.05),
              pose_q_theta(0.001),
              pose_q_omega(0.01),
              pose_q_alpha(0.1),
              pose_r(0.05),
              coupling_sigma(0.5),
              min_speed(0.1),
              min_rho(0.0),
              max_rho(0.9),
              rho_threshold(0.3),
              virtual_obs_base_sigma(0.3),
              virtual_obs_scaling(2.0),
              init_vel_cov(1.0),
              init_acc_cov(10.0),
              init_omega_cov(0.1),
              init_alpha_cov(1.0) {}
    };

public:
    // 修改构造函数声明，使用重载而不是默认参数
    explicit DecoupledTracker(const Parameters& params);
    DecoupledTracker();  // 无参构造函数
    
    bool init(const std::deque<Eigen::Matrix<double, 5, 1>>& poses);        // 初始化函数
    void update(const MeasVec& measurement, TimePoint timestamp);           // 更新函数
    MeasVec predict(double dt) const;                                       // 预测函数
    Eigen::Matrix<double, 11, 1> getFusedState() const;                     // 获取状态向量，将两个滤波器耦合之后
    
    CouplingInfo getCouplingInfo() const { return coupling_info_; }         // 获取耦合信息
    bool isInitialized() const { return initialized_; }                     // 初始化判断
    
    std::pair<TransStateVec, Eigen::Matrix<double, TRANS_DIM, TRANS_DIM>>   // 获取平移滤波器的协方差
    getTransStateWithCovariance() const;
    
    std::pair<PoseStateVec, Eigen::Matrix<double, POSE_DIM, POSE_DIM>>      // 获取姿态滤波器协方差
    getPoseStateWithCovariance() const;
    
    void reset();                                                           // 重置滤波器
    void setParameters(const Parameters& params) { params_ = params; }      

private:
    // 状态转移模型和观测模型定义
    struct TransitionModelTrans {
        double dt;
        
        TransitionModelTrans(double dt_) : dt(dt_) {}                       // 平移状态转移模型
        
        template<typename T>
        void operator()(const T x_prev[TRANS_DIM], T x_curr[TRANS_DIM]) const {
            x_curr[0] = x_prev[0] + x_prev[3]*dt + T(0.5)*x_prev[6]*dt*dt;
            x_curr[1] = x_prev[1] + x_prev[4]*dt + T(0.5)*x_prev[7]*dt*dt;
            x_curr[2] = x_prev[2] + x_prev[5]*dt + T(0.5)*x_prev[8]*dt*dt;
            x_curr[3] = x_prev[3] + x_prev[6]*dt;
            x_curr[4] = x_prev[4] + x_prev[7]*dt;
            x_curr[5] = x_prev[5] + x_prev[8]*dt;
            x_curr[6] = x_prev[6];
            x_curr[7] = x_prev[7];
            x_curr[8] = x_prev[8];
        }
    };
    
    struct MeasurementModelTrans {                                          // 平移观测
        template<typename T>
        void operator()(const T x[TRANS_DIM], T y[TRANS_OBS]) const {
            y[0] = x[0];
            y[1] = x[1];
            y[2] = x[2];
        }
    };
    
    struct TransitionModelPose {                                            // 姿态转移
        double dt;      
        
        TransitionModelPose(double dt_) : dt(dt_) {}
        
        template<typename T>
        void operator()(const T x_prev[POSE_DIM], T x_curr[POSE_DIM]) const {
            T theta_new = x_prev[0] + x_prev[1]*dt + T(0.5)*x_prev[2]*dt*dt;
            x_curr[0] = normalizeAngle(theta_new);
            x_curr[1] = x_prev[1] + x_prev[2]*dt;
            x_curr[2] = x_prev[2];
        }
        
        template<typename T>
        static T normalizeAngle(const T& angle) {
            T normalized = angle;
            while (normalized > T(M_PI)) normalized -= T(2*M_PI);
            while (normalized < T(-M_PI)) normalized += T(2*M_PI);
            return normalized;
        }
    };
    
    struct MeasurementModelPose {                                       // 姿态观测
        template<typename T>
        void operator()(const T x[POSE_DIM], T y[POSE_OBS]) const {
            y[0] = x[0];
        }
    };
    
    struct VirtualMeasurementModel {                                   // 虚拟观测方法，当前未使用
        const TransStateVec& trans_state;
        const PoseStateVec& pose_state;
        
        VirtualMeasurementModel(const TransStateVec& ts, const PoseStateVec& ps) 
            : trans_state(ts), pose_state(ps) {}
        
        template<typename T>
        void operator()(const T x[TRANS_DIM], T y[1]) const {
            T vel_angle = ceres::atan2(x[4], x[3]);
            y[0] = vel_angle - T(pose_state(0));
            while (y[0] > T(M_PI)) y[0] -= T(2*M_PI);
            while (y[0] < T(-M_PI)) y[0] += T(2*M_PI);
        }
    };

private:
    void updateCouplingStrength();                                     // 更新耦合强度
    void applyCouplingCorrection(double dt);                           // 应用耦合校正
    static double normalizeAngle(double angle);                        // 角度归一化
    static double angleDiff(double a1, double a2);
    bool estimateInitialStates(const std::deque<Eigen::Matrix<double, 5, 1>>& poses);   // 初始状态估计
    Eigen::Matrix<double, TRANS_DIM, TRANS_DIM> buildTransQ(double dt) const;   // 过程噪声构建：平移和姿态
    Eigen::Matrix<double, POSE_DIM, POSE_DIM> buildPoseQ(double dt) const;

private:
    std::unique_ptr<AdaptiveEkf<TRANS_DIM, TRANS_OBS>> translation_filter_;     // 平移滤波器
    std::unique_ptr<AdaptiveEkf<POSE_DIM, POSE_OBS>> pose_filter_;              // 姿态滤波器
    bool initialized_ = false;
    TimePoint last_update_time_;
    int update_count_ = 0;
    CouplingInfo coupling_info_;
    Parameters params_;
    std::deque<Eigen::Vector3d> velocity_history_;
    std::deque<double> heading_history_;
    std::deque<double> omega_history_;
    static constexpr size_t HISTORY_SIZE = 20;
};

} // namespace filter_lib

#endif // FILTER_LIB__DECOUPLED_TRACKER_HPP_