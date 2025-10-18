/**
 * @file rotating_target_tracker.hpp
 * @brief 旋转目标跟踪滤波器 - 多装甲板目标的位置和姿态联合估计
 * @details 
 * 核心设计理念：
 * - 使用三个协同工作的滤波器系统来跟踪旋转目标
 * - 主EKF跟踪目标中心位置和旋转状态
 * - 辅助KF分别优化中心位置和角速度估计
 * - 支持多装甲板目标的切换检测和管理
 */

#ifndef FILTER_LIB__ROTATING_TARGET_TRACKER_HPP_
#define FILTER_LIB__ROTATING_TARGET_TRACKER_HPP_

#include "base/ekf.hpp"
#include "base/kalman.hpp"
#include "base/math.hpp"
#include "base/slide_weighted_avg.hpp"
#include <Eigen/Dense>
#include <array>
#include <deque>
#include <memory>

namespace filter_lib {

/**
 * @brief RotatingTargetTracker 旋转目标跟踪器
 * @details 
 * 主状态向量（9维）：[x, y, z, theta, vx, vy, vz, omega, r]
 * - x, y, z: 目标中心的三维位置
 * - theta: 装甲板朝向角
 * - vx, vy, vz: 中心的三维速度
 * - omega: 旋转角速度
 * - r: 装甲板到中心的半径
 * 
 * 观测向量（4维）：[x, y, z, theta] - 装甲板的位置和朝向
 */
class RotatingTargetTracker {
public:
    // 维度常量定义
    static constexpr int MAIN_DIM_STATE = 9;   // 主EKF状态维度
    static constexpr int MAIN_DIM_MEAS = 4;    // 主EKF观测维度
    static constexpr int CENTER_DIM_STATE = 4; // 中心KF状态维度 [x, y, vx, vy]
    static constexpr int CENTER_DIM_MEAS = 2;  // 中心KF观测维度 [x, y]
    static constexpr int OMEGA_DIM_STATE = 3;  // 角速度KF状态维度 [theta, omega, alpha]
    static constexpr int OMEGA_DIM_MEAS = 1;   // 角速度KF观测维度 [theta]
    
    // 类型别名
    using MainState = Eigen::Matrix<double, MAIN_DIM_STATE, 1>;
    using Meas = Eigen::Matrix<double, MAIN_DIM_MEAS, 1>;
    using CenterState = Eigen::Matrix<double, CENTER_DIM_STATE, 1>;
    using OmegaState = Eigen::Matrix<double, OMEGA_DIM_STATE, 1>;
    
    /**
     * @brief 观测数据结构
     */
    struct Observation {
        double x, y, z;
        double theta;
        double timestamp;
        
        Meas vec() const { 
            return Meas(x, y, z, theta); 
        }
        
        Observation(double x_, double y_, double z_, double theta_, double t)
            : x(x_), y(y_), z(z_), theta(theta_), timestamp(t) {}
    };
    
    /**
     * @brief 跟踪器参数配置
     */
    struct Params {
        // 物理约束
        double radius_min;      // 最小半径（米）
        double radius_max;      // 最大半径（米）
        int armor_count;        // 装甲板数量（2或4）
        
        // 滤波器参数
        bool enable_height_filter;  // 是否启用高度加权平均
        
        // 开火控制参数
        int min_update_count;    // 开火前最少更新次数
        double max_fire_delay;   // 最大开火延迟（秒）
        double armor_fire_angle; // 跟随模式开火角度阈值（弧度）
        double center_fire_angle;// 中心模式开火角度阈值（弧度）
        
        // 构造函数提供默认值
        Params() :
            radius_min(0.15),
            radius_max(0.4),
            armor_count(4),
            enable_height_filter(false),
            min_update_count(100),
            max_fire_delay(0.5),
            armor_fire_angle(0.5),
            center_fire_angle(0.2)
        {}
    };
    
    /**
     * @brief 详细状态信息
     */
    struct Status {
        int update_count;          // 更新次数
        double radius;             // 当前半径估计
        int current_armor;         // 当前跟踪的装甲板索引
        Eigen::Vector3d center;    // 目标中心位置
        Eigen::Vector2d center_vel;// 中心速度（水平）
        double omega;              // 角速度
        bool ready_to_fire;        // 是否可以开火
    };
    
    /**
     * @brief 构造函数
     * @param params 参数配置
     */
    explicit RotatingTargetTracker(const Params& params);
    
    /**
     * @brief 使用默认参数的构造函数
     */
    RotatingTargetTracker() : RotatingTargetTracker(Params()) {}
    
    /**
     * @brief 析构函数
     */
    ~RotatingTargetTracker() = default;
    
    /**
     * @brief 初始化滤波器
     * @param observations 观测序列
     * @return 是否成功初始化
     */
    bool init(const std::deque<Observation>& observations);
    
    /**
     * @brief 更新滤波器
     * @param obs 新的观测
     */
    void update(const Observation& obs);
    
    /**
     * @brief 预测装甲板位置（用于瞄准装甲板）
     * @param dt 预测时间间隔（秒）
     * @return 预测的装甲板位置和朝向
     */
    Meas predict_armor(double dt) const;
    
    /**
     * @brief 预测目标中心（用于直接打击中心）
     * @param dt 预测时间间隔（秒）
     * @return 预测的中心位置（theta为预测的装甲板朝向）
     */
    Meas predict_center(double dt) const;
    
    /**
     * @brief 获取当前状态
     * @return 主滤波器状态向量
     */
    MainState state() const { 
        return main_ekf_ ? main_ekf_->get_x() : MainState::Zero(); 
    }
    
    /**
     * @brief 获取中心位置
     * @return 目标中心三维位置
     */
    Eigen::Vector3d center_position() const {
        const auto& x = state();
        return Eigen::Vector3d(x(0), x(1), x(2));
    }
    
    /**
     * @brief 获取中心速度
     * @return 目标中心三维速度
     */
    Eigen::Vector3d center_velocity() const {
        const auto& x = state();
        return Eigen::Vector3d(x(4), x(5), x(6));
    }
    
    /**
     * @brief 获取角速度
     * @return 旋转角速度（弧度/秒）
     */
    double angular_velocity() const { 
        return omega_filter_ ? omega_filter_->get_x_k1()(1) : 0.0; 
    }
    
    /**
     * @brief 获取当前半径
     * @return 装甲板到中心的半径（米）
     */
    double radius() const {
        return main_ekf_ ? main_ekf_->get_x()(8) : 0.0;
    }
    
    /**
     * @brief 获取当前跟踪的装甲板索引
     * @return 装甲板索引（0到armor_count-1）
     */
    int current_armor() const { return current_armor_; }
    
    /**
     * @brief 获取详细状态信息
     * @return 状态结构体
     */
    Status status() const;
    
    /**
     * @brief 检查是否适合开火（装甲板模式）
     * @param predicted_pose 预测的位置
     * @return 是否可以开火
     */
    bool check_fire_armor(const Meas& predicted_pose) const;
    
    /**
     * @brief 检查是否适合开火（中心模式）
     * @param predicted_pose 预测的位置
     * @return 是否可以开火
     */
    bool check_fire_center(const Meas& predicted_pose) const;
    
    /**
     * @brief 检查是否已初始化
     * @return 初始化状态
     */
    bool initialized() const { return initialized_; }
    
    /**
     * @brief 获取更新次数
     * @return 已执行的更新次数
     */
    int update_count() const { return update_count_; }
    
    /**
     * @brief 重置滤波器
     */
    void reset();
    
    /**
     * @brief 设置参数
     * @param params 新的参数配置
     */
    void set_params(const Params& params) { params_ = params; }
    
    /**
     * @brief 设置主滤波器过程噪声
     */
    void set_main_process_noise(const Eigen::Matrix<double, 9, 9>& Q);
    
    /**
     * @brief 设置主滤波器测量噪声
     */
    void set_main_measurement_noise(const Eigen::Matrix<double, 4, 4>& R);

private:
    /**
     * @brief 主EKF状态转移模型
     * @details 实现恒速旋转运动模型
     */
    struct MainTransitionModel {
        double dt;
        
        explicit MainTransitionModel(double dt_) : dt(dt_) {}
        
        template<typename T>
        void operator()(const T x_prev[MAIN_DIM_STATE], T x_curr[MAIN_DIM_STATE]) const;
    };
    
    /**
     * @brief 主EKF观测模型
     * @details 从中心状态和半径计算装甲板位置
     */
    struct MainMeasurementModel {
        template<typename T>
        void operator()(const T x[MAIN_DIM_STATE], T y[MAIN_DIM_MEAS]) const;
    };
    
    // 内部辅助函数
    double normalize_angle(double angle) const;
    double angle_diff(double a1, double a2) const;
    int detect_armor_switch(double new_angle, double old_angle) const;
    double calc_switch_weight(double theta) const;
    bool estimate_initial_state(const std::deque<Observation>& observations);
    void update_armor_params(const Observation& obs);

private:
    // 三个滤波器对象
    std::unique_ptr<AdaptiveEkf<MAIN_DIM_STATE, MAIN_DIM_MEAS>> main_ekf_;     // 主EKF
    std::unique_ptr<Kalman<CENTER_DIM_MEAS, CENTER_DIM_STATE>> center_filter_; // 中心位置KF
    std::unique_ptr<Kalman<OMEGA_DIM_MEAS, OMEGA_DIM_STATE>> omega_filter_;    // 角速度KF
    
    // 装甲板参数存储（支持2个装甲板的情况）
    std::array<double, 2> armor_r_;  // 两个装甲板的半径
    std::array<double, 2> armor_z_;  // 两个装甲板的高度
    
    // 参数配置
    Params params_;
    
    // 状态变量
    int current_armor_ = 0;      // 当前装甲板标识
    int update_count_ = 0;       // 更新次数计数
    bool initialized_ = false;   // 是否已初始化
    double last_timestamp_ = 0;  // 上次更新时间戳
    
    // 高度加权平均（可选功能）
    std::array<SlideWeightedAvg<double>, 2> weighted_z_;
    
    // 噪声矩阵
    Eigen::Matrix<double, MAIN_DIM_STATE, MAIN_DIM_STATE> Q_main_;
    Eigen::Matrix<double, MAIN_DIM_MEAS, MAIN_DIM_MEAS> R_main_;
};

} // namespace filter_lib

#endif // FILTER_LIB__ROTATING_TARGET_TRACKER_HPP_