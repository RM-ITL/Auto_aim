/**
 * @file track_queue_v4.hpp
 * @brief 简化版极坐标系恒定加速度跟踪滤波器
 * @details 专注于EKF核心功能，提供最简洁的接口
 */

#ifndef FILTER_LIB__TRACK_QUEUE_V4_HPP_
#define FILTER_LIB__TRACK_QUEUE_V4_HPP_

#include "base/ekf.hpp"
#include "base/math.hpp"
#include <Eigen/Dense>
#include <deque>

namespace filter_lib {

/**
 * @brief TrackQueueV4 简化版跟踪滤波器
 * @details 
 * 状态向量（8维）：[x, y, z, v, vz, theta, omega, a]
 * - x, y, z: 三维位置
 * - v: 水平速度大小
 * - vz: 垂直速度
 * - theta: 水平运动方向角
 * - omega: 角速度
 * - a: 水平加速度大小
 * 
 * 观测向量（3维）：[x, y, z]
 * 
 * 设计理念：
 * - 只提供核心的EKF算法功能
 * - 不管理时间，由外部调用者提供时间差
 * - 参数在构造时设置，使用合理的默认值
 */
class TrackQueueV4 {
public:
    // 维度常量
    static constexpr int DIM_STATE = 8;
    static constexpr int DIM_MEAS = 3;
    
    // 类型别名 - 使用简洁的命名
    using State = Eigen::Matrix<double, DIM_STATE, 1>;
    using Meas = Eigen::Matrix<double, DIM_MEAS, 1>;
    using StateCov = Eigen::Matrix<double, DIM_STATE, DIM_STATE>;
    using MeasCov = Eigen::Matrix<double, DIM_MEAS, DIM_MEAS>;
    
    /**
     * @brief 观测数据结构 - 包含位置和时间戳
     */
    struct Observation {
        double x, y, z;
        double timestamp;
        
        // 转换为向量格式
        Meas vec() const { 
            return Meas(x, y, z); 
        }
        
        // 从向量构造
        Observation(const Meas& m, double t) 
            : x(m(0)), y(m(1)), z(m(2)), timestamp(t) {}
            
        Observation(double x_, double y_, double z_, double t)
            : x(x_), y(y_), z(z_), timestamp(t) {}
    };
    
    /**
     * @brief 构造函数
     */
    TrackQueueV4();
    
    /**
     * @brief 使用观测序列初始化滤波器
     * @param observations 观测序列，至少需要3个点
     * @return 是否成功初始化
     */
    bool init(const std::deque<Observation>& observations);
    
    /**
     * @brief 更新滤波器状态
     * @param meas 新的观测值
     * @param dt 时间间隔（秒）
     */
    void update(const Meas& meas, double dt);
    
    /**
     * @brief 预测未来状态
     * @param dt 预测时间间隔（秒）
     * @return 预测的状态向量
     */
    State predict(double dt) const;
    
    /**
     * @brief 获取当前状态
     * @return 状态向量
     */
    State state() const { return ekf_.get_x(); }
    
    /**
     * @brief 获取状态协方差
     * @return 协方差矩阵
     */
    StateCov covariance() const { return ekf_.get_P(); }
    
    /**
     * @brief 获取当前位置
     * @return 三维位置向量
     */
    Meas position() const { 
        return ekf_.get_x().template head<3>(); 
    }
    
    /**
     * @brief 获取当前速度（水平速度大小和垂直速度）
     * @return [v, vz]
     */
    Eigen::Vector2d velocity() const {
        const auto& x = ekf_.get_x();
        return Eigen::Vector2d(x(3), x(4));
    }
    
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
     * @brief 设置过程噪声
     * @param q_pos 位置噪声标准差
     * @param q_vel 速度噪声标准差
     * @param q_acc 加速度噪声标准差
     */
    void set_process_noise(double q_pos, double q_vel, double q_acc);
    
    /**
     * @brief 设置测量噪声
     * @param r_pos 位置测量噪声标准差
     */
    void set_measurement_noise(double r_pos);

private:
    /**
     * @brief 状态转移函数对象
     */
    struct TransitionModel {
        double dt;
        
        explicit TransitionModel(double dt_) : dt(dt_) {}
        
        template<typename T>
        void operator()(const T x_prev[DIM_STATE], T x_curr[DIM_STATE]) const;
    };
    
    /**
     * @brief 观测函数对象
     */
    struct MeasurementModel {
        template<typename T>
        void operator()(const T x[DIM_STATE], T y[DIM_MEAS]) const {
            y[0] = x[0];  // x
            y[1] = x[1];  // y
            y[2] = x[2];  // z
        }
    };
    
    /**
     * @brief 从观测序列估计初始状态
     */
    bool estimate_initial_state(const std::deque<Observation>& observations, 
                               State& init_state);
    
    /**
     * @brief 构建过程噪声矩阵
     */
    StateCov build_process_noise(double dt) const;

private:
    AdaptiveEkf<DIM_STATE, DIM_MEAS> ekf_;  // EKF对象
    bool initialized_ = false;               // 初始化标志
    int update_count_ = 0;                   // 更新计数
    
    // 噪声参数
    double q_pos_ = 0.01;   // 位置过程噪声
    double q_vel_ = 0.1;    // 速度过程噪声  
    double q_acc_ = 1.0;    // 加速度过程噪声
    double r_pos_ = 0.05;   // 位置测量噪声
};

} // namespace filter_lib

#endif // FILTER_LIB__TRACK_QUEUE_V4_HPP_