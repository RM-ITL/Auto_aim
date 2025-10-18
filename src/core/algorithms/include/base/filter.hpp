/**
 * @file filter.hpp
 * @brief 高级滤波器模块
 * @details 提供单变量滤波、位置滤波和角度滤波等高级功能
 */

#ifndef FILTER_LIB__FILTER_HPP_
#define FILTER_LIB__FILTER_HPP_

#include "ekf.hpp"
#include "kalman.hpp"
#include "math.hpp"
#include <memory>

namespace filter_lib {

/**
 * @brief 单变量预测函数对象
 * @tparam N 状态维度（阶数）
 * 
 * @details 实现了N阶泰勒展开的状态预测
 * - N=1: 常值模型 x(t+dt) = x(t)
 * - N=2: 匀速模型 x(t+dt) = x(t) + v(t)*dt
 * - N=3: 匀加速模型 x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
 */
template<int N>
class SinglePredict {
public:
    explicit SinglePredict(double delta_t) : delta_t_(delta_t) {}
    
    template<typename T>
    void operator()(const T x_prev[N], T x_curr[N]) const {
        // 计算泰勒展开系数：1, 1/1!, 1/2!, ...
        double coeff[N] = {1.0};
        for (int i = 1; i < N; ++i) {
            coeff[i] = coeff[i-1] / static_cast<double>(i);
        }
        
        // 应用泰勒展开进行预测
        for (int i = 0; i < N; ++i) {
            x_curr[i] = x_prev[i];
            for (int j = i + 1; j < N; ++j) {
                x_curr[i] += coeff[j-i] * std::pow(delta_t_, j-i) * x_prev[j];
            }
        }
    }
    
private:
    double delta_t_;
};

/**
 * @brief 单变量观测函数对象
 * @tparam N 状态维度
 * 
 * @details 只观测状态的第一个分量（通常是位置）
 */
template<int N>
class SingleMeasure {
public:
    template<typename T>
    void operator()(const T x[N], T y[1]) const {
        y[0] = x[0];  // 只观测位置
    }
};

/**
 * @brief 单变量滤波器
 * @tparam ORDER 滤波器阶数（1-3）
 * 
 * @details 封装了AdaptiveEkf，提供简单的单变量滤波接口
 * 适用于一维信号的滤波，如单个角度或距离
 */
template<int ORDER>
class SingleFilter {
public:
    using Ekf = AdaptiveEkf<ORDER, 1>;
    using StateVector = Eigen::Matrix<double, ORDER, 1>;
    
    SingleFilter() = default;
    
    /**
     * @brief 初始化滤波器状态
     * @param x 初始状态向量
     * 
     * @details 设置初始状态并将协方差初始化为单位矩阵
     */
    void init_x(const StateVector& x) {
        ekf_.init_x(x);
    }
    
    /**
     * @brief 设置滤波器状态
     * @param x 新的状态向量
     */
    void set_x(const StateVector& x) {
        ekf_.set_x(x);
    }
    
    /**
     * @brief 设置当前时间
     * @param t 时间戳
     */
    void set_t(double t) {
        ekf_t_ = t;
    }
    
    /**
     * @brief 获取当前状态估计
     * @return 状态向量
     */
    StateVector get_x() const {
        return ekf_.get_x();
    }
    
    /**
     * @brief 预测指定时刻的状态
     * @param t 目标时刻
     * @return 预测的状态向量
     */
    StateVector predict(double t) const {
        SinglePredict<ORDER> predict_func(t - ekf_t_);
        auto pre_res = ekf_.predict(predict_func);
        return pre_res.x_p;
    }
    
    /**
     * @brief 更新滤波器
     * @param x 观测值
     * @param t 观测时刻
     * @param q_vec 过程噪声（对角元素）
     * @param r_vec 测量噪声（单元素向量）
     */
    void update(double x, double t, 
                const std::vector<double>& q_vec,
                const std::vector<double>& r_vec) {
        SingleMeasure<ORDER> measure_func;
        SinglePredict<ORDER> predict_func(t - ekf_t_);
        
        ekf_.update(
            measure_func,
            predict_func,
            math::vec_to_column_matrix<1>({x}),
            math::vec_to_diagonal_matrix<ORDER>(q_vec),
            math::vec_to_diagonal_matrix<1>(r_vec)
        );
        
        ekf_t_ = t;
    }
    
private:
    Ekf ekf_;
    double ekf_t_ = 0.0;
};

/**
 * @brief 位置预测器接口
 * @details 定义了位置预测器的标准接口
 */
class PositionPredictorInterface {
public:
    virtual ~PositionPredictorInterface() = default;
    
    /**
     * @brief 预测指定时刻的位置
     * @param t 目标时刻
     * @return 3D位置向量
     */
    virtual Eigen::Vector3d predict_pos(double t) const = 0;
    
    /**
     * @brief 预测指定时刻的速度
     * @param t 目标时刻
     * @return 3D速度向量
     */
    virtual Eigen::Vector3d predict_v(double t) const = 0;
};

/**
 * @brief EKF预测函数对象（匀速模型）
 * @details 用于6维状态（x,vx,y,vy,z,vz）的预测
 */
class EkfPredict {
public:
    explicit EkfPredict(double delta_t) : delta_t_(delta_t) {}
    
    template<typename T>
    void operator()(const T x_prev[6], T x_curr[6]) const {
        // 位置更新：x = x + v * dt
        x_curr[0] = x_prev[0] + x_prev[1] * delta_t_;  // x
        x_curr[2] = x_prev[2] + x_prev[3] * delta_t_;  // y
        x_curr[4] = x_prev[4] + x_prev[5] * delta_t_;  // z
        
        // 速度保持不变（匀速模型）
        x_curr[1] = x_prev[1];  // vx
        x_curr[3] = x_prev[3];  // vy
        x_curr[5] = x_prev[5];  // vz
    }
    
private:
    double delta_t_;
};

/**
 * @brief EKF测量函数对象
 * @details 将笛卡尔坐标转换为YPD坐标进行观测
 */
class EkfMeasure {
public:
    template<typename T>
    void operator()(const T x[6], T y[3]) const {
        // 从状态中提取位置
        T pos[3] = {x[0], x[2], x[4]};
        // 转换为YPD坐标
        math::ceres_xyz_to_ypd(pos, y);
    }
};

/**
 * @brief 位置扩展卡尔曼滤波器
 * @details 专门用于3D位置跟踪的EKF实现
 * 
 * 状态向量：[x, vx, y, vy, z, vz]
 * 观测向量：[yaw, pitch, distance]
 * 
 * 这种设计允许在笛卡尔坐标系中进行状态估计，
 * 同时支持球坐标系的观测输入
 */
class PositionEkf : public PositionPredictorInterface {
public:
    using Ekf = AdaptiveEkf<6, 3>;
    
    PositionEkf() = default;
    
    /**
     * @brief 预测指定时刻的位置
     * @param t 目标时刻
     * @return 3D位置向量
     */
    Eigen::Vector3d predict_pos(double t) const override {
        EkfPredict predict_func(t - ekf_t_);
        auto pre_res = ekf_.predict(predict_func);
        return Eigen::Vector3d(pre_res.x_p(0), pre_res.x_p(2), pre_res.x_p(4));
    }
    
    /**
     * @brief 预测指定时刻的速度
     * @param t 目标时刻
     * @return 3D速度向量
     */
    Eigen::Vector3d predict_v(double t) const override {
        EkfPredict predict_func(t - ekf_t_);
        auto pre_res = ekf_.predict(predict_func);
        return Eigen::Vector3d(pre_res.x_p(1), pre_res.x_p(3), pre_res.x_p(5));
    }
    
    /**
     * @brief 使用YPD观测初始化滤波器
     * @param ypd 初始YPD坐标
     * @param t 初始时刻
     */
    void init(const math::YpdCoord& ypd, double t) {
        Eigen::Vector3d pos = math::ypd_to_xyz(ypd);
        Eigen::Matrix<double, 6, 1> initial_state;
        initial_state << pos(0), 0.0, pos(1), 0.0, pos(2), 0.0;
        ekf_.init_x(initial_state);
        ekf_t_ = t;
    }
    
    /**
     * @brief 使用YPD观测更新滤波器
     * @param ypd 观测的YPD坐标
     * @param t 观测时刻
     * @param q_vec 过程噪声（6个元素）
     * @param r_vec 测量噪声（3个元素）
     * 
     * @details 处理角度跳变问题，确保滤波器稳定
     */
    void update(const math::YpdCoord& ypd, double t,
                const std::vector<double>& q_vec,
                const std::vector<double>& r_vec) {
        EkfMeasure measure_func;
        EkfPredict predict_func(t - ekf_t_);
        
        // 先进行预测步骤
        ekf_.predict_forward(predict_func, math::vec_to_diagonal_matrix<6>(q_vec));
        
        // 获取预测的YPD值
        auto mea_res = ekf_.measure(measure_func);
        math::YpdCoord ypd_pred(mea_res.y_e(0), mea_res.y_e(1), mea_res.y_e(2));
        
        // 处理角度跳变：找到最接近预测值的观测角度表示
        std::vector<double> ypd_closest = {
            math::get_closest(ypd.yaw, ypd_pred.yaw, 2.0 * M_PI),
            math::get_closest(ypd.pitch, ypd_pred.pitch, 2.0 * M_PI),
            ypd.dis
        };
        
        // 执行更新步骤
        ekf_.update_forward(
            measure_func,
            math::vec_to_column_matrix<3>(ypd_closest),
            math::vec_to_diagonal_matrix<3>(r_vec)
        );
        
        ekf_t_ = t;
    }
    
private:
    Ekf ekf_;
    double ekf_t_ = 0.0;
};

/**
 * @brief 角度滤波器
 * @tparam V_Z 观测维度（通常为1）
 * @tparam V_X 状态维度（1-3）
 * 
 * @details 专门处理角度数据的滤波器
 * 解决了角度周期性带来的跳变问题
 */
template<int V_Z, int V_X>
class AngleFilter {
public:
    /**
     * @brief 构造函数
     * @param range 角度范围（通常是2π）
     * @param Q 过程噪声向量
     * @param R 测量噪声
     */
    AngleFilter(double range, const std::vector<double>& Q, double R)
        : filter_(Q, R, V_X), range_(range) {}
    
    /**
     * @brief 更新滤波器
     * @param angle 观测角度
     * @param t 观测时刻
     * 
     * @details 自动处理角度跳变
     */
    void update(double angle, double t) {
        // 找到最接近预测值的角度表示
        double closest_angle = math::get_closest(
            angle, 
            filter_.predict(t)(0, 0), 
            range_
        );
        
        filter_.update(closest_angle, t);
        
        // 确保内部角度在有效范围内
        auto x = filter_.get_x_k1();
        x(0, 0) = math::reduced(x(0, 0), range_);
        filter_.set_x(x);
    }
    
    /**
     * @brief 带自定义噪声的更新
     * @param angle 观测角度
     * @param t 观测时刻
     * @param R 测量噪声
     */
    void update(double angle, double t, double R) {
        double closest_angle = math::get_closest(
            angle, 
            filter_.predict(t)(0, 0), 
            range_
        );
        
        filter_.update(closest_angle, t, R);
        
        auto x = filter_.get_x_k1();
        x(0, 0) = math::reduced(x(0, 0), range_);
        filter_.set_x(x);
    }
    
    /**
     * @brief 初始化滤波器
     */
    void init() {
        filter_.init();
    }
    
    /**
     * @brief 获取当前角度估计
     * @return 角度值（弧度）
     */
    double get_angle() const {
        return filter_.get_x_k1()(0, 0);
    }
    
    /**
     * @brief 获取角速度估计
     * @return 角速度（弧度/秒）
     * 
     * @note 仅当V_X >= 2时有效
     */
    double get_angular_velocity() const {
        if (V_X >= 2) {
            return filter_.get_x_k1()(1, 0);
        }
        return 0.0;
    }
    
    /**
     * @brief 预测指定时刻的角度
     * @param t 目标时刻
     * @return 预测的角度值
     */
    double predict_angle(double t) const {
        return math::reduced(filter_.predict(t)(0, 0), range_);
    }
    
private:
    Kalman<V_Z, V_X> filter_;
    double range_;
};

} // namespace filter_lib

#endif // FILTER_LIB__FILTER_HPP_