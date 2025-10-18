/**
 * @file adaptive_ekf.hpp
 * @brief 扩展卡尔曼滤波器实现
 * @details 使用Ceres自动微分计算雅可比矩阵
 */

#ifndef FILTER_LIB__ADAPTIVE_EKF_HPP_
#define FILTER_LIB__ADAPTIVE_EKF_HPP_

#include <Eigen/Dense>
#include <ceres/jet.h>
#include <functional>

namespace filter_lib {

/**
 * @brief 扩展卡尔曼滤波器模板类
 * @tparam N_X 状态维度
 * @tparam N_Y 观测维度
 * 
 * @details 
 * 特点：
 * - 使用Ceres的Jet类型自动计算雅可比矩阵
 * - 支持任意非线性预测和观测模型
 * - 通过函数对象传入模型，保持灵活性
 */
template<int N_X, int N_Y>
class AdaptiveEkf {
public:
    // 类型别名
    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixYX = Eigen::Matrix<double, N_Y, N_X>;
    using MatrixXY = Eigen::Matrix<double, N_X, N_Y>;
    using MatrixYY = Eigen::Matrix<double, N_Y, N_Y>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixY1 = Eigen::Matrix<double, N_Y, 1>;

private:
    MatrixX1 x_e_;    // 状态估计
    MatrixXX p_mat_;  // 估计误差协方差
    
    static constexpr double INF = 1e9;

public:
    /**
     * @brief 默认构造函数
     * @details 初始化状态为零，协方差为大值
     */
    AdaptiveEkf() 
        : x_e_(MatrixX1::Zero()), 
          p_mat_(MatrixXX::Identity() * INF) {}
    
    /**
     * @brief 带初始状态的构造函数
     * @param x 初始状态向量
     */
    explicit AdaptiveEkf(const MatrixX1& x) 
        : x_e_(x), 
          p_mat_(MatrixXX::Identity() * INF) {}

    /**
     * @brief 初始化状态和协方差
     * @param x0 初始状态
     * @details 协方差初始化为单位矩阵
     */
    void init_x(const MatrixX1& x0) {
        x_e_ = x0;
        p_mat_ = MatrixXX::Identity();
    }

    /**
     * @brief 获取当前状态估计
     * @return 状态向量
     */
    MatrixX1 get_x() const {
        return x_e_;
    }

    /**
     * @brief 设置状态估计
     * @param x 新的状态向量
     */
    void set_x(const MatrixX1& x) {
        x_e_ = x;
    }
    
    /**
     * @brief 获取当前协方差矩阵
     * @return 协方差矩阵
     * @details 
     * 协方差矩阵反映了状态估计的不确定性。
     * 对角线元素是各状态分量的方差，非对角线元素是协方差。
     * 这对于评估预测质量和双轨模式切换非常重要。
     */
    MatrixXX get_P() const {
        return p_mat_;
    }
    
    /**
     * @brief 设置协方差矩阵
     * @param P 新的协方差矩阵
     * @details 
     * 在双轨模式中，当两个EKF需要同步状态时，
     * 不仅要同步状态向量，还要同步协方差矩阵，
     * 以保持不确定性估计的一致性。
     */
    void set_P(const MatrixXX& P) {
        p_mat_ = P;
    }

    /**
     * @brief 预测结果结构体
     */
    struct PredictResult {
        MatrixX1 x_p = MatrixX1::Zero();    // 预测状态
        MatrixXX f_mat = MatrixXX::Zero();  // 状态转移雅可比矩阵
    };

    /**
     * @brief 执行预测步骤（不更新内部状态）
     * @tparam PredictFunc 预测函数类型
     * @param predict_func 预测函数对象
     * @return 预测结果
     * 
     * @details 
     * predict_func应该具有签名：void(const T x_prev[N_X], T x_curr[N_X])
     * 其中T可以是double或ceres::Jet<double, N_X>
     */
    template<class PredictFunc>
    PredictResult predict(PredictFunc&& predict_func) const {
        // 创建Jet类型的状态变量用于自动微分
        ceres::Jet<double, N_X> x_e_jet[N_X];
        for (int i = 0; i < N_X; ++i) {
            x_e_jet[i].a = x_e_[i];     // 设置值
            x_e_jet[i].v[i] = 1.0;      // 设置对自身的偏导数为1
        }
        
        // 执行预测，自动计算雅可比矩阵
        ceres::Jet<double, N_X> x_p_jet[N_X];
        predict_func(x_e_jet, x_p_jet);
        
        // 提取预测值
        MatrixX1 x_p = MatrixX1::Zero();
        for (int i = 0; i < N_X; ++i) {
            x_p[i] = x_p_jet[i].a;
        }
        
        // 提取雅可比矩阵
        MatrixXX f_mat = MatrixXX::Zero();
        for (int i = 0; i < N_X; ++i) {
            f_mat.block(i, 0, 1, N_X) = x_p_jet[i].v.transpose();
        }
        
        return PredictResult{x_p, f_mat};
    }

    /**
     * @brief 执行预测步骤并更新内部状态
     * @tparam PredictFunc 预测函数类型
     * @param predict_func 预测函数对象
     * @param q_mat 过程噪声协方差矩阵
     */
    template<class PredictFunc>
    void predict_forward(PredictFunc&& predict_func, const MatrixXX& q_mat) {
        PredictResult pre_res = predict(predict_func);
        x_e_ = pre_res.x_p;
        p_mat_ = pre_res.f_mat * p_mat_ * pre_res.f_mat.transpose() + q_mat;
    }

    /**
     * @brief 测量结果结构体
     */
    struct MeasureResult {
        MatrixY1 y_e = MatrixY1::Zero();     // 预测观测
        MatrixYX h_mat = MatrixYX::Zero();   // 观测雅可比矩阵
    };

    /**
     * @brief 计算预测观测值
     * @tparam MeasureFunc 观测函数类型
     * @param measure_func 观测函数对象
     * @return 测量结果
     * 
     * @details 
     * measure_func应该具有签名：void(const T x[N_X], T y[N_Y])
     */
    template<class MeasureFunc>
    MeasureResult measure(MeasureFunc&& measure_func) const {
        // 创建Jet类型的状态变量
        ceres::Jet<double, N_X> x_e_jet[N_X];
        for (int i = 0; i < N_X; ++i) {
            x_e_jet[i].a = x_e_[i];
            x_e_jet[i].v[i] = 1.0;
        }
        
        // 计算观测值，自动计算雅可比矩阵
        ceres::Jet<double, N_X> y_e_jet[N_Y];
        measure_func(x_e_jet, y_e_jet);
        
        // 提取观测值
        MatrixY1 y_e = MatrixY1::Zero();
        for (int i = 0; i < N_Y; ++i) {
            y_e[i] = y_e_jet[i].a;
        }
        
        // 提取雅可比矩阵
        MatrixYX h_mat = MatrixYX::Zero();
        for (int i = 0; i < N_Y; ++i) {
            h_mat.block(i, 0, 1, N_X) = y_e_jet[i].v.transpose();
        }
        
        return MeasureResult{y_e, h_mat};
    }

    /**
     * @brief 执行更新步骤
     * @tparam MeasureFunc 观测函数类型
     * @param measure_func 观测函数对象
     * @param y_mat 实际观测值
     * @param r_mat 测量噪声协方差矩阵
     */
    template<class MeasureFunc>
    void update_forward(MeasureFunc&& measure_func, 
                       const MatrixY1& y_mat, 
                       const MatrixYY& r_mat) {
        MeasureResult mea_res = measure(measure_func);
        
        // 计算卡尔曼增益
        MatrixXY k_mat = p_mat_ * mea_res.h_mat.transpose() * 
            (mea_res.h_mat * p_mat_ * mea_res.h_mat.transpose() + r_mat).inverse();
        
        // 更新状态估计
        x_e_ = x_e_ + k_mat * (y_mat - mea_res.y_e);
        
        // 更新协方差
        p_mat_ = (MatrixXX::Identity() - k_mat * mea_res.h_mat) * p_mat_;
    }

    /**
     * @brief 完整的更新步骤（预测+更新）
     * @tparam MeasureFunc 观测函数类型
     * @tparam PredictFunc 预测函数类型
     * @param measure_func 观测函数对象
     * @param predict_func 预测函数对象
     * @param y_mat 实际观测值
     * @param q_mat 过程噪声协方差矩阵
     * @param r_mat 测量噪声协方差矩阵
     */
    template<class MeasureFunc, class PredictFunc>
    void update(MeasureFunc&& measure_func,
                PredictFunc&& predict_func,
                const MatrixY1& y_mat,
                const MatrixXX& q_mat,
                const MatrixYY& r_mat) {
        predict_forward(std::forward<PredictFunc>(predict_func), q_mat);
        update_forward(std::forward<MeasureFunc>(measure_func), y_mat, r_mat);
    }
};

} // namespace filter_lib

#endif // FILTER_LIB__ADAPTIVE_EKF_HPP_