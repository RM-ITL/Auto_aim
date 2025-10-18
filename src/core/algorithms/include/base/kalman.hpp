/**
 * @file kalman.hpp
 * @brief 传统卡尔曼滤波器实现
 * @details 提供多维度、多阶的卡尔曼滤波器模板类
 */

#ifndef FILTER_LIB__KALMAN_HPP_
#define FILTER_LIB__KALMAN_HPP_

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace filter_lib {

/**
 * @brief 传统卡尔曼滤波器模板类
 * @tparam V_Z 观测维度
 * @tparam V_X 状态维度
 * 
 * @details 
 * 状态维度说明：
 * - V_X = 1: 仅位置
 * - V_X = 2: 位置 + 速度
 * - V_X = 3: 位置 + 速度 + 加速度
 */
template<int V_Z, int V_X>
class Kalman {
public:
    // 类型别名，提高代码可读性
    using MatrixZZ = Eigen::Matrix<double, V_Z, V_Z>;
    using MatrixXX = Eigen::Matrix<double, V_X, V_X>;
    using MatrixZX = Eigen::Matrix<double, V_Z, V_X>;
    using MatrixXZ = Eigen::Matrix<double, V_X, V_Z>;
    using MatrixX1 = Eigen::Matrix<double, V_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, V_Z, 1>;

private:
    MatrixX1 x_k1_;          // 后验状态估计
    MatrixZX H_;             // 观测矩阵
    MatrixXX Q_;             // 过程噪声协方差
    MatrixZZ default_R_;     // 默认测量噪声协方差
    MatrixXX P_;             // 估计误差协方差
    int predict_order_;      // 预测阶数
    double t_;               // 当前时间戳
    
    static constexpr double INF = 998244353.0;  // 初始化时的大协方差值

public:

    /**
     * @brief 支持矩阵R的构造函数
     * @param Q 过程噪声协方差向量（对角元素）
     * @param R 测量噪声协方差矩阵
     * @param predict_order 预测阶数（1-3）
     */
    Kalman(const std::vector<double>& Q, 
        const MatrixZZ& R, 
        const int& predict_order)
        : predict_order_(predict_order), t_(0.0) {
        
        // 初始化观测矩阵
        H_ = MatrixZX::Zero();
        for (int i = 0; i < V_Z; ++i) {
            H_(i, i) = 1;  // 对角线为1
        }
        
        // 初始化过程噪声协方差矩阵
        Q_ = MatrixXX::Zero();
        for (int i = 0; i < V_X && i < static_cast<int>(Q.size()); ++i) {
            Q_(i, i) = Q[i];
        }
        
        // 使用矩阵R
        default_R_ = R;
        
        // 初始化状态和协方差
        x_k1_ = MatrixX1::Zero();
        P_ = MatrixXX::Ones() * INF;
    }
    /**
     * @brief 构造函数
     * @param Q 过程噪声协方差向量（对角元素）
     * @param R 测量噪声协方差标量
     * @param predict_order 预测阶数（1-3）
     */
    Kalman(const std::vector<double>& Q, const double& R, const int& predict_order)
        : predict_order_(predict_order), t_(0.0) {
        
        // 初始化观测矩阵：只观测第一个状态（位置）
        H_ = MatrixZX::Zero();
        H_(0, 0) = 1;
        
        // 初始化过程噪声协方差矩阵
        Q_ = MatrixXX::Zero();
        for (int i = 0; i < V_X && i < static_cast<int>(Q.size()); ++i) {
            Q_(i, i) = Q[i];
        }
        
        // 初始化测量噪声协方差
        default_R_ = MatrixZZ::Zero();
        default_R_(0, 0) = R;
        
        // 初始化状态和协方差
        x_k1_ = MatrixX1::Zero();
        P_ = MatrixXX::Ones() * INF;
    }

    /**
     * @brief 获取当前状态估计
     * @return 状态向量
     */
    MatrixX1 get_x_k1() const {
        return x_k1_;
    }

    /**
     * @brief 重新初始化滤波器
     */
    void init() {
        x_k1_ = MatrixX1::Zero();
        P_ = MatrixXX::Ones() * INF;
        t_ = 0.0;
    }

    /**
     * @brief 设置状态的第一个分量（通常是位置）
     * @param x 位置值
     */
    void set_x(const double& x) {
        x_k1_(0, 0) = x;
    }

    /**
     * @brief 设置完整状态向量
     * @param x 状态向量
     */
    void set_x(const MatrixX1& x) {
        x_k1_ = x;
    }

    /**
     * @brief 设置当前时间戳
     * @param t 时间戳
     */
    void set_t(const double& t) {
        t_ = t;
    }

    /**
     * @brief 预测指定时刻的状态
     * @param t 目标时刻
     * @return 预测的状态向量
     * 
     * @details 根据运动模型进行状态外推：
     * - 1阶：x(t) = x
     * - 2阶：x(t) = x + v*dt
     * - 3阶：x(t) = x + v*dt + 0.5*a*dt²
     */
    MatrixX1 predict(const double& t) const {
        double dt = t - t_;
        MatrixXX A = build_transition_matrix(dt);
        return A * x_k1_;
    }

    /**
     * @brief 使用标量观测值更新滤波器（仅适用于V_Z=1的情况）
     * @param x 观测值
     * @param t 观测时刻
     */
    template<int VZ = V_Z>
    typename std::enable_if<VZ == 1, void>::type
    update(const double& x, const double& t) {
        MatrixZ1 z_k;
        z_k << x;
        update(z_k, t, default_R_);
    }

    /**
     * @brief 使用标量观测值和自定义噪声更新滤波器（仅适用于V_Z=1的情况）
     * @param x 观测值
     * @param t 观测时刻
     * @param R 测量噪声协方差
     */
    template<int VZ = V_Z>
    typename std::enable_if<VZ == 1, void>::type
    update(const double& x, const double& t, const double& R) {
        MatrixZ1 z_k;
        z_k << x;
        MatrixZZ R_mat;
        R_mat(0, 0) = R;
        update(z_k, t, R_mat);
    }

    /**
     * @brief 使用双标量观测值更新滤波器（仅适用于V_Z=2的情况）
     * @param x 第一个观测值
     * @param y 第二个观测值
     * @param t 观测时刻
     */
    template<int VZ = V_Z>
    typename std::enable_if<VZ == 2, void>::type
    update(const double& x, const double& y, const double& t) {
        MatrixZ1 z_k;
        z_k << x, y;
        update(z_k, t, default_R_);
    }

    /**
     * @brief 使用向量观测值更新滤波器（完整更新步骤）
     * @param z_k 观测向量
     * @param t 观测时刻
     * @param R 测量噪声协方差矩阵
     */
    void update(const MatrixZ1& z_k, const double& t, const MatrixZZ& R) {
        double dt = t - t_;
        
        // 预测步骤
        MatrixXX A = build_transition_matrix(dt);
        MatrixX1 x_pred = A * x_k1_;
        MatrixXX P_pred = A * P_ * A.transpose() + Q_;
        
        // 更新步骤
        MatrixXZ K = P_pred * H_.transpose() * 
                     (H_ * P_pred * H_.transpose() + R).inverse();
        
        x_k1_ = x_pred + K * (z_k - H_ * x_pred);
        P_ = (MatrixXX::Identity() - K * H_) * P_pred;
        t_ = t;
    }

    /**
     * @brief 使用向量观测值更新滤波器（使用默认噪声矩阵）
     * @param z_k 观测向量
     * @param t 观测时刻
     */
    void update(const MatrixZ1& z_k, const double& t) {
        update(z_k, t, default_R_);
    }

    /**
     * @brief 设置测量噪声协方差矩阵
     * @param R 新的测量噪声协方差矩阵
     */
    void setR(const MatrixZZ& R) {
        default_R_ = R;
    }

    /**
     * @brief 设置过程噪声协方差矩阵
     * @param Q 新的过程噪声协方差矩阵
     */
    void setQ(const MatrixXX& Q) {
        Q_ = Q;
    }

private:
    /**
     * @brief 构建状态转移矩阵
     * @param dt 时间间隔
     * @return 状态转移矩阵
     * 
     * @details 构建的矩阵形式：
     * - 对角线为1（状态保持）
     * - 上三角部分包含dt和dt²/2项（速度和加速度的影响）
     */
    MatrixXX build_transition_matrix(double dt) const {
        MatrixXX A = MatrixXX::Zero();
        
        // 对角线元素
        for (int i = 0; i < std::min(predict_order_, V_X); ++i) {
            A(i, i) = 1.0;
        }
        
        // 速度项
        for (int i = 1; i < std::min(predict_order_, V_X); ++i) {
            A(i - 1, i) = dt;
        }
        
        // 加速度项
        for (int i = 2; i < std::min(predict_order_, V_X); ++i) {
            A(i - 2, i) = 0.5 * dt * dt;
        }
        
        return A;
    }
};

} // namespace filter_lib

#endif // FILTER_LIB__KALMAN_HPP_