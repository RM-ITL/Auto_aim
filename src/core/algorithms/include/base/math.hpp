/**
 * @file math_utils.hpp
 * @brief 滤波器库所需的数学工具函数
 * @details 从原始math库中提取的必要函数，专门为滤波器设计
 */

#ifndef FILTER_LIB__MATH__MATH_UTILS_HPP_
#define FILTER_LIB__MATH__MATH_UTILS_HPP_

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>

namespace filter_lib {
namespace math {

/**
 * @brief YPD坐标系表示
 * @details Yaw-Pitch-Distance球坐标系，常用于目标跟踪
 * 
 * 在机器人视觉中，这种表示方式直观地描述了目标相对于观察者的方位：
 * - yaw: 水平旋转角（左右）
 * - pitch: 垂直旋转角（上下）
 * - dis: 到目标的距离
 */
struct YpdCoord {
    double yaw = 0.0;      // 偏航角（弧度）
    double pitch = 0.0;    // 俯仰角（弧度）
    double dis = 0.0;      // 距离

    YpdCoord() = default;
    YpdCoord(double y, double p, double d) : yaw(y), pitch(p), dis(d) {}

    // 运算符重载，方便坐标运算
    YpdCoord operator+(const YpdCoord& other) const {
        return YpdCoord(yaw + other.yaw, pitch + other.pitch, dis + other.dis);
    }

    YpdCoord& operator+=(const YpdCoord& other) {
        yaw += other.yaw;
        pitch += other.pitch;
        dis += other.dis;
        return *this;
    }
};

// 输出流操作符，方便调试
inline std::ostream& operator<<(std::ostream& os, const YpdCoord& ypd) {
    os << "YPD(yaw=" << ypd.yaw << ", pitch=" << ypd.pitch << ", dis=" << ypd.dis << ")";
    return os;
}

/**
 * @brief 将角度限制在 [-π, π] 范围内
 * @param x 输入角度（弧度）
 * @return 规范化后的角度
 * 
 * @details 使用三角函数的方法确保结果在主值范围内
 * 例如：reduced_angle(3π) = -π
 */
inline double reduced_angle(double x) {
    return std::atan2(std::sin(x), std::cos(x));
}

/**
 * @brief 将实数限制在 [0, range] 范围内
 * @param x 输入值
 * @param range 范围（通常是2π）
 * @return 规范化后的值
 * 
 * @details 这个函数保证返回值在 [0, range] 区间内
 * 主要用于处理角度的周期性
 */
inline double reduced(double x, double range) {
    double times = range / (2.0 * M_PI);
    return times * (reduced_angle(x / times - M_PI) + M_PI);
}

/**
 * @brief 获取最接近目标值的等价角度
 * @param cur 当前角度
 * @param tar 目标角度
 * @param period 周期（通常是2π）
 * @return 最接近tar的cur的等价表示
 * 
 * @details 处理角度跳变问题
 * 例如：当tar=179°，cur=-179°时，返回181°而不是-179°
 * 这样避免了滤波器中出现大的跳变
 */
inline double get_closest(double cur, double tar, double period) {
    double reduced_cur = reduced(cur, period);
    double possibles[3] = {
        reduced_cur - period,
        reduced_cur,
        reduced_cur + period
    };
    
    double closest = possibles[0];
    double min_diff = std::fabs(tar - closest);
    
    for (int i = 1; i < 3; ++i) {
        double diff = std::fabs(tar - possibles[i]);
        if (diff < min_diff) {
            closest = possibles[i];
            min_diff = diff;
        }
    }
    
    return closest;
}

/**
 * @brief 笛卡尔坐标转YPD坐标
 * @param xyz 笛卡尔坐标向量
 * @return YPD坐标
 * 
 * @details 标准的笛卡尔到球坐标转换
 * 注意：返回的角度单位是弧度
 */
inline YpdCoord xyz_to_ypd(const Eigen::Vector3d& xyz) {
    YpdCoord ypd;
    double x = xyz(0), y = xyz(1), z = xyz(2);
    
    ypd.yaw = std::atan2(y, x);
    ypd.pitch = std::atan2(z, std::sqrt(x*x + y*y));
    ypd.dis = xyz.norm();
    
    return ypd;
}

/**
 * @brief YPD坐标转笛卡尔坐标
 * @param ypd YPD坐标
 * @return 笛卡尔坐标向量
 * 
 * @details 球坐标到笛卡尔坐标的标准转换
 */
inline Eigen::Vector3d ypd_to_xyz(const YpdCoord& ypd) {
    Eigen::Vector3d xyz;
    double cos_pitch = std::cos(ypd.pitch);
    
    xyz(0) = ypd.dis * cos_pitch * std::cos(ypd.yaw);
    xyz(1) = ypd.dis * cos_pitch * std::sin(ypd.yaw);
    xyz(2) = ypd.dis * std::sin(ypd.pitch);
    
    return xyz;
}

/**
 * @brief Ceres自动微分版本的笛卡尔转YPD
 * @tparam T 数据类型（double或ceres::Jet）
 * @param xyz 输入的笛卡尔坐标
 * @param ypd 输出的YPD坐标
 * 
 * @details 这个版本支持Ceres的自动微分
 * 在EKF中用于自动计算雅可比矩阵
 */
template<typename T>
void ceres_xyz_to_ypd(const T xyz[3], T ypd[3]) {
    // yaw = atan2(y, x)
    ypd[0] = ceres::atan2(xyz[1], xyz[0]);
    
    // pitch = atan2(z, sqrt(x^2 + y^2))
    T xy_norm = ceres::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
    ypd[1] = ceres::atan2(xyz[2], xy_norm);
    
    // distance = sqrt(x^2 + y^2 + z^2)
    ypd[2] = ceres::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);
}

/**
 * @brief 将向量转换为对角矩阵
 * @tparam N 矩阵维度
 * @param vec 输入向量
 * @return N×N对角矩阵
 * 
 * @details 用于构建噪声协方差矩阵
 * 例如：[1,2,3] → diag(1,2,3)
 */
template<int N>
Eigen::Matrix<double, N, N> vec_to_diagonal_matrix(const std::vector<double>& vec) {
    Eigen::Matrix<double, N, N> mat = Eigen::Matrix<double, N, N>::Zero();
    
    int size = std::min(N, static_cast<int>(vec.size()));
    for (int i = 0; i < size; ++i) {
        mat(i, i) = vec[i];
    }
    
    return mat;
}

/**
 * @brief 将向量转换为列矩阵
 * @tparam N 向量维度
 * @param vec 输入向量
 * @return N×1列矩阵
 * 
 * @details 用于将std::vector转换为Eigen列向量
 */
template<int N>
Eigen::Matrix<double, N, 1> vec_to_column_matrix(const std::vector<double>& vec) {
    Eigen::Matrix<double, N, 1> mat = Eigen::Matrix<double, N, 1>::Zero();
    
    int size = std::min(N, static_cast<int>(vec.size()));
    for (int i = 0; i < size; ++i) {
        mat(i, 0) = vec[i];
    }
    
    return mat;
}

} // namespace math
} // namespace filter_lib

#endif // FILTER_LIB__MATH__MATH_UTILS_HPP_