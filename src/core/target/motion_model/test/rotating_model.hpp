/**
 * @file rotating_motion_model.hpp
 * @brief 旋转目标运动模型 - 单目标旋转运动跟踪
 * @details 封装RotatingTargetTracker，支持装甲板跟随和中心打击两种模式
 */

#ifndef ROTATING_MOTION_MODEL_HPP_
#define ROTATING_MOTION_MODEL_HPP_

#include "filter/rotating_target_tracker.hpp"
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <deque>

namespace motion_model {

/**
 * @brief 旋转目标观测数据
 */
struct RotatingObservation {
    Eigen::Vector3d position;  // 装甲板位置
    double orientation;        // 装甲板朝向角
    double timestamp;          // 时间戳
    
    RotatingObservation(const Eigen::Vector3d& pos, double orient, double time)
        : position(pos), orientation(orient), timestamp(time) {}
};

/**
 * @brief 旋转目标预测结果
 */
struct RotatingPrediction {
    // 装甲板预测
    Eigen::Vector3d armor_position;
    double armor_orientation;
    
    // 中心预测
    Eigen::Vector3d center_position;
    Eigen::Vector3d center_velocity;
    
    // 旋转参数
    double angular_velocity;
    double radius;
    int current_armor;         // 当前跟踪的装甲板索引
    
    // 开火判断
    bool armor_fire_ready;
    bool center_fire_ready;
    
    bool valid;
    
    RotatingPrediction() : valid(false) {}
};

/**
 * @brief 旋转运动模型 - 管理单个旋转目标
 */
class RotatingMotionModel {
public:
    /**
     * @brief 模型参数
     */
    struct Params {
        int armor_count;              // 装甲板数量
        double radius_min;            // 最小半径（米）
        double radius_max;            // 最大半径（米）
        double timeout;               // 超时时间（秒）
        bool enable_height_filter;    // 高度滤波
        
        // 开火参数
        int min_fire_updates;         // 最小更新次数
        double fire_delay;            // 开火延迟（秒）
        double armor_angle_thr;       // 装甲板角度阈值（弧度）
        double center_angle_thr;      // 中心角度阈值（弧度）
        
        // 构造函数，设置默认值
        Params() :
            armor_count(4),
            radius_min(0.15),
            radius_max(0.40),
            timeout(1.0),
            enable_height_filter(false),
            min_fire_updates(20),
            fire_delay(0.1),
            armor_angle_thr(0.3),
            center_angle_thr(0.2) {}
    };
    
    /**
     * @brief 构造函数
     * @param params 模型参数
     */
    explicit RotatingMotionModel(const Params& params = Params());
    
    /**
     * @brief 更新观测
     * @param obs 新的观测数据
     */
    void update(const RotatingObservation& obs);
    
    /**
     * @brief 获取预测结果
     * @param dt 预测时间间隔（秒）
     * @return 包含装甲板和中心两种模式的预测
     */
    RotatingPrediction predict(double dt) const;
    
    /**
     * @brief 获取状态信息
     * @return 状态描述字符串列表
     */
    std::vector<std::string> status() const;
    
    /**
     * @brief 检查是否已初始化
     * @return 初始化状态
     */
    bool initialized() const { return initialized_; }
    
    /**
     * @brief 获取更新次数
     * @return 更新计数
     */
    int update_count() const { return update_count_; }
    
    /**
     * @brief 获取最后更新时间
     * @return 时间戳（秒）
     */
    double last_timestamp() const { return last_timestamp_; }
    
    /**
     * @brief 检查是否超时
     * @param current_time 当前时间戳
     * @return 是否超时
     */
    bool is_timeout(double current_time) const;
    
    /**
     * @brief 重置模型
     */
    void reset();
    
    /**
     * @brief 设置参数
     * @param params 新的参数
     */
    void set_params(const Params& params);
    
    /**
     * @brief 设置过程噪声
     */
    void set_process_noise(double pos, double vel, double angle, double omega);
    
    /**
     * @brief 设置测量噪声
     */
    void set_measurement_noise(double pos, double angle);

private:
    Params params_;
    std::unique_ptr<filter_lib::RotatingTargetTracker> tracker_;
    std::deque<filter_lib::RotatingTargetTracker::Observation> init_buffer_;
    
    bool initialized_;
    int update_count_;
    double last_timestamp_;
};

} // namespace motion_model

#endif // ROTATING_MOTION_MODEL_HPP_