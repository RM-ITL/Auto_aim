/**
 * @file single_target_motion_model.hpp
 * @brief 单目标运动模型 - 基于4D观测的目标跟踪
 */

#ifndef SINGLE_TARGET_MOTION_MODEL_HPP_
#define SINGLE_TARGET_MOTION_MODEL_HPP_

#include "filter/track_queue_v3.hpp"
#include <Eigen/Dense>
#include <memory>
#include <deque>

namespace motion_model {

/**
 * @brief 目标观测数据
 */
struct Observation {
    Eigen::Vector3d position;  // 位置 [x, y, z]
    double orientation;        // 朝向角
    double timestamp;          // 时间戳
    
    Observation(const Eigen::Vector3d& pos, double orient, double time)
        : position(pos), orientation(orient), timestamp(time) {}
};

/**
 * @brief 目标预测结果
 */
struct Prediction {
    Eigen::Vector3d position;      // 预测位置
    Eigen::Vector3d velocity;      // 预测速度
    Eigen::Vector3d acceleration;  // 预测加速度
    double orientation;            // 预测朝向角
    double angular_velocity;       // 预测角速度
    double angular_acceleration;   // 预测角加速度
    bool valid;                   // 是否有效
    
    Prediction() : valid(false) {}
};

/**
 * @brief 跟踪器状态信息
 */
struct TrackerStatus {
    bool initialized;
    int update_count;
    double last_timestamp;
    double position_error;      // 位置估计误差
    double orientation_error;   // 朝向估计误差
    double speed;              // 当前速度
    double heading_rate;       // 当前转向率
};

/**
 * @brief 单目标运动模型
 */
class SingleTargetMotionModel {
public:
    /**
     * @brief 默认构造函数
     */
    SingleTargetMotionModel();
    
    /**
     * @brief 带参数构造函数
     * @param params 滤波器参数
     */
    explicit SingleTargetMotionModel(const filter_lib::TrackQueueV3::Parameters& params);
    
    /**
     * @brief 使用观测序列初始化
     * @param observations 初始观测序列（至少需要2个）
     * @return 初始化是否成功
     */
    bool initialize(const std::deque<Observation>& observations);
    
    /**
     * @brief 更新观测
     * @param obs 新的观测数据
     */
    void update(const Observation& obs);
    
    /**
     * @brief 预测未来状态
     * @param dt 预测时间间隔（秒）
     * @return 预测结果
     */
    Prediction predict(double dt) const;
    
    /**
     * @brief 获取当前状态
     * @return 当前位置、速度、朝向等完整状态
     */
    Prediction get_current_state() const;
    
    /**
     * @brief 获取跟踪器状态
     */
    TrackerStatus get_status() const;
    
    /**
     * @brief 检查是否已初始化
     */
    bool is_initialized() const { return initialized_; }
    
    /**
     * @brief 获取更新次数
     */
    int get_update_count() const { return update_count_; }
    
    /**
     * @brief 获取最后更新时间戳
     */
    double get_last_timestamp() const { return last_timestamp_; }
    
    /**
     * @brief 重置跟踪器
     */
    void reset();
    
    /**
     * @brief 设置滤波器参数
     */
    void set_parameters(const filter_lib::TrackQueueV3::Parameters& params);
    
    /**
     * @brief 获取滤波器参数
     */
    filter_lib::TrackQueueV3::Parameters get_parameters() const { 
        return filter_params_; 
    }
    
    /**
     * @brief 获取状态协方差矩阵（用于不确定性分析）
     */
    Eigen::Matrix<double, 11, 11> get_covariance() const;
    
    /**
     * @brief 设置过程噪声参数（便捷接口）
     */
    void set_process_noise(double pos, double vel, double acc, 
                          double angle, double omega, double alpha);
    
    /**
     * @brief 设置测量噪声参数（便捷接口）
     */
    void set_measurement_noise(double pos, double angle);

private:
    filter_lib::TrackQueueV3 filter_;                    // EKF滤波器
    filter_lib::TrackQueueV3::Parameters filter_params_; // 滤波器参数
    std::deque<filter_lib::TrackQueueV3::Observation> init_buffer_; // 初始化缓冲
    
    bool initialized_;       // 是否已初始化
    int update_count_;      // 更新计数
    double last_timestamp_; // 最后更新时间戳
    
    // 内部辅助函数
    filter_lib::TrackQueueV3::Observation convert_observation(const Observation& obs) const;
};

} // namespace motion_model

#endif // SINGLE_TARGET_MOTION_MODEL_HPP_