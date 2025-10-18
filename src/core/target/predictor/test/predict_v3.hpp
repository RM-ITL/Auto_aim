#ifndef TARGET_PREDICTOR_NODE_HPP_
#define TARGET_PREDICTOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <autoaim_msgs/msg/targets.hpp>
#include <autoaim_msgs/msg/predictions.hpp>
#include <autoaim_msgs/msg/prediction.hpp>

// 使用 TrackQueueV3 替代 TrackQueueV4
#include "track_queue_v3.hpp"
#include "math.hpp"

#include <memory>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <deque>

namespace aimer {

/**
 * @brief 装甲板跟踪器类 - 使用TrackQueueV3版本
 * @details 为单个装甲板维护滤波状态，包含装甲板的类型信息
 *          内部统一使用毫米作为长度单位，避免频繁的单位转换
 *          
 * @note 核心改动：
 *       1. 使用TrackQueueV3替代TrackQueueV4
 *       2. V3使用笛卡尔坐标恒定加速度模型，更适合直线运动
 *       3. 仅使用位置观测，不使用角度信息
 *       4. V3需要至少2个观测点进行初始化
 */
class ArmorTracker {
public:
    // 单位转换常量 - 统一管理，避免魔数
    static constexpr double MM_TO_M = 0.001;
    static constexpr double M_TO_MM = 1000.0;
    
    /**
     * @brief 构造函数
     * @param armor_id 装甲板数字ID
     * @param armor_type 装甲板类型（big/small）
     * @param initial_position_mm 初始位置（毫米）
     * @param initial_time 初始时间戳
     */
    ArmorTracker(int armor_id,
                 const std::string& armor_type,
                 const Eigen::Vector3d& initial_position_mm, 
                 double initial_time);
    
    /**
     * @brief 更新装甲板状态
     * @param position_mm 新的位置观测值（毫米）
     * @param time 观测时间
     * 
     * @note V3版本使用仅位置观测模式，不需要角度信息
     */
    void update(const Eigen::Vector3d& position_mm, double time);
    
    /**
     * @brief 预测指定时间后的位置
     * @param predict_time 预测时间（秒）
     * @return 预测的3D位置（毫米）
     */
    Eigen::Vector3d predictPosition(double predict_time) const;
    
    /**
     * @brief 预测指定时间后的速度
     * @param predict_time 预测时间（秒）
     * @return 预测的3D速度（毫米/秒）
     * 
     * @note V3使用笛卡尔坐标速度模型，直接从状态向量提取vx,vy,vz
     */
    Eigen::Vector3d predictVelocity(double predict_time) const;
    
    /**
     * @brief 获取装甲板ID
     */
    int getArmorId() const { return armor_id_; }
    
    /**
     * @brief 获取装甲板类型
     */
    const std::string& getArmorType() const { return armor_type_; }
    
    /**
     * @brief 获取最后更新时间
     */
    double getLastUpdateTime() const { return last_update_time_; }
    
    /**
     * @brief 获取更新次数
     */
    int getUpdateCount() const { return update_count_; }
    
    /**
     * @brief 检查滤波器是否已初始化
     * @return 是否已完成初始化（V3需要至少2个观测点）
     */
    bool isInitialized() const { return filter_initialized_; }

private:
    // 成员变量
    int armor_id_;                              // 装甲板数字ID（1-5）
    std::string armor_type_;                    // 装甲板类型（big/small）
    filter_lib::TrackQueueV3 position_filter_; // V3滤波器（笛卡尔坐标恒定加速度模型）
    
    // 初始化相关
    std::deque<Eigen::Matrix<double, 5, 1>> init_buffer_;  // V3初始化缓冲区 [x,y,z,theta,time]
    bool filter_initialized_;                              // 滤波器是否已初始化
    
    // 时间管理
    double last_update_time_;                   // 最后更新时间
    int update_count_;                          // 更新计数
};

/**
 * @brief 目标预测节点 - 使用TrackQueueV3版本
 * @details 订阅PnP解算节点发布的目标数据，进行滤波和预测
 *          
 * @note 主要改动：
 *       1. 使用TrackQueueV3的笛卡尔坐标恒定加速度模型
 *       2. 仅使用位置观测，忽略角度信息
 *       3. V3的固定噪声参数，无需额外配置
 */
class TargetPredictorNode : public rclcpp::Node {
public:
    TargetPredictorNode();
    
private:
    /**
     * @brief 处理解算节点发布的目标数据
     * @param msg 包含多个目标的消息
     * 
     * @note 直接使用消息中的position字段（XYZ坐标）
     */
    void targetsCallback(const autoaim_msgs::msg::Targets::SharedPtr msg);
    
    /**
     * @brief 定时发布预测结果
     */
    void predictionTimerCallback();
    
    /**
     * @brief 清理超时的跟踪器
     * @param current_time 当前时间戳
     */
    void cleanupOldTrackers(double current_time);
    
    /**
     * @brief 从YAML文件加载配置参数
     * @param config_path 配置文件路径
     * @return 是否加载成功
     * 
     * @note V3版本使用固定噪声参数，主要保留跟踪管理参数
     */
    bool loadConfigFromYAML(const std::string& config_path);
    
    /**
     * @brief 将毫米单位的笛卡尔坐标转换为YPD坐标
     * @param position_mm 位置向量（毫米）
     * @return YPD坐标（弧度，毫米）
     * 
     * @note 这个函数仅用于将预测的笛卡尔位置转换为YPD用于输出
     */
    filter_lib::math::YpdCoord millimetersXyzToYpd(const Eigen::Vector3d& position_mm);
    
    /**
     * @brief 更新系统延迟的滑动平均值
     * @param new_latency 新的延迟测量值（秒）
     */
    void updateLatencyEstimate(double new_latency);
    
    // ROS2接口
    rclcpp::Subscription<autoaim_msgs::msg::Targets>::SharedPtr targets_sub_;
    rclcpp::Publisher<autoaim_msgs::msg::Predictions>::SharedPtr predictions_pub_;
    rclcpp::TimerBase::SharedPtr prediction_timer_;
    
    // 跟踪器管理 - 使用装甲板ID作为键
    std::unordered_map<int, std::unique_ptr<ArmorTracker>> trackers_;
    std::mutex trackers_mutex_;
    
    // 延迟估计相关
    std::deque<double> latency_history_;  // 延迟历史记录
    double current_prediction_time_;      // 当前使用的预测时间
    std::mutex latency_mutex_;
    
    // 配置参数结构体 - V3版本，去除了EKF噪声参数
    struct Config {
        // 预测参数
        double base_prediction_time = 0.005;    // 基础预测时间（秒）
        double latency_filter_alpha = 0.3;     // 延迟滤波系数
        int latency_history_size = 20;         // 延迟历史记录大小
        double max_prediction_time = 0.1;      // 最大预测时间限制（秒）
        double min_prediction_time = 0.01;     // 最小预测时间限制（秒）
        
        // 预测发布参数
        double prediction_interval = 0.01;     // 预测发布间隔（秒）
        
        // 跟踪器管理参数
        double tracker_timeout = 1.0;          // 跟踪器超时时间（秒）
        int min_update_count = 3;              // 开始预测前的最小更新次数
        int min_init_count = 2;                // V3初始化所需的最小观测点数（至少2个）
        
        // 配置文件路径
        std::string config_file_path;
    } config_;
};

} // namespace aimer

#endif // TARGET_PREDICTOR_NODE_HPP_