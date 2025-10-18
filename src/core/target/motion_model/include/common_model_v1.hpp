/**
 * @file test_model.hpp
 * @brief 解耦式运动模型 - 基于DecoupledTracker的目标跟踪模型
 * @details 
 * 这个模型通过分离平移和旋转运动来提高跟踪稳定性。
 * 主要特点：
 * 1. 避免朝向角测量噪声污染速度估计
 * 2. 自适应耦合机制处理不同运动模式
 * 3. 提供详细的运动状态分析
 */

#ifndef TEST_MODEL_HPP_
#define TEST_MODEL_HPP_

#include "filter/decoupled_tracker.hpp"
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <chrono>
#include <vector>
#include <map>

namespace motion_model {

/**
 * @brief 目标观测数据 - 包含位置和朝向信息
 */
struct DecoupledObservation {
    Eigen::Vector3d position;       // 目标位置 [x, y, z] (单位：米)
    double orientation;             // 目标朝向角 (单位：弧度)
    double timestamp;               // 时间戳 (单位：秒)
    
    // 可选的额外信息
    double position_confidence = 1.0;    // 位置测量置信度 [0,1]
    double orientation_confidence = 1.0; // 朝向测量置信度 [0,1]
    
    // 构造函数
    DecoupledObservation(const Eigen::Vector3d& pos, double orient, double time)
        : position(pos), orientation(orient), timestamp(time) {}
    
    DecoupledObservation(const Eigen::Vector3d& pos, double orient, double time,
                        double pos_conf, double orient_conf)
        : position(pos), orientation(orient), timestamp(time),
          position_confidence(pos_conf), orientation_confidence(orient_conf) {}
};

/**
 * @brief 解耦式预测结果 - 包含分离的平移和旋转信息
 */
struct DecoupledPrediction {
    // 位置预测
    Eigen::Vector3d position;           // 预测的位置 (米)
    Eigen::Vector3d velocity;           // 预测的速度 (米/秒)
    Eigen::Vector3d acceleration;       // 预测的加速度 (米/秒²)
    
    // 姿态预测
    double orientation;                 // 预测的朝向角 (弧度)
    double angular_velocity;            // 预测的角速度 (弧度/秒)
    double angular_acceleration;        // 预测的角加速度 (弧度/秒²)
    
    // 运动耦合信息
    double coupling_strength;           // 耦合强度 ρ [0,1]
    double motion_consistency;          // 速度方向与朝向的一致性 [-1,1]
    double velocity_angle;              // 速度方向角 (弧度)
    double heading_velocity_diff;       // 朝向与速度方向的差值 (弧度)
    
    // 运动模式
    std::string motion_mode;            // 运动模式描述
    double speed;                       // 速度大小 (米/秒)
    
    // 不确定性估计
    Eigen::Vector3d position_std;       // 位置标准差 (米)
    Eigen::Vector3d velocity_std;       // 速度标准差 (米/秒)
    double orientation_std;             // 朝向角标准差 (弧度)
    double angular_velocity_std;        // 角速度标准差 (弧度/秒)
    
    // 预测质量
    bool is_valid;                      // 预测是否有效
    double prediction_confidence;       // 预测置信度 [0,1]
    
    DecoupledPrediction() 
        : is_valid(false), prediction_confidence(0.0),
          coupling_strength(0.0), motion_consistency(0.0) {}
};

/**
 * @brief 解耦式运动模型 - 管理基于DecoupledTracker的目标跟踪
 * 
 * 这个模型的核心优势：
 * 1. 分离处理平移和旋转，提高稳定性
 * 2. 自适应耦合机制，智能处理不同运动模式
 * 3. 提供丰富的运动分析信息
 */
class TestModel {
public:
    /**
     * @brief 构造函数
     * @param model_name 模型名称（用于日志）
     * @param init_threshold 初始化所需的最小观测数量
     * @param tracker_timeout 跟踪器超时时间 (秒)
     * @param enable_coupling 是否启用耦合机制
     */
    TestModel(const std::string& model_name = "DecoupledModel",
              size_t init_threshold = 3,
              double tracker_timeout = 1.0,
              bool enable_coupling = true);
    
    /**
     * @brief 更新目标观测数据
     * @param observation 新的观测数据
     * @return 是否成功更新（false表示数据被拒绝）
     */
    bool updateObservation(const DecoupledObservation& observation);
    
    /**
     * @brief 获取目标的预测结果
     * @param prediction_time 预测时间差 (秒)
     * @return 预测结果，包含详细的运动状态信息
     */
    DecoupledPrediction getPrediction(double prediction_time) const;
    
    /**
     * @brief 批量更新观测数据
     * @param observations 观测数据序列
     * @param force_init 是否强制重新初始化
     * @return 成功处理的观测数量
     */
    int batchUpdate(const std::vector<DecoupledObservation>& observations,
                    bool force_init = false);
    
    /**
     * @brief 判断模型是否已初始化
     * @return true表示已初始化，false表示未初始化
     */
    bool isInitialized() const { return is_initialized_; }
    
    /**
     * @brief 获取模型的详细状态信息
     * @param info 输出的状态信息字符串向量
     */
    void getStatusInfo(std::vector<std::string>& info) const;
    
    /**
     * @brief 设置滤波器参数
     * @param params DecoupledTracker的参数结构体
     */
    void setFilterParameters(const filter_lib::DecoupledTracker::Parameters& params);
    
    /**
     * @brief 设置耦合参数
     * @param coupling_sigma 耦合强度的尺度参数
     * @param min_speed 最小速度阈值
     * @param rho_threshold 应用耦合的阈值
     */
    void setCouplingParameters(double coupling_sigma, double min_speed, 
                              double rho_threshold);
    
    /**
     * @brief 设置噪声自适应参数
     * @param enable_adaptive 是否启用自适应噪声
     * @param confidence_scale 置信度对噪声的影响系数
     */
    void setAdaptiveNoiseParameters(bool enable_adaptive, 
                                   double confidence_scale = 1.0);
    
    /**
     * @brief 手动重置模型
     */
    void reset();
    
    /**
     * @brief 获取更新次数
     * @return 从初始化或重置以来的更新次数
     */
    int getUpdateCount() const { return update_count_; }
    
    /**
     * @brief 获取最后更新时间
     * @return 最后一次更新的时间戳（秒）
     */
    double getLastUpdateTime() const { return last_update_time_; }
    
    /**
     * @brief 检查是否超时
     * @param current_time 当前时间戳（秒）
     * @return true表示已超时，false表示正常
     */
    bool isTimeout(double current_time) const;
    
    /**
     * @brief 获取运动分析报告
     * @return 包含运动特征分析的文本报告
     */
    std::string getMotionAnalysisReport() const;
    
    /**
     * @brief 设置日志级别
     * @param level 0=无日志, 1=基本信息, 2=详细调试
     */
    void setLogLevel(int level) { log_level_ = level; }

private:
    /**
     * @brief 将观测数据转换为滤波器格式
     */
    filter_lib::DecoupledTracker::MeasVec 
    observationToMeasurement(const DecoupledObservation& obs) const;
    
    /**
     * @brief 将时间戳转换为时间点
     */
    std::chrono::steady_clock::time_point 
    timestampToTimePoint(double timestamp) const;
    
    /**
     * @brief 检查观测数据的有效性
     */
    bool isObservationValid(const DecoupledObservation& obs) const;
    
    /**
     * @brief 计算预测置信度
     */
    double calculatePredictionConfidence(double prediction_time) const;
    
    /**
     * @brief 收集初始化数据
     */
    void collectInitData(const DecoupledObservation& obs);
    
    /**
     * @brief 尝试初始化滤波器
     */
    bool tryInitialize();

private:
    // 模型标识
    std::string model_name_;
    
    // 滤波器对象
    std::unique_ptr<filter_lib::DecoupledTracker> tracker_;
    filter_lib::DecoupledTracker::Parameters tracker_params_;
    
    // 状态信息
    bool is_initialized_;
    int update_count_;
    double last_update_time_;
    size_t init_threshold_;
    double tracker_timeout_;
    bool enable_coupling_;
    
    // 初始化数据缓存
    std::deque<Eigen::Matrix<double, 5, 1>> init_observations_;
    
    // 噪声自适应
    bool enable_adaptive_noise_;
    double confidence_scale_;
    
    // 日志级别
    int log_level_;
    
    // 运动历史（用于分析）
    struct MotionSnapshot {
        double timestamp;
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        double orientation;
        double angular_velocity;
        double coupling_strength;
        std::string motion_mode;
    };
    std::deque<MotionSnapshot> motion_history_;
    static constexpr size_t MAX_HISTORY_SIZE = 100;
};

} // namespace motion_model

#endif // TEST_MODEL_HPP_