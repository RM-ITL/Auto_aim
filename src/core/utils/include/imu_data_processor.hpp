#ifndef UTILS_IMU_DATA_PROCESSOR_HPP
#define UTILS_IMU_DATA_PROCESSOR_HPP

#include <Eigen/Dense>
#include <deque>
#include <mutex>

namespace utils {

/**
 * @brief IMU数据结构，包含四元数和欧拉角信息
 */
struct IMUData {
    Eigen::Quaterniond quaternion;  // 四元数姿态表示
    double timestamp;                // 时间戳（秒）
    double roll;                     // 横滚角（弧度）
    double pitch;                    // 俯仰角（弧度）
    double yaw;                      // 偏航角（弧度）
    bool has_euler;                  // 标记是否包含欧拉角数据
    
    // 默认构造函数，初始化所有成员
    IMUData() : quaternion(Eigen::Quaterniond::Identity()),
                timestamp(0.0),
                roll(0.0),
                pitch(0.0),
                yaw(0.0),
                has_euler(false) {}
};

/**
 * @brief IMU数据处理器类
 * 
 * 该类用于管理和同步IMU数据，支持四元数和欧拉角两种姿态表示方式。
 * 主要功能包括：
 * - 缓存IMU数据并自动清理过期数据
 * - 时间同步，根据目标时间戳查找最接近的IMU数据
 * - 线程安全的数据访问
 */
class IMUDataProcessor {
public:
    /**
     * @brief 构造函数
     * @param buffer_time 数据缓存时间长度（秒），默认1.0秒
     * @param max_time_diff 时间同步的最大允许时间差（秒），默认0.05秒
     */
    IMUDataProcessor(double buffer_time = 1.0, double max_time_diff = 0.05);
    
    /**
     * @brief 添加IMU四元数数据
     * @param quaternion 姿态四元数
     * @param timestamp 数据时间戳（秒）
     */
    void addIMUData(const Eigen::Quaterniond& quaternion, double timestamp);
    
    /**
     * @brief 添加IMU欧拉角数据
     * @param roll 横滚角（弧度）
     * @param pitch 俯仰角（弧度）
     * @param yaw 偏航角（弧度）
     * @param timestamp 数据时间戳（秒）
     * 
     * @note 该函数会尝试将欧拉角数据与最近的四元数数据关联
     */
    void addEulerData(double roll, double pitch, double yaw, double timestamp);
    
    /**
     * @brief 获取与目标时间同步的IMU数据
     * @param target_time 目标时间戳（秒）
     * @param imu_data 输出参数，返回找到的IMU数据
     * @return true 如果成功找到数据，false 如果缓冲区为空或数据时间差过大
     */
    bool getSyncedIMUData(double target_time, IMUData& imu_data);
    
    /**
     * @brief 清理超出缓存时间的旧数据
     */
    void cleanOldData();
    
    /**
     * @brief 清空所有缓存数据
     */
    void clear();
    
    /**
     * @brief 获取当前缓冲区大小
     * @return 缓冲区中数据条目的数量
     */
    size_t getBufferSize() const;

private:
    std::deque<IMUData> imu_buffer_;  // IMU数据缓冲队列
    mutable std::mutex mutex_;         // 互斥锁，保证线程安全
    
    double buffer_time_;               // 缓存时间长度（秒）
    double max_time_diff_;             // 最大允许时间差（秒）
};

} // namespace utils

#endif // UTILS_IMU_DATA_PROCESSOR_HPP