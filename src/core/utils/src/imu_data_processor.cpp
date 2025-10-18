#include "imu_data_processor.hpp"
#include <algorithm>
#include <limits>

namespace utils {

IMUDataProcessor::IMUDataProcessor(double buffer_time, double max_time_diff)
    : buffer_time_(buffer_time), max_time_diff_(max_time_diff) {}

void IMUDataProcessor::addIMUData(const Eigen::Quaterniond& quaternion, double timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    IMUData imu_data;
    imu_data.quaternion = quaternion;
    imu_data.timestamp = timestamp;
    
    // 将新数据添加到缓冲区末尾
    imu_buffer_.push_back(imu_data);
    
    // 自动清理过期数据，保持缓冲区大小合理
    cleanOldData();
}

void IMUDataProcessor::addEulerData(double roll, double pitch, double yaw, double timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!imu_buffer_.empty()) {
        // 从后向前查找最近的且未设置欧拉角的数据
        // 这种设计允许将欧拉角数据与已存在的四元数数据关联
        for (auto it = imu_buffer_.rbegin(); it != imu_buffer_.rend(); ++it) {
            double time_diff = std::abs(it->timestamp - timestamp);
            
            // 如果时间差在可接受范围内（10ms以内）
            // 这个阈值确保了数据的时间相关性
            if (time_diff < 0.01) {
                it->roll = roll;
                it->pitch = pitch;
                it->yaw = yaw;
                it->has_euler = true;
                break;
            }
        }
    } else {
        // 如果缓冲区为空，创建一个只包含欧拉角的数据条目
        // 这允许系统在没有四元数数据时仍能记录欧拉角信息
        IMUData imu_data;
        imu_data.timestamp = timestamp;
        imu_data.roll = roll;
        imu_data.pitch = pitch;
        imu_data.yaw = yaw;
        imu_data.has_euler = true;
        
        imu_buffer_.push_back(imu_data);
    }
    
    cleanOldData();
}

bool IMUDataProcessor::getSyncedIMUData(double target_time, IMUData& imu_data) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (imu_buffer_.empty()) {
        return false;
    }
    
    // 查找与目标时间最接近的数据
    double min_time_diff = std::numeric_limits<double>::max();
    auto best_it = imu_buffer_.end();
    
    // 遍历整个缓冲区，寻找时间差最小的数据点
    for (auto it = imu_buffer_.begin(); it != imu_buffer_.end(); ++it) {
        double time_diff = std::abs(it->timestamp - target_time);
        if (time_diff < min_time_diff) {
            min_time_diff = time_diff;
            best_it = it;
        }
    }
    
    // 如果找到了合适的数据（时间差在允许范围内）
    if (best_it != imu_buffer_.end() && min_time_diff < max_time_diff_) {
        imu_data = *best_it;
        return true;
    }
    
    // 如果没有找到时间差足够小的数据，但缓冲区不为空
    // 返回最新的数据作为备选方案
    if (!imu_buffer_.empty()) {
        imu_data = imu_buffer_.back();
        return true;
    }
    
    return false;
}

void IMUDataProcessor::cleanOldData() {
    if (imu_buffer_.empty()) {
        return;
    }
    
    // 基于最新数据的时间戳，计算需要保留的时间范围
    double current_time = imu_buffer_.back().timestamp;
    double cutoff_time = current_time - buffer_time_;
    
    // 从队列前端移除所有超过缓存时间的数据
    // 使用deque的pop_front保证了O(1)的删除效率
    while (!imu_buffer_.empty() && imu_buffer_.front().timestamp < cutoff_time) {
        imu_buffer_.pop_front();
    }
}

void IMUDataProcessor::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    imu_buffer_.clear();
}

size_t IMUDataProcessor::getBufferSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return imu_buffer_.size();
}

} // namespace utils