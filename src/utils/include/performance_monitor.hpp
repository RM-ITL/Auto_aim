#ifndef UTILS_PERFORMANCE_MONITOR_HPP
#define UTILS_PERFORMANCE_MONITOR_HPP

#include <chrono>
#include <atomic>
#include <string>
#include <mutex>
#include <memory>
#include <functional>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <rclcpp/rclcpp.hpp>

namespace utils {

/**
 * @brief 性能统计数据结构
 * 
 * 使用智能指针来管理原子变量，避免复制问题
 */
class PerformanceMetrics {
public:
    // 使用普通成员变量，通过互斥锁保护
    uint64_t total_count = 0;        // 总处理次数
    uint64_t success_count = 0;      // 成功处理次数
    uint64_t total_time_us = 0;      // 总处理时间（微秒）
    uint64_t min_time_us = UINT64_MAX;  // 最小处理时间
    uint64_t max_time_us = 0;        // 最大处理时间
    
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update;
    
    // 专门的互斥锁保护这个度量的数据
    mutable std::mutex data_mutex;
    
    PerformanceMetrics() {
        reset();
    }
    
    /**
     * @brief 重置所有统计数据
     */
    void reset() {
        std::lock_guard<std::mutex> lock(data_mutex);
        total_count = 0;
        success_count = 0;
        total_time_us = 0;
        min_time_us = UINT64_MAX;
        max_time_us = 0;
        start_time = std::chrono::steady_clock::now();
        last_update = start_time;
    }
    
    /**
     * @brief 记录一次操作
     */
    void record_operation(uint64_t time_us, bool success) {
        std::lock_guard<std::mutex> lock(data_mutex);
        total_count++;
        if (success) {
            success_count++;
        }
        total_time_us += time_us;
        
        if (time_us < min_time_us) {
            min_time_us = time_us;
        }
        if (time_us > max_time_us) {
            max_time_us = time_us;
        }
        
        last_update = std::chrono::steady_clock::now();
    }
    
    /**
     * @brief 计算成功率
     * @return 成功率百分比 (0-100)
     */
    double get_success_rate() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        return total_count > 0 ? 
            static_cast<double>(success_count) / total_count * 100.0 : 0.0;
    }
    
    /**
     * @brief 计算平均处理时间
     * @return 平均处理时间（毫秒）
     */
    double get_avg_time_ms() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        return total_count > 0 ? 
            static_cast<double>(total_time_us) / total_count / 1000.0 : 0.0;
    }
    
    /**
     * @brief 计算当前FPS（每秒处理次数）
     * @return FPS值
     */
    double get_current_fps() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - start_time).count();
        return elapsed > 0 ? 
            static_cast<double>(total_count) / elapsed : 0.0;
    }
    
    /**
     * @brief 获取统计数据的快照（用于避免长时间锁定）
     */
    struct Snapshot {
        uint64_t total_count;
        uint64_t success_count;
        uint64_t total_time_us;
        uint64_t min_time_us;
        uint64_t max_time_us;
        std::chrono::steady_clock::time_point start_time;
    };
    
    Snapshot get_snapshot() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        return {total_count, success_count, total_time_us, 
                min_time_us, max_time_us, start_time};
    }
};

/**
 * @brief 性能监控器类
 * 
 * 提供通用的性能监控功能，使用智能指针管理度量对象
 */
class PerformanceMonitor {
public:
    /**
     * @brief 性能监控器配置结构
     */
    struct Config {
        bool enable_logging;           // 是否启用日志输出
        double print_interval_sec;     // 日志输出间隔（秒）
        std::string logger_name;        // 日志名称
        bool enable_detailed_stats;    // 是否输出详细统计
        
        // 提供默认构造函数
        Config() 
            : enable_logging(true),
              print_interval_sec(5.0),
              logger_name("utils.PerformanceMonitor"),
              enable_detailed_stats(false) {}
    };
    
    /**
     * @brief 构造函数
     * @param config 配置参数
     */
    explicit PerformanceMonitor(const Config& config = Config())
        : config_(config)
        , last_print_time_(std::chrono::steady_clock::now()) {}
    
    /**
     * @brief 注册一个新的监控指标
     * @param name 指标名称
     */
    void register_metric(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (metrics_.find(name) == metrics_.end()) {
            // 使用智能指针创建新的度量对象
            metrics_[name] = std::make_shared<PerformanceMetrics>();
        }
    }
    
    /**
     * @brief RAII风格的作用域计时器
     */
    class ScopedTimer {
    public:
        ScopedTimer(PerformanceMonitor& monitor, 
                   const std::string& metric_name,
                   bool success = true)
            : monitor_(monitor)
            , metric_name_(metric_name)
            , success_(success)
            , start_time_(std::chrono::steady_clock::now()) {}
        
        ~ScopedTimer() {
            auto end_time = std::chrono::steady_clock::now();
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time_).count();
            monitor_.record(metric_name_, duration_us, success_);
        }
        
        void set_success(bool success) { success_ = success; }
        
    private:
        PerformanceMonitor& monitor_;
        std::string metric_name_;
        bool success_;
        std::chrono::steady_clock::time_point start_time_;
    };
    
    /**
     * @brief 创建一个作用域计时器
     */
    ScopedTimer create_timer(const std::string& metric_name, bool success = true) {
        return ScopedTimer(*this, metric_name, success);
    }
    
    /**
     * @brief 手动记录一次操作
     */
    void record(const std::string& metric_name, 
                uint64_t time_us, 
                bool success = true) {
        std::shared_ptr<PerformanceMetrics> metric;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // 自动注册未知的指标
            if (metrics_.find(metric_name) == metrics_.end()) {
                metrics_[metric_name] = std::make_shared<PerformanceMetrics>();
            }
            
            metric = metrics_[metric_name];
        }
        
        // 在锁外部进行实际的记录操作
        metric->record_operation(time_us, success);
        
        // 检查是否需要打印统计信息
        if (config_.enable_logging) {
            check_and_print_stats();
        }
    }
    
    /**
     * @brief 获取指定指标的统计信息快照
     */
    PerformanceMetrics::Snapshot get_metrics_snapshot(const std::string& metric_name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metrics_.find(metric_name);
        if (it != metrics_.end() && it->second) {
            return it->second->get_snapshot();
        }
        // 返回默认值
        return PerformanceMetrics().get_snapshot();
    }
    
    /**
     * @brief 获取指定指标（用于简单的FPS查询）
     * 
     * 专门为FPS查询优化的接口
     */
    struct SimpleMetrics {
        double fps;
        uint64_t total_count;
        double avg_time_ms;
    };
    
    SimpleMetrics get_metrics(const std::string& metric_name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = metrics_.find(metric_name);
        if (it != metrics_.end() && it->second) {
            auto snapshot = it->second->get_snapshot();
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - snapshot.start_time).count();
            
            double fps = elapsed > 0 ? 
                static_cast<double>(snapshot.total_count) / elapsed : 0.0;
            
            double avg_ms = snapshot.total_count > 0 ? 
                static_cast<double>(snapshot.total_time_us) / snapshot.total_count / 1000.0 : 0.0;
            
            return {fps, snapshot.total_count, avg_ms};
        }
        return {0.0, 0, 0.0};
    }
    
    /**
     * @brief 重置指定指标
     */
    void reset_metric(const std::string& metric_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (metrics_.find(metric_name) != metrics_.end() && metrics_[metric_name]) {
            metrics_[metric_name]->reset();
        }
    }
    
    /**
     * @brief 重置所有指标
     */
    void reset_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [name, metric] : metrics_) {
            if (metric) {
                metric->reset();
            }
        }
    }
    
    /**
     * @brief 手动触发统计信息输出
     */
    void print_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (metrics_.empty()) {
            return;
        }
        
        std::stringstream ss;
        ss << "\n========== 性能统计 ==========\n";
        
        for (const auto& [name, metric] : metrics_) {
            if (!metric) continue;
            
            auto snapshot = metric->get_snapshot();
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                now - snapshot.start_time).count();
            double fps = elapsed > 0 ? 
                static_cast<double>(snapshot.total_count) / elapsed : 0.0;
            
            double success_rate = snapshot.total_count > 0 ? 
                static_cast<double>(snapshot.success_count) / snapshot.total_count * 100.0 : 0.0;
            
            double avg_ms = snapshot.total_count > 0 ? 
                static_cast<double>(snapshot.total_time_us) / snapshot.total_count / 1000.0 : 0.0;
            
            ss << name << ": ";
            ss << "总数=" << snapshot.total_count;
            ss << ", 成功率=" << std::fixed << std::setprecision(1) << success_rate << "%";
            ss << ", 平均=" << std::fixed << std::setprecision(2) << avg_ms << "ms";
            
            if (config_.enable_detailed_stats) {
                ss << ", 最小=" << std::fixed << std::setprecision(2) 
                   << snapshot.min_time_us / 1000.0 << "ms";
                ss << ", 最大=" << std::fixed << std::setprecision(2) 
                   << snapshot.max_time_us / 1000.0 << "ms";
            }
            
            ss << ", FPS=" << std::fixed << std::setprecision(1) << fps;
            ss << "\n";
        }
        
        RCLCPP_INFO(rclcpp::get_logger(config_.logger_name), "%s", ss.str().c_str());
    }
    
    /**
     * @brief 设置配置
     */
    void set_config(const Config& config) {
        config_ = config;
    }
    
private:
    /**
     * @brief 检查并根据配置输出统计信息
     */
    void check_and_print_stats() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_print_time_).count();
        
        if (elapsed >= config_.print_interval_sec) {
            print_stats();
            last_print_time_ = now;
        }
    }
    
    Config config_;
    mutable std::mutex mutex_;
    // 使用智能指针避免复制问题
    std::unordered_map<std::string, std::shared_ptr<PerformanceMetrics>> metrics_;
    std::chrono::steady_clock::time_point last_print_time_;
};

} // namespace utils

#endif // UTILS_PERFORMANCE_MONITOR_HPP