/**
 * @file slide_weighted_avg.hpp
 * @brief 滑动窗口加权平均实现
 */

#ifndef FILTER_LIB__SLIDE_WEIGHTED_AVG_HPP_
#define FILTER_LIB__SLIDE_WEIGHTED_AVG_HPP_

#include <deque>
#include <cmath>
#include <algorithm>

namespace filter_lib {

/**
 * @brief 滑动窗口加权平均类
 * @tparam T 数据类型
 * 
 * @details 
 * 维护一个固定大小的滑动窗口，计算窗口内数据的加权平均值。
 * 当窗口满时，新数据会挤出最旧的数据。
 */
template<typename T>
class SlideWeightedAvg {
public:
    /**
     * @brief 默认构造函数，窗口大小为20
     */
    SlideWeightedAvg() : size_(20) {}
    
    /**
     * @brief 指定窗口大小的构造函数
     * @param size 窗口大小
     */
    explicit SlideWeightedAvg(int size) : size_(static_cast<size_t>(size)) {}
    
    /**
     * @brief 添加新的数据和权重
     * @param value 数据值
     * @param weight 权重
     */
    void push(T value, T weight) {
        // 如果窗口已满，移除最旧的数据
        if (values_.size() >= size_) {
            sum_ -= values_.front() * weights_.front();
            weight_sum_ -= weights_.front();
            values_.pop_front();
            weights_.pop_front();
        }
        
        // 添加新数据
        values_.push_back(value);
        weights_.push_back(weight);
        sum_ += value * weight;
        weight_sum_ += weight;
        
        // 更新平均值
        if (weight_sum_ > T(0)) {
            avg_ = sum_ / weight_sum_;
        }
    }
    
    /**
     * @brief 获取当前加权平均值
     * @return 加权平均值
     */
    T getAvg() const { return avg_; }
    
    /**
     * @brief 获取当前窗口中的数据个数
     * @return 数据个数
     */
    size_t getSize() const { return values_.size(); }
    
    /**
     * @brief 清空所有数据
     */
    void clear() {
        values_.clear();
        weights_.clear();
        sum_ = T(0);
        weight_sum_ = T(0);
        avg_ = T(0);
    }
    
    /**
     * @brief 检查是否为空
     * @return 是否为空
     */
    bool empty() const { return values_.empty(); }

private:
    std::deque<T> values_;   // 数据队列
    std::deque<T> weights_;  // 权重队列
    size_t size_;            // 窗口大小
    T sum_ = T(0);          // 加权和
    T weight_sum_ = T(0);   // 权重和
    T avg_ = T(0);          // 加权平均值
};

} // namespace filter_lib

#endif // FILTER_LIB__SLIDE_WEIGHTED_AVG_HPP_