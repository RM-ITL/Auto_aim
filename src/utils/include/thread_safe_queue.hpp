#ifndef TOOLS__THREAD_SAFE_QUEUE_HPP
#define TOOLS__THREAD_SAFE_QUEUE_HPP

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>

namespace tools
{
template <typename T, bool PopWhenFull = false>
class ThreadSafeQueue
{
public:
  ThreadSafeQueue(
    size_t max_size, std::function<void(void)> full_handler = [] {})
  : max_size_(max_size), full_handler_(full_handler)
  {
  }

  /// 通知所有等待线程退出（配合 shutdown 使用）
  void notify_all()
  {
    not_empty_condition_.notify_all();
  }

  void push(const T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.size() >= max_size_) {
      if (PopWhenFull) {
        queue_.pop();
      } else {
        full_handler_();
        return;
      }
    }

    queue_.push(value);
    not_empty_condition_.notify_all();
  }

  void pop(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    not_empty_condition_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

    if (shutdown_ || queue_.empty()) {
      return;
    }

    value = queue_.front();
    queue_.pop();
  }

  T pop()
  {
    std::unique_lock<std::mutex> lock(mutex_);

    not_empty_condition_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

    if (shutdown_ || queue_.empty()) {
      return T{};
    }

    T value = std::move(queue_.front());
    queue_.pop();
    return std::move(value);
  }

  /// 带超时的 pop，超时返回 false
  bool try_pop(T & value, std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (!not_empty_condition_.wait_for(
          lock, timeout, [this] { return !queue_.empty() || shutdown_; })) {
      return false;
    }

    if (shutdown_ || queue_.empty()) {
      return false;
    }

    value = queue_.front();
    queue_.pop();
    return true;
  }

  T front()
  {
    std::unique_lock<std::mutex> lock(mutex_);

    not_empty_condition_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

    if (shutdown_ || queue_.empty()) {
      return T{};
    }

    return queue_.front();
  }

  void back(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.empty()) {
      std::cerr << "Error: Attempt to access the back of an empty queue." << std::endl;
      return;
    }

    value = queue_.back();
  }

  bool empty()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void clear()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
      queue_.pop();
    }
    not_empty_condition_.notify_all();
  }

  /// 标记为关闭，唤醒所有等待线程
  void shutdown()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    shutdown_ = true;
    not_empty_condition_.notify_all();
  }


private:
  std::queue<T> queue_;
  size_t max_size_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_condition_;
  std::function<void(void)> full_handler_;
  bool shutdown_{false};
};

}  // namespace tools

#endif  // TOOLS__THREAD_SAFE_QUEUE_HPP
