#ifndef UTILS__BLOCKING_QUEUE_HPP
#define UTILS__BLOCKING_QUEUE_HPP

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <utility>

namespace utils
{

template<typename T>
class BlockingQueue
{
public:
  explicit BlockingQueue(
    std::size_t max_size,
    std::function<void(void)> wait_handler = nullptr)
  : max_size_(max_size == 0 ? 1 : max_size),
    wait_handler_(std::move(wait_handler))
  {
  }

  void push(T value)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.size() >= max_size_) {
      if (wait_handler_) {
        lock.unlock();
        wait_handler_();
        lock.lock();
      }
      not_full_condition_.wait(lock);
    }

    queue_.push(std::move(value));

    lock.unlock();
    not_empty_condition_.notify_one();
  }

  void pop(T & out)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_condition_.wait(lock, [this]() { return !queue_.empty(); });

    out = std::move(queue_.front());
    queue_.pop();

    lock.unlock();
    not_full_condition_.notify_one();
  }

  T pop()
  {
    T value;
    pop(value);
    return value;
  }

  bool empty() const
  {
    std::scoped_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void clear()
  {
    std::scoped_lock<std::mutex> lock(mutex_);
    std::queue<T> empty_queue;
    std::swap(queue_, empty_queue);
    not_full_condition_.notify_all();
  }

private:

  const std::size_t max_size_;
  std::function<void(void)> wait_handler_;

  mutable std::mutex mutex_;
  std::condition_variable not_empty_condition_;
  std::condition_variable not_full_condition_;
  std::queue<T> queue_;
};

}  // namespace utils

#endif  // UTILS__BLOCKING_QUEUE_HPP
