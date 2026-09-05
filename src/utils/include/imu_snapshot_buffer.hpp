#ifndef TOOLS__IMU_SNAPSHOT_BUFFER_HPP
#define TOOLS__IMU_SNAPSHOT_BUFFER_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

namespace tools
{

// 线程安全、非破坏性的 IMU 姿态环形缓冲 + 二分插值。
//
// 动机（对标 Alliance rmcs_auto_aim_v2 的 ImuSnapshotBuffer）：
// 取代旧的「破坏性 pop/front + 200 次循环 + 超时注入 identity」查询——那套设计
// 会静默返回零姿态(identity)污染 EKF，或在 IMU 迟到时阻塞查询线程。
//
// 本缓冲：
//   push(q,t)          写入（生产者线程，如串口 read_thread），保持时间单调、越界丢弃乱序样本
//   interpolate(t)     只读二分查找相邻两样本 slerp；不破坏缓冲、从不阻塞
//                       - 命中区间          → 插值成功
//                       - 略超出区间(≤clamp) → clamp 到最近端(零阶保持)，成功
//                       - 大幅超出/缓冲为空   → 返回 std::nullopt（让上层丢帧/告警）
// 失败以 std::optional 表达，调用方必须区分成功/失败——从接口上消除「静默 identity」。
class ImuSnapshotBuffer
{
public:
  // capacity: 环形缓冲深度；clamp_tolerance_s: 允许零阶保持外推的最大间隙(秒)。
  explicit ImuSnapshotBuffer(std::size_t capacity = 1000, double clamp_tolerance_s = 0.010)
  : capacity_(capacity == 0 ? 1 : capacity), clamp_tolerance_s_(clamp_tolerance_s)
  {
  }

  void push(const Eigen::Quaterniond & q, std::chrono::steady_clock::time_point t)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // 时间必须严格递增；时钟回退/重复戳的乱序样本直接丢弃，保证 buf_ 有序可二分。
    if (!buf_.empty() && t <= buf_.back().t) return;
    buf_.push_back({q.normalized(), t});
    if (buf_.size() > capacity_) buf_.pop_front();
  }

  // 插值到时刻 t；无有效包围/clamp 样本时返回 nullopt。
  std::optional<Eigen::Quaterniond> interpolate(std::chrono::steady_clock::time_point t) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buf_.empty()) return std::nullopt;

    const Sample & oldest = buf_.front();
    const Sample & newest = buf_.back();

    if (t >= oldest.t && t <= newest.t) {
      // 区间内：二分找第一个时间 > t 的样本，与其前一个构成包围对。
      auto it = std::upper_bound(
        buf_.begin(), buf_.end(), t,
        [](std::chrono::steady_clock::time_point tt, const Sample & s) { return tt < s.t; });
      if (it == buf_.end()) return newest.q;  // t == newest.t
      const Sample & b = *it;
      const Sample & a = *(it - 1);
      double denom = sec(b.t - a.t);
      double k = denom > 1e-9 ? sec(t - a.t) / denom : 0.0;
      k = std::clamp(k, 0.0, 1.0);
      return a.q.slerp(k, b.q).normalized();
    }

    // 区间外：小间隙 clamp 到最近端(ZOH)，大间隙判失败。
    if (t > newest.t) {
      return sec(t - newest.t) <= clamp_tolerance_s_
               ? std::optional<Eigen::Quaterniond>(newest.q)
               : std::nullopt;
    }
    return sec(oldest.t - t) <= clamp_tolerance_s_
             ? std::optional<Eigen::Quaterniond>(oldest.q)
             : std::nullopt;
  }

  // 遗留接口兜底：返回最近一条样本；缓冲为空才给 identity（仅开机瞬间）。
  Eigen::Quaterniond newest_or_identity() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return buf_.empty() ? Eigen::Quaterniond::Identity() : buf_.back().q;
  }

  bool empty() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return buf_.empty();
  }

private:
  struct Sample
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point t;
  };

  static double sec(std::chrono::steady_clock::duration d)
  {
    return std::chrono::duration<double>(d).count();
  }

  mutable std::mutex mutex_;
  std::deque<Sample> buf_;
  std::size_t capacity_;
  double clamp_tolerance_s_;
};

}  // namespace tools

#endif  // TOOLS__IMU_SNAPSHOT_BUFFER_HPP
