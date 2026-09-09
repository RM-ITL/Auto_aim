#ifndef TOOLS__DEVICE_CLOCK_MAPPER_HPP
#define TOOLS__DEVICE_CLOCK_MAPPER_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>

namespace tools
{

// 相机硬件时间戳(设备时钟域，硬件锁存、无 USB/线程抖动) → 主机 steady_clock 域。
//
// 动机：相机驱动原本在采集线程 GetOneFrame 返回后才 steady_clock::now() 打戳，
// 戳里裹了「读出 + USB 传输 + 线程调度/sleep」的抖动，且相对曝光时刻整体偏晚。
// 本映射器用相机每帧硬件锁存的 dev_tick 拟合到主机时钟，剥离逐帧抖动，
// 让上层减去半个曝光即可得到「曝光中点」这一物理锚点。
//
// 模型：  host ≈ skew·(dev_tick - dev_ref) + offset
//   skew  ：窗口最小二乘斜率(中心化，数值稳定)。吸收「未知 tick 频率」+「晶振相对漂移」，
//           不受恒定传输延迟影响(常延迟只整体抬高截距，不改斜率)。
//   offset：窗口内 (host - skew·x) 的【最小值】(NTP 式 min-filter)。挑出传输延迟最小的那帧
//           定截距，把逐帧抖动尖峰全部丢弃 → 映射结果贴住延迟下包络，最接近真实曝光时刻。
//
// 失败以 std::optional 表达(未收敛 / dev_tick 非单调 / 拟合退化)，调用方须回退到 now()。
// tick 单位无关：skew 自拟合频率，不论 ns 还是 0.1ms/tick 都成立，只要求单调递增。
//
// 线程约束：仅在单个采集线程内调用 update()/reset()，故【无锁】。切勿跨线程共享实例。
class DeviceClockMapper
{
public:
  // window：拟合窗口帧数(128 帧 @200fps ≈ 0.64s，足够跟漂移又不过度平滑)。
  // min_samples：达到此样本数前一律返回 nullopt(开机收敛期回退 now)。
  explicit DeviceClockMapper(std::size_t window = 128, std::size_t min_samples = 16)
  : window_(window == 0 ? 1 : window), min_samples_(min_samples)
  {
  }

  // 每帧调用：dev_tick=相机硬件时间戳(拼好的 64 位)，host_recv=该帧收帧时的 now()。
  // 返回映射到主机域的时刻(≈曝光结束附近，未减曝光半段)；未收敛/异常返回 nullopt。
  std::optional<std::chrono::steady_clock::time_point> update(
    uint64_t dev_tick, std::chrono::steady_clock::time_point host_recv)
  {
    // 单调性处理，区分两种非递增：
    //  - 重复/轻微倒退(SDK 偶发抖动)：仅丢弃该帧，保留已收敛窗口，避免无谓重收敛期回退 now()。
    //  - 大幅倒退(相机拔插/USB reset/计数器归零)：clear() 重锚，旧参考已失效。
    if (have_ref_ && dev_tick <= last_dev_) {
      // dev_ref_ 之后的相对量恒正；dev_tick 落到参考点之前视为大幅倒退。
      if (dev_tick < dev_ref_) {
        reset();
      } else {
        return std::nullopt;  // 非递增但未越过锚点：跳过此帧，窗口不动
      }
    }

    if (!have_ref_) {
      have_ref_ = true;
      dev_ref_ = dev_tick;
      host_ref_ = host_recv;
      last_dev_ = dev_tick;
      buf_.push_back({0.0, 0.0});  // 首帧 x=0,y=0
      return std::nullopt;
    }

    last_dev_ = dev_tick;
    // x：相对首帧 tick(避免大数吃精度)；y：相对首帧的主机秒(host_ref_ 用 time_point 存，此处只算差)。
    double x = static_cast<double>(dev_tick - dev_ref_);
    double y = std::chrono::duration<double>(host_recv - host_ref_).count();
    buf_.push_back({x, y});
    if (buf_.size() > window_) buf_.pop_front();
    if (buf_.size() < min_samples_) return std::nullopt;

    // 中心化最小二乘斜率：skew = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²。
    // 中心化后分母是正平方和(无大数相减)，规避原始正规方程 (n·Sxx-Sx²) 的灾难性抵消。
    const double n = static_cast<double>(buf_.size());
    double sx = 0.0, sy = 0.0;
    for (const auto & p : buf_) { sx += p.x; sy += p.y; }
    const double mx = sx / n, my = sy / n;

    double sxx = 0.0, sxy = 0.0;
    for (const auto & p : buf_) {
      const double dx = p.x - mx;
      sxx += dx * dx;
      sxy += dx * (p.y - my);
    }
    if (sxx < 1e-9) return std::nullopt;  // 退化：窗口内 tick 几乎不变
    const double skew = sxy / sxx;
    if (skew <= 0.0) return std::nullopt;  // 斜率非正 → 拟合异常

    // offset = min(host - skew·x)：min-filter 取延迟最小帧定截距，丢弃抖动尖峰。
    // 用未中心化的 p.x，与下面 mapped = skew·x + offset 的重建式保持一致。
    double offset = buf_.front().y - skew * buf_.front().x;
    for (const auto & p : buf_) {
      const double v = p.y - skew * p.x;
      if (v < offset) offset = v;
    }

    const double mapped_sec = skew * x + offset;  // 当前帧(相对首帧)的映射秒
    return host_ref_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                         std::chrono::duration<double>(mapped_sec));
  }

  // 相机重连/重新 StartGrabbing 后必须调用：清空窗口与参考，下次 update 重新锚定。
  void reset()
  {
    buf_.clear();
    have_ref_ = false;
    dev_ref_ = 0;
    last_dev_ = 0;
  }

  bool converged() const { return buf_.size() >= min_samples_; }

private:
  struct Sample
  {
    double x;  // dev_tick - dev_ref_
    double y;  // host_recv - host_ref_ (秒)
  };

  std::size_t window_;
  std::size_t min_samples_;
  std::deque<Sample> buf_;

  bool have_ref_ = false;
  uint64_t dev_ref_ = 0;
  uint64_t last_dev_ = 0;
  std::chrono::steady_clock::time_point host_ref_;
};

}  // namespace tools

#endif  // TOOLS__DEVICE_CLOCK_MAPPER_HPP
