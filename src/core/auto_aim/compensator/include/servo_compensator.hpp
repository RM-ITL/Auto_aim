#ifndef AUTO_AIM__SERVO_COMPENSATOR_HPP
#define AUTO_AIM__SERVO_COMPENSATOR_HPP

#include <deque>
#include <string>
#include <vector>

namespace compensator
{

/// 单轴 ZPETC 前馈滤波器
/// 实现离散传递函数: r(k) = [A_c(z⁻¹) * B_c^u*(z⁻¹)] / [B_c^a(z⁻¹) * (B_c^u(1))²] * y_d(k+d+s)
/// 简化为通用差分方程: r(k) = sum(ff_i * y_d(k+d+s-i)) - sum(fb_j * r(k-j))
class ZPETCFilter
{
public:
  ZPETCFilter() = default;

  /// 从辨识得到的闭环传函系数初始化
  /// @param b_c  闭环零点多项式系数 B_c(z⁻¹) = b_c[0] + b_c[1]*z⁻¹ + ...
  /// @param a_c  闭环极点多项式系数 A_c(z⁻¹) = 1 + a_c[1]*z⁻¹ + ...  (a_c[0]须为1)
  /// @param d    闭环系统纯延迟步数
  void init(const std::vector<double> & b_c, const std::vector<double> & a_c, int d);

  /// 应用滤波器: 输入期望值序列(当前及未来), 输出补偿后的指令
  /// @param y_d_future  期望值序列, y_d_future[0] = y_d(k), [1] = y_d(k+1), ...
  ///                    长度至少为 d + s + 1
  /// @return 补偿后的指令 r(k)
  double apply(const std::vector<double> & y_d_future);

  /// 简化接口: 只用当前值和速度做线性外推来生成未来值
  /// @param y_d      当前期望值
  /// @param y_d_vel  当前期望速度 (rad/s)
  /// @param dt       控制周期 (s)
  /// @return 补偿后的指令 r(k)
  double apply(double y_d, double y_d_vel, double dt);

  /// 重置滤波器状态
  void reset();

  /// 是否已初始化
  bool ready() const { return ready_; }

  /// 获取需要的未来步数 (d + s)
  int preview_steps() const { return d_ + s_; }

private:
  bool ready_{false};
  int d_{0};  // 纯延迟步数
  int s_{0};  // 不可消除零点个数

  // ZPETC 差分方程系数
  // r(k) = sum_{i=0}^{nff} ff_[i] * y_d(k + d + s - i)  -  sum_{j=1}^{nfb} fb_[j] * r(k-j)
  std::vector<double> ff_;  // 前馈系数 (分子)
  std::vector<double> fb_;  // 反馈系数 (分母, fb_[0] 归一化为 1)

  // 历史输出 buffer
  std::deque<double> r_hist_;
};

/// 双轴伺服补偿器, 封装 yaw 和 pitch 两个独立的 ZPETC 滤波器
class ServoCompensator
{
public:
  ServoCompensator() = default;

  /// 从配置文件加载参数
  explicit ServoCompensator(const std::string & config_path);

  /// 补偿 yaw 和 pitch
  /// @param yaw        期望 yaw (rad)
  /// @param yaw_vel    期望 yaw 速度 (rad/s)
  /// @param pitch      期望 pitch (rad)
  /// @param pitch_vel  期望 pitch 速度 (rad/s)
  /// @param dt         控制周期 (s)
  /// @param[out] yaw_cmd   补偿后的 yaw 指令
  /// @param[out] pitch_cmd 补偿后的 pitch 指令
  void compensate(
    double yaw, double yaw_vel,
    double pitch, double pitch_vel,
    double dt,
    double & yaw_cmd, double & pitch_cmd);

  /// 重置状态（目标丢失时调用）
  void reset();

  /// 是否启用
  bool enabled() const { return enabled_; }

private:
  bool enabled_{false};
  ZPETCFilter yaw_filter_;
  ZPETCFilter pitch_filter_;
};

}  // namespace compensator

#endif  // AUTO_AIM__SERVO_COMPENSATOR_HPP
