#include "math_tools.hpp"

#include <cmath>
#include <opencv2/core.hpp>  // CV_PI

namespace utils
{
double limit_rad(double angle)
{
  while (angle > CV_PI) angle -= 2 * CV_PI;
  while (angle <= -CV_PI) angle += 2 * CV_PI;
  return angle;
}

Eigen::Vector3d eulers(Eigen::Quaterniond q, int axis0, int axis1, int axis2, bool extrinsic)
{
  if (!extrinsic) std::swap(axis0, axis2);

  auto i = axis0, j = axis1, k = axis2;
  auto is_proper = (i == k);
  if (is_proper) k = 3 - i - j;
  auto sign = (i - j) * (j - k) * (k - i) / 2;

  double a, b, c, d;
  Eigen::Vector4d xyzw = q.coeffs();
  if (is_proper) {
    a = xyzw[3];
    b = xyzw[i];
    c = xyzw[j];
    d = xyzw[k] * sign;
  } else {
    a = xyzw[3] - xyzw[j];
    b = xyzw[i] + xyzw[k] * sign;
    c = xyzw[j] + xyzw[3];
    d = xyzw[k] * sign - xyzw[i];
  }

  Eigen::Vector3d eulers;
  auto n2 = a * a + b * b + c * c + d * d;
  eulers[1] = std::acos(2 * (a * a + b * b) / n2 - 1);

  auto half_sum = std::atan2(b, a);
  auto half_diff = std::atan2(-d, c);

  auto eps = 1e-7;
  auto safe1 = std::abs(eulers[1]) >= eps;
  auto safe2 = std::abs(eulers[1] - CV_PI) >= eps;
  auto safe = safe1 && safe2;
  if (safe) {
    eulers[0] = half_sum + half_diff;
    eulers[2] = half_sum - half_diff;
  } else {
    if (!extrinsic) {
      eulers[0] = 0;
      if (!safe1) eulers[2] = 2 * half_sum;
      if (!safe2) eulers[2] = -2 * half_diff;
    } else {
      eulers[2] = 0;
      if (!safe1) eulers[0] = 2 * half_sum;
      if (!safe2) eulers[0] = 2 * half_diff;
    }
  }

  for (int i = 0; i < 3; i++) eulers[i] = limit_rad(eulers[i]);

  if (!is_proper) {
    eulers[2] *= sign;
    eulers[1] -= CV_PI / 2;
  }

  if (!extrinsic) std::swap(eulers[0], eulers[2]);

  return eulers;
}

Eigen::Vector3d eulers(Eigen::Matrix3d R, int axis0, int axis1, int axis2, bool extrinsic)
{
  Eigen::Quaterniond q(R);
  return eulers(q, axis0, axis1, axis2, extrinsic);
}

Eigen::Vector3d eulers_zyx(const Eigen::Matrix3d& R) {
    Eigen::Vector3d eulers;
    
    // 调试输出：关键的矩阵元素
    // utils::logger()->debug(
    //     "eulers函数 - 关键矩阵元素: R(2,0)={:.6f}, R(1,0)={:.6f}, R(0,0)={:.6f}",
    //     R(2,0), R(1,0), R(0,0)
    // );
    
    // 计算 pitch
    double sin_pitch = -R(2,0);
    // 限制 sin_pitch 在 [-1, 1] 范围内，防止 asin 出错
    sin_pitch = std::max(-1.0, std::min(1.0, sin_pitch));
    eulers[1] = asin(sin_pitch);
    
    // utils::logger()->debug(
    //     "eulers函数 - pitch计算: sin(pitch)={:.6f}, pitch(rad)={:.6f}, pitch(deg)={:.2f}°",
    //     sin_pitch, eulers[1], eulers[1] * 180.0 / M_PI
    // );
    
    // 检查万向锁
    double cos_pitch = cos(eulers[1]);
    
    if (fabs(cos_pitch) > 1e-6) {
        // 正常情况
        eulers[0] = atan2(R(1,0), R(0,0));  // yaw
        eulers[2] = atan2(R(2,1), R(2,2));  // roll
        
        // utils::logger()->debug(
        //     "eulers函数 - 正常提取: yaw(rad)={:.6f}, roll(rad)={:.6f}",
        //     eulers[0], eulers[2]
        // );
    } else {
        // 万向锁情况
        eulers[0] = atan2(-R(0,1), R(1,1));
        eulers[2] = 0;
        
        // utils::logger()->debug(
        //     "eulers函数 - 万向锁情况! yaw(rad)={:.6f}, roll=0",
        //     eulers[0]
        // );
    }
    
    // // 范围限制前的值
    // utils::logger()->debug(
    //     "eulers函数 - 范围限制前(rad): [{:.6f}, {:.6f}, {:.6f}]",
    //     eulers[0], eulers[1], eulers[2]
    // );
    
    // 确保角度在 [-π, π] 范围内
    for (int i = 0; i < 3; ++i) {
        int wrap_count = 0;
        while (eulers[i] > M_PI) {
            eulers[i] -= 2 * M_PI;
            wrap_count++;
        }
        while (eulers[i] < -M_PI) {
            eulers[i] += 2 * M_PI;
            wrap_count--;
        }
        if (wrap_count != 0) {
            utils::logger()->debug(
                "eulers函数 - 第{}个角度进行了{}次2π调整",
                i, wrap_count
            );
        }
    }
    
    // // 范围限制后的值
    // utils::logger()->debug(
    //     "eulers函数 - 最终结果(rad): yaw={:.6f}, pitch={:.6f}, roll={:.6f}",
    //     eulers[0], eulers[1], eulers[2]
    // );
    // utils::logger()->debug(
    //     "eulers函数 - 最终结果(deg): yaw={:.2f}°, pitch={:.2f}°, roll={:.2f}°",
    //     eulers[0] * 180.0 / M_PI, 
    //     eulers[1] * 180.0 / M_PI, 
    //     eulers[2] * 180.0 / M_PI
    // );
    
    return eulers;  // 返回 [yaw, pitch, roll] 弧度值
}

Eigen::Vector3d eulers_yxz(const Eigen::Matrix3d& R_) {
    Eigen::Vector3d eulers;
    
    // 调试输出：关键的矩阵元素
    // utils::logger()->debug(
    //     "eulers函数 - 关键矩阵元素: R(0,2)={:.6f}, R(1,2)={:.6f}, R(2,2)={:.6f}",
    //     R(0,2), R(1,2), R(2,2)
    // );
    
    // 计算 pitch (绕X轴旋转)
    double sin_pitch = -R_(1,2);  // 修改：使用不同的矩阵元素
    // 限制 sin_pitch 在 [-1, 1] 范围内，防止 asin 出错
    sin_pitch = std::max(-1.0, std::min(1.0, sin_pitch));
    eulers[1] = asin(sin_pitch);
    
    // utils::logger()->debug(
    //     "eulers函数 - pitch计算: sin(pitch)={:.6f}, pitch(rad)={:.6f}, pitch(deg)={:.2f}°",
    //     sin_pitch, eulers[1], eulers[1] * 180.0 / M_PI
    // );
    
    // 检查万向锁
    double cos_pitch = cos(eulers[1]);
    
    if (fabs(cos_pitch) > 1e-6) {
        // 正常情况
        eulers[0] = atan2(-R_(0,2), R_(2,2));  // yaw (绕Y轴旋转)
        eulers[2] = atan2(-R_(1,0), R_(1,1));  // roll (绕Z轴旋转)
        
        // utils::logger()->debug(
        //     "eulers函数 - 正常提取: yaw(rad)={:.6f}, roll(rad)={:.6f}",
        //     eulers[0], eulers[2]
        // );
    } else {
        // 万向锁情况 (pitch接近±90°时)
        eulers[0] = atan2(R_(2,0), R_(0,0));  // yaw
        eulers[2] = 0;  // roll设为0
        
        // utils::logger()->debug(
        //     "eulers函数 - 万向锁情况! yaw(rad)={:.6f}, roll=0",
        //     eulers[0]
        // );
    }
    
    // // 范围限制前的值
    // utils::logger()->debug(
    //     "eulers函数 - 范围限制前(rad): [{:.6f}, {:.6f}, {:.6f}]",
    //     eulers[0], eulers[1], eulers[2]
    // );
    
    // 确保角度在 [-π, π] 范围内
    for (int i = 0; i < 3; ++i) {
        int wrap_count = 0;
        while (eulers[i] > M_PI) {
            eulers[i] -= 2 * M_PI;
            wrap_count++;
        }
        while (eulers[i] < -M_PI) {
            eulers[i] += 2 * M_PI;
            wrap_count--;
        }
        if (wrap_count != 0) {
            utils::logger()->debug(
                "eulers函数 - 第{}个角度进行了{}次2π调整",
                i, wrap_count
            );
        }
    }
    
    // // 范围限制后的值
    // utils::logger()->debug(
    //     "eulers函数 - 最终结果(rad): yaw={:.6f}, pitch={:.6f}, roll={:.6f}",
    //     eulers[0], eulers[1], eulers[2]
    // );
    // utils::logger()->debug(
    //     "eulers函数 - 最终结果(deg): yaw={:.2f}°, pitch={:.2f}°, roll={:.2f}°",
    //     eulers[0] * 180.0 / M_PI, 
    //     eulers[1] * 180.0 / M_PI, 
    //     eulers[2] * 180.0 / M_PI
    // );
    
    return eulers;  // 返回 [yaw, pitch, roll] 弧度值
}

Eigen::Matrix3d rotation_matrix_zyx(double yaw, double pitch, double roll)
{

  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  double cos_pitch = cos(pitch);
  double sin_pitch = sin(pitch);
  double cos_roll = cos(roll);
  double sin_roll = sin(roll);
  // clang-format off
    Eigen::Matrix3d R{
      {cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll},
      {sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll},
      {         -sin_pitch,                                cos_pitch * sin_roll,                                cos_pitch * cos_roll}
    };
  // clang-format on
  return R;
}

Eigen::Matrix3d rotation_matrix_yxz(double yaw, double pitch, double roll)
{
    // double roll = ypr[2];
    // double pitch = ypr[1];
    // double yaw = ypr[0];
    auto sin_yaw = std::sin(yaw);
    auto cos_yaw = std::cos(yaw);
    auto sin_pitch = std::sin(pitch);
    auto cos_pitch = std::cos(pitch);
    auto sin_roll = std::sin(roll);
    auto cos_roll =  std::cos(roll);
    
    const Eigen::Matrix3d R {
        {cos_yaw * cos_roll + sin_yaw * sin_pitch * sin_roll, -cos_yaw * sin_roll +  sin_yaw * sin_pitch * cos_roll, sin_yaw * cos_pitch},
        {                                 cos_pitch *sin_roll ,                           cos_pitch * cos_roll ,              -sin_pitch},
        { -sin_yaw * cos_roll + cos_yaw * sin_pitch * sin_roll, sin_yaw * sin_roll + cos_yaw*sin_pitch * cos_roll,  cos_yaw * cos_pitch}
    };
    
    return R;
}

Eigen::Vector3d xyz2ypd(const Eigen::Vector3d & xyz)
{
  auto x = xyz[0], y = xyz[1], z = xyz[2];
  auto yaw = std::atan2(y, x);
  auto pitch = std::atan2(z, std::sqrt(x * x + y * y));
  auto distance = std::sqrt(x * x + y * y + z * z);
  return {yaw, pitch, distance};
}

Eigen::MatrixXd xyz2ypd_jacobian(const Eigen::Vector3d & xyz)
{
  auto x = xyz[0], y = xyz[1], z = xyz[2];

  auto dyaw_dx = -y / (x * x + y * y);
  auto dyaw_dy = x / (x * x + y * y);
  auto dyaw_dz = 0.0;

  auto dpitch_dx = -(x * z) / ((z * z / (x * x + y * y) + 1) * std::pow((x * x + y * y), 1.5));
  auto dpitch_dy = -(y * z) / ((z * z / (x * x + y * y) + 1) * std::pow((x * x + y * y), 1.5));
  auto dpitch_dz = 1 / ((z * z / (x * x + y * y) + 1) * std::pow((x * x + y * y), 0.5));

  auto ddistance_dx = x / std::pow((x * x + y * y + z * z), 0.5);
  auto ddistance_dy = y / std::pow((x * x + y * y + z * z), 0.5);
  auto ddistance_dz = z / std::pow((x * x + y * y + z * z), 0.5);

  // clang-format off
  Eigen::MatrixXd J{
    {dyaw_dx, dyaw_dy, dyaw_dz},
    {dpitch_dx, dpitch_dy, dpitch_dz},
    {ddistance_dx, ddistance_dy, ddistance_dz}
  };
  // clang-format on

  return J;
}

Eigen::Vector3d ypd2xyz(const Eigen::Vector3d & ypd)
{
  auto yaw = ypd[0], pitch = ypd[1], distance = ypd[2];
  auto x = distance * std::cos(pitch) * std::cos(yaw);
  auto y = distance * std::cos(pitch) * std::sin(yaw);
  auto z = distance * std::sin(pitch);
  return {x, y, z};
}

Eigen::MatrixXd ypd2xyz_jacobian(const Eigen::Vector3d & ypd)
{
  auto yaw = ypd[0], pitch = ypd[1], distance = ypd[2];
  double cos_yaw = std::cos(yaw);
  double sin_yaw = std::sin(yaw);
  double cos_pitch = std::cos(pitch);
  double sin_pitch = std::sin(pitch);

  auto dx_dyaw = distance * cos_pitch * -sin_yaw;
  auto dy_dyaw = distance * cos_pitch * cos_yaw;
  auto dz_dyaw = 0.0;

  auto dx_dpitch = distance * -sin_pitch * cos_yaw;
  auto dy_dpitch = distance * -sin_pitch * sin_yaw;
  auto dz_dpitch = distance * cos_pitch;

  auto dx_ddistance = cos_pitch * cos_yaw;
  auto dy_ddistance = cos_pitch * sin_yaw;
  auto dz_ddistance = sin_pitch;

  // clang-format off
  Eigen::MatrixXd J{
    {dx_dyaw, dx_dpitch, dx_ddistance},
    {dy_dyaw, dy_dpitch, dy_ddistance},
    {dz_dyaw, dz_dpitch, dz_ddistance}
  };
  // clang-format on

  return J;
}

double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b)
{
  std::chrono::duration<double> c = a - b;
  return c.count();
}

double get_abs_angle(const Eigen::Vector2d & vec1, const Eigen::Vector2d & vec2)
{
  if (vec1.norm() == 0. || vec2.norm() == 0.) {
    return 0.;
  }
  return std::acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));
}

double limit_min_max(double input, double min, double max)
{
  if (input > max)
    return max;
  else if (input < min)
    return min;
  return input;
}

double sin_interp(double x, double x0, double x1, double y0, double y1)
{
  // 计算归一化参数 t ∈ [0, 1]
  double t = (x - x0) / (x1 - x0);
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  // 使用正弦曲线实现平滑过渡
  double s = std::sin(t * CV_PI / 2.0);
  return y0 + s * (y1 - y0);
}



}  // namespace tools