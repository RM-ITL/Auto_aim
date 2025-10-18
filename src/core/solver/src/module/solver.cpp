// solver.cpp
#include "module/solver.hpp"
#include <cmath>

namespace solver {

YawPitch::YawPitch() 
    : yaw(0.0), pitch(0.0), timestamp(0.0) {
}

YawPitch::YawPitch(double y, double p, double t) 
    : yaw(y), pitch(p), timestamp(t) {
}

SphericalCoord::SphericalCoord()
    : yaw(0.0), pitch(0.0), distance(0.0), timestamp(0.0) {
}

SphericalCoord::SphericalCoord(double y, double p, double d, double t)
    : yaw(y), pitch(p), distance(d), timestamp(t) {
}

SphericalCoord::SphericalCoord(const cv::Point3d& cartesian, double t)
    : timestamp(t) {
    Eigen::Vector3d xyz(cartesian.x, cartesian.y, cartesian.z);
    Eigen::Vector3d ypd = utils::xyz2ypd(xyz);
    
    yaw = ypd[0] * 180.0 / CV_PI;     // 转换为度
    pitch = ypd[1] * 180.0 / CV_PI;
    distance = ypd[2];
}

SphericalCoord::SphericalCoord(const Eigen::Vector3d& cartesian, double t)
    : timestamp(t) {
    Eigen::Vector3d ypd = utils::xyz2ypd(cartesian);
    
    // 这是角度值
    // yaw = ypd[0] * 180.0 / CV_PI;
    // pitch = ypd[1] * 180.0 / CV_PI;
    // distance = ypd[2];

    // 这是弧度值
    yaw = ypd[0];
    pitch = ypd[1];
    distance = ypd[2];
}

Orientation::Orientation()
    : yaw(0.0), pitch(0.0), roll(0.0), timestamp(0.0)   {

}

Orientation::Orientation(double y, double p, double r, double t)
    : yaw(y), pitch(p), roll(r), timestamp(t)   {

}

// solver.cpp 中添加
// solver.cpp 中的 Orientation 构造函数
// solver.cpp 中的 Orientation 构造函数
Orientation::Orientation(const Eigen::Matrix3d& rotation, double t)
    : timestamp(t) {
    
    // // 调试输出：打印输入的旋转矩阵
    // utils::logger()->debug(
    //     "Orientation构造 - 输入旋转矩阵:\n"
    //     "  [{:.6f}, {:.6f}, {:.6f}]\n"
    //     "  [{:.6f}, {:.6f}, {:.6f}]\n"
    //     "  [{:.6f}, {:.6f}, {:.6f}]",
    //     rotation(0,0), rotation(0,1), rotation(0,2),
    //     rotation(1,0), rotation(1,1), rotation(1,2),
    //     rotation(2,0), rotation(2,1), rotation(2,2)
    // );
    
    // 调用欧拉角提取,这里调用修改后的yxz轴提取,现在得到的angles就是ypr了
    Eigen::Vector3d angles = utils::eulers_yxz(rotation);
    
    // 调试输出：提取的弧度值
    // utils::logger()->debug(
    //     "Orientation构造 - 提取的欧拉角(弧度): yaw={:.6f}, pitch={:.6f}, roll={:.6f}",
    //     angles[0], angles[1], angles[2]
    // );


    
    // 转换为角度
    // roll = angles[0] * 180.0 / CV_PI;       // 对应真实yaw
    // yaw = angles[1] * 180.0 / CV_PI;        // 对应真实pitch
    // pitch = angles[2] * 180.0 / CV_PI;      // 对应真实roll
    
    // 这是弧度值
    // roll = angles[0];       // 对应真实yaw
    // yaw = angles[1];        // 对应真实pitch
    // pitch = angles[2];      // 对应真实roll

    yaw = angles[0];
    pitch = angles[1];
    roll = angles[2];
    
    // // 调试输出：转换后的角度值
    // utils::logger()->debug(
    //     "Orientation构造 - 转换后的角度: yaw={:.2f}°, pitch={:.2f}°, roll={:.2f}°",
    //     yaw, pitch, roll
    // );
}

Armor_pose::Armor_pose()
    : id(armor_auto_aim::ArmorName::not_armor),
      type(armor_auto_aim::ArmorType::small),
      world_position(0, 0, 0),
      world_spherical(0, 0, 0, 0),
      timestamp(0.0) {
}

Armor_pose::Armor_pose(armor_auto_aim::ArmorName id,
               armor_auto_aim::ArmorType type,
               const Eigen::Vector3d& camera_pos,
               const Eigen::Vector3d& world_pos,
               const Eigen::Matrix3d& camera_rotation,
               const Eigen::Matrix3d& world_rotation,
               double t)
    : id(id),
      type(type),
      camera_position(camera_pos),
      world_position(world_pos),
      world_spherical(world_pos, t),
      camera_orientation(camera_rotation, t),
      world_orientation(world_rotation,t),
      timestamp(t) {
}

Gimbal::Gimbal()
    : current_angles(0, 0, 0),
      target_angles(0, 0, 0),
      timestamp(0.0) {
}

Gimbal::Gimbal(const YawPitch& current,
               const YawPitch& target,
               double t)
    : current_angles(current),
      target_angles(target),
      timestamp(t) {
}

// solver.cpp 中修改
Gimbal::Gimbal(const YawPitch& imu_angles,
               const Eigen::Vector3d& target_world_position,
               double t)
    : current_angles(imu_angles),
      timestamp(t) {
    
    Eigen::Vector3d ypd = utils::xyz2ypd(target_world_position);
    
    target_angles = YawPitch(
        ypd[0] * 180.0 / CV_PI,   // yaw in degrees
        ypd[1] * 180.0 / CV_PI,   // pitch in degrees
        t
    );
}

}