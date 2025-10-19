// solver.hpp
#pragma once
#include <opencv2/core.hpp>
#include <array>
#include <Eigen/Dense>
#include "common/armor.hpp"  // 使用实际的armor.hpp
#include "math_tools.hpp"
#include "logger.hpp"

namespace solver {

// 装甲板世界坐标点定义
// const std::vector<cv::Point3f> PW_SMALL = {
//     cv::Point3f(-67.5, -27.5, 0),  // 对应图像左上
//     cv::Point3f(67.5, -27.5, 0),   // 对应图像右上
//     cv::Point3f(67.5, 27.5, 0),    // 对应图像右下
//     cv::Point3f(-67.5, 27.5, 0)    // 对应图像左下
// };

// 现在这个与IMU的坐标系定义一致
const std::vector<cv::Point3f> PW_SMALL = {
    cv::Point3f(0, 67.5, 27.5),  // 对应图像左上
    cv::Point3f(0, -67.5, 27.5),   // 对应图像右上
    cv::Point3f(0, -67.5, -27.5),    // 对应图像右下
    cv::Point3f(0, 67.5, -27.5)    // 对应图像左下
};

// const std::vector<cv::Point3f> PW_SMALL = {
//     cv::Point3f( 27.5, 0, -67.5),  // 左上：(0,-67.5,-27.5)→(27.5,0,-67.5)
//     cv::Point3f( 27.5, 0,  67.5),  // 右上：(0,67.5,-27.5)→(27.5,0,67.5)
//     cv::Point3f(-27.5, 0,  67.5),  // 右下：(0,67.5,27.5)→(-27.5,0,67.5)
//     cv::Point3f(-27.5, 0, -67.5)   // 左下：(0,-67.5,27.5)→(-27.5,0,-67.5)
// };


const std::vector<cv::Point3f> PW_BIG = {
    cv::Point3f(-115.0f, -27.5f, 0.0f),
    cv::Point3f(115.0f, -27.5f, 0.0f),
    cv::Point3f(115.0f, 27.5f, 0.0f),
    cv::Point3f(-115.0f, 27.5f, 0.0f)
};

enum class CoordinateFrame {
    CAMERA,
    IMU,
    WORLD,
    GIMBAL
};

enum class EulerOrder{
    YXZ,
    ZYX
};



// 云台角度信息
struct YawPitch {
    double yaw;
    double pitch;
    double timestamp;
    
    YawPitch();
    YawPitch(double y, double p, double t = 0.0);
};

// 球坐标表示
struct SphericalCoord {
    double yaw;
    double pitch;
    double distance;
    double timestamp;
    
    SphericalCoord();
    SphericalCoord(double y, double p, double d, double t = 0.0);
    SphericalCoord(const cv::Point3d& cartesian, double t = 0.0);
    SphericalCoord(const Eigen::Vector3d& cartesian, double t = 0.0);
};


// 装甲板自身姿态表示
struct Orientation
{
    double yaw;
    double pitch;
    double roll;
    double timestamp;

    Orientation();
    Orientation(double y, double p, double r, double t = 0.0);
    Orientation(const Eigen::Matrix3d& rotation, EulerOrder order, double t = 0.0);
};


// 简化后的目标结构体 - 只包含世界坐标系下的信息
struct Armor_pose {
    armor_auto_aim::ArmorName id;        // 装甲板ID
    armor_auto_aim::ArmorType type;      // 装甲板类型
    Eigen::Vector3d camera_position;      // 相机坐标系位置
    Eigen::Vector3d gimbal_position;      // 云台坐标系位置
    Eigen::Vector3d world_position;      // 世界坐标系位置
    SphericalCoord world_spherical;      // 世界坐标系球坐标
    Orientation gimbal_orientation;       // 云台坐标系下的姿态
    Orientation world_orientation;       // 世界坐标系下的姿态
    double timestamp;
    
    Armor_pose();
    Armor_pose(armor_auto_aim::ArmorName id, 
           armor_auto_aim::ArmorType type,
           const Eigen::Vector3d& camera_pos,
           const Eigen::Vector3d& world_pos,
           const Eigen::Matrix3d& camera_rotation,
           const Eigen::Matrix3d& world_rotation,
           double t = 0.0);
};

struct PnPResult {
    bool success;
    Eigen::Vector3d camera_position;     // 目标在相机坐标系下的位置
    Eigen::Matrix3d R_armor_to_camera;   // 装甲板到相机的旋转矩阵
    SphericalCoord camera_spherical;     // 相机坐标系下的球坐标
    double reprojection_error;           // 重投影误差
    
    PnPResult() : success(false), reprojection_error(0) {
        camera_position = Eigen::Vector3d::Zero();
        R_armor_to_camera = Eigen::Matrix3d::Identity();
        camera_spherical = SphericalCoord();
    }
};

// 云台状态信息
struct Gimbal {
    YawPitch current_angles;      // 当前云台角度
    YawPitch target_angles;       // 目标云台角度
    double timestamp;
    
    Gimbal();
    Gimbal(const YawPitch& current,
           const YawPitch& target,
           double t = 0.0);
    
    // 便捷构造函数：从当前角度和目标世界坐标计算
    Gimbal(const YawPitch& imu_angles,
           const Eigen::Vector3d& target_world_position,
           double t = 0.0);
};

}