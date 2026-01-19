// coord_converter.hpp
#ifndef COORD_CONVERTER_HPP
#define COORD_CONVERTER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <string>
#include "solver.hpp"
#include "math_tools.hpp"
#include "logger.hpp"
#include <utility>

namespace solver {

using ArmorType = armor_auto_aim::ArmorType;
using ArmorName = armor_auto_aim::ArmorName;

class CoordConverter {
public:
    explicit CoordConverter(const std::string& yaml_config_path);
    
    void updateIMU(const Eigen::Quaterniond& q_absolute, double timestamp);
    void updateIMU(double yaw, double pitch, double timestamp);
    
    Eigen::Vector3d transform(const Eigen::Vector3d& point,
                             CoordinateFrame from,
                             CoordinateFrame to) const;

    Eigen::Matrix3d transformRotation(const Eigen::Matrix3d& rotation,
                                  CoordinateFrame from,
                                  CoordinateFrame to) const;
    
    Gimbal createGimbal(const Eigen::Vector3d& target_position,
                       CoordinateFrame frame,
                       double timestamp = 0.0) const;
    
    Orientation getCurrentAngles() const;

    // bool isInitialized() const { return is_initialized_; }

    // std::vector<cv::Point2f> reproject_armor(const Eigen::Vector3d & xyz_in_world, 
    //                                     double yaw, ArmorType type, ArmorName name) const;

    
    std::pair<Eigen::Matrix3d, Eigen::Matrix3d> getCameraToWorldRotation() const;

private:
    // void initializeWorldFrame();
    bool loadCalibrationFromYAML(const std::string& yaml_path);
    

    Eigen::Vector3d cameraToGimbal(const Eigen::Vector3d& point_camera) const;
    Eigen::Vector3d cameraToWorld(const Eigen::Vector3d& point_camera) const;
    Eigen::Vector3d GimbalToWorld(const Eigen::Vector3d& point_gimbal) const;
    Eigen::Vector3d GimbalToCamera(const Eigen::Vector3d& point_gimbal) const;
    Eigen::Vector3d WorldToCamera(const Eigen::Vector3d& point_world) const;
    Eigen::Vector3d WorldToGimbal(const Eigen::Vector3d& point_world) const;


    
    // Eigen::Vector3d extractGravityFromIMU(const Eigen::Quaterniond& q_imu) const;
    // Eigen::Matrix3d computeWorldAlignment(const Eigen::Vector3d& gravity_in_imu) const;
    
    cv::Mat camera_matrix_; // 相机内参
    cv::Mat dist_coeffs_;   // 畸变参数
    
    // Eigen::Matrix3d R_camera_to_imu;  // 相机到IMU的旋转
    // Eigen::Vector3d t_camera_to_imu;  // 相机到IMU的平移向量
    // Eigen::Matrix3d R_imu_to_camera;  // IMU到相机的旋转

    
    bool is_initialized_;
    // Eigen::Matrix3d R_world_alignment_;    // 世界坐标系对齐矩阵
    // Eigen::Matrix3d R_yaw_init_;           // 消除初始时刻的yaw矩阵
    Eigen::Matrix3d R_imu_to_world_;       // IMU到世界坐标系的矩阵
    
    Eigen::Matrix3d R_imu_current;
    Eigen::Matrix3d R_gimbal_to_world;
    Eigen::Matrix3d R_gimbal_to_imu;
    Eigen::Matrix3d R_camera_to_gimbal;

    Eigen::Vector3d t_camera_to_gimbal_;   // 相机到云台的平移向量

    Eigen::Quaterniond current_q_abs_;     // 当前时刻的四元数
    double current_timestamp_;

    Orientation current_imu_angles_;       // 当前时刻的云台角度（包含yaw, pitch, roll）
};

}

#endif