// coord_converter.cpp
#include "module/coord_converter.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace solver {

CoordConverter::CoordConverter(const std::string& yaml_config_path)
    : is_initialized_(false),
      current_timestamp_(0.0),
      current_imu_angles_(0.0, 0.0, 0.0, 0.0) {

    camera_matrix_ = cv::Mat();
    dist_coeffs_ = cv::Mat();
    current_q_abs_ = Eigen::Quaterniond::Identity();
    R_imu_to_world_ = Eigen::Matrix3d::Identity();
    R_gimbal_to_world = Eigen::Matrix3d::Identity();
    R_camera_to_gimbal = Eigen::Matrix3d::Identity();
    R_gimbal_to_imu = Eigen::Matrix3d::Identity();
    t_camera_to_gimbal_ = Eigen::Vector3d::Zero();  // 初始化平移向量

    if (!loadCalibrationFromYAML(yaml_config_path)) {
        throw std::runtime_error("Cannot load calibration file: " + yaml_config_path);
    }
}

void CoordConverter::updateIMU(const Eigen::Quaterniond& q_absolute, double timestamp) {
    current_q_abs_ = q_absolute;
    current_timestamp_ = timestamp;
    
    // if (!is_initialized_) {
    //     initializeWorldFrame();
    // }
    
    R_imu_current = current_q_abs_.toRotationMatrix();

    // 这是上个版本的变换接口，暂时保留

    // R_imu_to_world_ = R_yaw_init_ * R_world_alignment_ * R_imu_current;

    // double yaw = std::atan2(R_imu_to_world_(1,0), R_imu_to_world_(0,0)) * 180.0 / M_PI;
    // double pitch = std::asin(-R_imu_to_world_(2,0)) * 180.0 / M_PI;
    // current_imu_angles_ = YawPitch(yaw, pitch, timestamp);

    R_gimbal_to_world = R_gimbal_to_imu.transpose() * R_imu_current * R_gimbal_to_imu;

    // 从旋转矩阵提取欧拉角 (ZYX顺序: yaw, pitch, roll)
    Eigen::Vector3d ypr = utils::eulers(R_gimbal_to_world, 2, 1, 0);

    // 更新当前云台角度（弧度值）
    current_imu_angles_ = Orientation(ypr[0], ypr[1], ypr[2], timestamp);

    // utils::logger()->debug("R_yaw_init旋转矩阵:");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         R_yaw_init_(i,0), R_yaw_init_(i,1), R_yaw_init_(i,2));
    // }

    // utils::logger()->debug("wrold_alignment旋转矩阵:");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         R_world_alignment_(i,0), R_world_alignment_(i,1), R_world_alignment_(i,2));
    // }
}

void CoordConverter::updateIMU(double yaw, double pitch, double timestamp) {
    current_imu_angles_ = Orientation(yaw, pitch, 0.0, timestamp);
    current_timestamp_ = timestamp;
    
    double yaw_rad = yaw * M_PI / 180.0;
    double pitch_rad = pitch * M_PI / 180.0;
    
    R_imu_to_world_ = Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()) *
                      Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY()) *
                      Eigen::Matrix3d::Identity();
    
    if (!is_initialized_) {
        is_initialized_ = true;
    }
}


// 获取关键的两个旋转矩阵接口，pair打包

std::pair<Eigen::Matrix3d, Eigen::Matrix3d> CoordConverter::getCameraToWorldRotation() const {
    // R_camera_to_world = R_imu_to_world * R_camera_to_imu
    // 这个矩阵描述了如何将相机坐标系中的向量转换到世界坐标系

    return std::make_pair(R_gimbal_to_world, R_camera_to_gimbal);
    // return R_camera_to_imu;
}


Eigen::Vector3d CoordConverter::transform(const Eigen::Vector3d& point,
                                          CoordinateFrame from,
                                          CoordinateFrame to) const {
    if (from == to) {
        return point;
    }
    
    Eigen::Vector3d point_world;
    
    switch (from) {
        case CoordinateFrame::CAMERA:
            point_world = cameraToWorld(point);
            break;      
        case CoordinateFrame::GIMBAL:
            point_world = GimbalToWorld(point);
            break;          
        case CoordinateFrame::WORLD:
            point_world = point;
            break;
        case CoordinateFrame::IMU:
            point_world = point;
            break;
    }
    
    switch (to) {
        case CoordinateFrame::CAMERA:
            return WorldToCamera(point_world);
        case CoordinateFrame::GIMBAL:
            return WorldToGimbal(point_world);           
        case CoordinateFrame::WORLD:
            return point_world;
        case CoordinateFrame::IMU:
            return point_world;
    }
    
    return point;
}

Eigen::Matrix3d CoordConverter::transformRotation(const Eigen::Matrix3d& rotation,
                                                  CoordinateFrame from,
                                                  CoordinateFrame to) const {
    if (from == to) {
        return rotation;
    }
    
    if (from == CoordinateFrame::CAMERA && to == CoordinateFrame::GIMBAL) {
        // R_armor_to_world = R_imu_to_world * R_camera_to_imu * R_armor_to_camera
        return R_camera_to_gimbal * rotation;
    }else if (from == CoordinateFrame::CAMERA && to == CoordinateFrame::WORLD){
        return R_gimbal_to_world * R_camera_to_gimbal * rotation;
    }else if (from == CoordinateFrame::GIMBAL && to == CoordinateFrame::WORLD){
        return R_gimbal_to_world * rotation;
    }else if(from == CoordinateFrame::WORLD && to == CoordinateFrame::CAMERA){
        return R_camera_to_gimbal.transpose() * R_gimbal_to_world.transpose() * rotation;
    }


    return rotation;
}



Gimbal CoordConverter::createGimbal(const Eigen::Vector3d& target_position,
                                    CoordinateFrame frame,
                                    double timestamp) const {
    Eigen::Vector3d target_world = transform(target_position, frame, CoordinateFrame::WORLD);

    // 将 Orientation 转换为 YawPitch 用于 Gimbal 构造
    YawPitch current_yp(current_imu_angles_.yaw, current_imu_angles_.pitch, current_imu_angles_.timestamp);
    return Gimbal(current_yp, target_world, timestamp);
}

Orientation CoordConverter::getCurrentAngles() const {
    return current_imu_angles_;
}


// 上个版本的坐标变换接口

// void CoordConverter::initializeWorldFrame() {
//     if (current_q_abs_.w() == 1.0 && current_q_abs_.x() == 0.0 && 
//         current_q_abs_.y() == 0.0 && current_q_abs_.z() == 0.0) {
//         return;
//     }
    
//     Eigen::Vector3d gravity_in_imu = extractGravityFromIMU(current_q_abs_);
//     R_world_alignment_ = computeWorldAlignment(gravity_in_imu);
    
//     Eigen::Matrix3d R_current = current_q_abs_.toRotationMatrix();
//     Eigen::Matrix3d R_aligned = R_world_alignment_ * R_current;
    
//     double yaw = std::atan2(R_aligned(1,0), R_aligned(0,0));
//     R_yaw_init_ = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    
//     is_initialized_ = true;
// }


// 坐标系变换函数

Eigen::Vector3d CoordConverter::cameraToGimbal(const Eigen::Vector3d& point_camera) const {
    // 相机到云台的变换：p_gimbal = R * p_camera + t
    return R_camera_to_gimbal * point_camera + t_camera_to_gimbal_;
}

Eigen::Vector3d CoordConverter::cameraToWorld(const Eigen::Vector3d& point_camera) const {
    Eigen::Vector3d point_in_gimbal = cameraToGimbal(point_camera);

    return R_gimbal_to_world * point_in_gimbal;
}

Eigen::Vector3d CoordConverter::GimbalToWorld(const Eigen::Vector3d& point_gimbal) const {
    return R_gimbal_to_world * point_gimbal;
}

Eigen::Vector3d CoordConverter::GimbalToCamera(const Eigen::Vector3d& point_gimbal) const {
    // 云台到相机的变换：p_camera = R^T * (p_gimbal - t)
    return R_camera_to_gimbal.transpose() * (point_gimbal - t_camera_to_gimbal_);
}

Eigen::Vector3d CoordConverter::WorldToGimbal(const Eigen::Vector3d& point_world) const {
    return R_gimbal_to_world.transpose() * point_world;
}

Eigen::Vector3d CoordConverter::WorldToCamera(const Eigen::Vector3d& point_world) const {
    Eigen::Vector3d point_in_gimbal = WorldToGimbal(point_world);
    // 云台到相机的变换：p_camera = R^T * (p_gimbal - t)
    return R_camera_to_gimbal.transpose() * (point_in_gimbal - t_camera_to_gimbal_);
}



// 上个版本的坐标变换的接口

// Eigen::Vector3d CoordConverter::extractGravityFromIMU(const Eigen::Quaterniond& q_imu) const {
//     Eigen::Vector3d gravity_world(0, 0, -9.8);
//     Eigen::Matrix3d R_imu = q_imu.toRotationMatrix();
//     return R_imu.transpose() * gravity_world;
// }

// Eigen::Matrix3d CoordConverter::computeWorldAlignment(const Eigen::Vector3d& gravity_in_imu) const {
//     Eigen::Vector3d target_gravity(0, 0, -9.8);
//     Eigen::Vector3d g_normalized = gravity_in_imu.normalized();
//     Eigen::Vector3d t_normalized = target_gravity.normalized();
    
//     Eigen::Vector3d axis = g_normalized.cross(t_normalized);
    
//     if (axis.norm() < 1e-6) {
//         if (g_normalized.dot(t_normalized) > 0) {
//             return Eigen::Matrix3d::Identity();
//         } else {
//             if (std::abs(g_normalized.x()) < 0.9) {
//                 axis = Eigen::Vector3d::UnitX();
//             } else {
//                 axis = Eigen::Vector3d::UnitY();
//             }
//             return Eigen::AngleAxisd(M_PI, axis).toRotationMatrix();
//         }
//     }
    
//     axis.normalize();
//     double angle = std::acos(g_normalized.dot(t_normalized));
    
//     Eigen::Matrix3d K;
//     K << 0, -axis.z(), axis.y(),
//          axis.z(), 0, -axis.x(),
//          -axis.y(), axis.x(), 0;
    
//     return Eigen::Matrix3d::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
// }

bool CoordConverter::loadCalibrationFromYAML(const std::string& yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        
        // 读取相机内参
        if (config["CalibParam"]["INTRI"]["Camera"]) {
            auto camera_node = config["CalibParam"]["INTRI"]["Camera"][0]["value"];
            
            YAML::Node camera_params;
            if (camera_node["ptr_wrapper"] && camera_node["ptr_wrapper"]["data"]) {
                camera_params = camera_node["ptr_wrapper"]["data"];
            } else {
                camera_params = camera_node;
            }
            
            std::vector<double> focal_length = camera_params["focal_length"].as<std::vector<double>>();
            std::vector<double> principal_point = camera_params["principal_point"].as<std::vector<double>>();
            
            camera_matrix_ = cv::Mat(3, 3, CV_64F);
            camera_matrix_.at<double>(0, 0) = focal_length[0];
            camera_matrix_.at<double>(0, 1) = 0.0;
            camera_matrix_.at<double>(0, 2) = principal_point[0];
            camera_matrix_.at<double>(1, 0) = 0.0;
            camera_matrix_.at<double>(1, 1) = focal_length[1];
            camera_matrix_.at<double>(1, 2) = principal_point[1];
            camera_matrix_.at<double>(2, 0) = 0.0;
            camera_matrix_.at<double>(2, 1) = 0.0;
            camera_matrix_.at<double>(2, 2) = 1.0;
            
            std::vector<double> disto_param = camera_params["disto_param"].as<std::vector<double>>();
            
            if (disto_param.size() >= 4) {
                dist_coeffs_ = cv::Mat(1, std::min(5, (int)disto_param.size()), CV_64F);
                for (int i = 0; i < dist_coeffs_.cols; i++) {
                    dist_coeffs_.at<double>(0, i) = disto_param[i];
                }
            }
        }
        
        // 读取相机到云台的旋转矩阵
        if(config["Solver"]["coord_converter"]) {
            std::vector<double> matrix_data = config["Solver"]["coord_converter"]["rotation_matrix_camera_to_gimbal"]["data"].as<std::vector<double>>();
            R_camera_to_gimbal << matrix_data[0], matrix_data[1], matrix_data[2],
                                matrix_data[3], matrix_data[4], matrix_data[5],
                                matrix_data[6], matrix_data[7], matrix_data[8];
        }

        // 读取云台到IMU的旋转矩阵
        if(config["Solver"]["coord_converter"]) {
            std::vector<double> matrix_data = config["Solver"]["coord_converter"]["rotation_matrix_gimbal_to_imu"]["data"].as<std::vector<double>>();
            R_gimbal_to_imu << matrix_data[0], matrix_data[1], matrix_data[2],
                            matrix_data[3], matrix_data[4], matrix_data[5],
                            matrix_data[6], matrix_data[7], matrix_data[8];
        }

        // 读取相机到云台的平移向量
        if(config["t_camera_to_gimbal"]) {
            std::vector<double> translation_data = config["t_camera_to_gimbal"].as<std::vector<double>>();
            if(translation_data.size() == 3) {
                t_camera_to_gimbal_ << translation_data[0], translation_data[1], translation_data[2];
                utils::logger()->info("成功加载相机到云台平移向量: [{:.3f}, {:.3f}, {:.3f}]",
                    t_camera_to_gimbal_(0), t_camera_to_gimbal_(1), t_camera_to_gimbal_(2));
            }
        }

        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

// std::vector<cv::Point2f> CoordConverter::reproject_armor(
//     const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
// {

//     auto pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;

//     double roll = 0.0;

//     // clang-format off
//     Eigen::Matrix3d R_armor_to_world = utils::rotation_matrix_zyx(yaw, pitch, roll);
//     Eigen::Matrix3d R_armor_to_camera = Coord_congverter_->transformRotation(
//         R_armor_to_world,
//         CoordinateFrame::CWORLD,
//         CoordinateFrame::CAMERA
//     );
//     // clang-format on
//     Eigen::Vector3d t_armor_to_camera = Coord_converter_->transform(
//         xyz_in_world,
//         CoordinateFrame::WORLD,
//         CoordinateFrame::CAMERA
//     );
//     // get R_armor2camera t_armor_to_camera


//     // get rvec tvec
//     cv::Vec3d rvec;
//     cv::Mat R_armor_to_camera_cv;
//     cv::eigen2cv(R_armor_to_camera, R_armor_to_camera_cv);
//     cv::Rodrigues(R_armor_to_camera_cv, rvec);
//     cv::Vec3d tvec(t_armor_to_camera[0], t_armor_to_camera[1], t_armor_to_camera[2]);

//     // reproject
//     std::vector<cv::Point2f> image_points;
    
//     // 需要确保 PW_BIG 和 PW_SMALL 已定义
//     const auto & object_points = (type == ArmorType::big) ? PW_BIG : PW_SMALL;
    
//     // 修正变量名：dist_coeffs_ 而不是 distort_coeffs_
//     cv::projectPoints(object_points, rvec, tvec, camera_matrix_, dist_coeffs_, image_points);
//     return image_points;
// }

}
