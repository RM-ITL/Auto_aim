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
      current_imu_angles_(0.0, 0.0, 0.0) {
    
    camera_matrix_ = cv::Mat();
    dist_coeffs_ = cv::Mat();
    R_camera_to_imu = Eigen::Matrix3d::Identity();
    t_camera_to_imu = Eigen::Vector3d::Zero();
    current_q_abs_ = Eigen::Quaterniond::Identity();
    R_world_alignment_ = Eigen::Matrix3d::Identity();
    R_yaw_init_ = Eigen::Matrix3d::Identity();
    R_imu_to_world_ = Eigen::Matrix3d::Identity();
    
    if (!loadCalibrationFromYAML(yaml_config_path)) {
        throw std::runtime_error("Cannot load calibration file: " + yaml_config_path);
    }
}

void CoordConverter::updateIMU(const Eigen::Quaterniond& q_absolute, double timestamp) {
    current_q_abs_ = q_absolute;
    current_timestamp_ = timestamp;
    
    if (!is_initialized_) {
        initializeWorldFrame();
    }
    
    Eigen::Matrix3d R_imu_current = current_q_abs_.toRotationMatrix();
    R_imu_to_world_ = R_yaw_init_ * R_world_alignment_ * R_imu_current;
    
    double yaw = std::atan2(R_imu_to_world_(1,0), R_imu_to_world_(0,0)) * 180.0 / M_PI;
    double pitch = std::asin(-R_imu_to_world_(2,0)) * 180.0 / M_PI;
    current_imu_angles_ = YawPitch(yaw, pitch, timestamp);
}

void CoordConverter::updateIMU(double yaw, double pitch, double timestamp) {
    current_imu_angles_ = YawPitch(yaw, pitch, timestamp);
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
        case CoordinateFrame::IMU:
            point_world = imuToWorld(point);
            break;
        case CoordinateFrame::WORLD:
            point_world = point;
            break;
    }
    
    switch (to) {
        case CoordinateFrame::CAMERA:
            return worldToCamera(point_world);
        case CoordinateFrame::IMU:
            return worldToIMU(point_world);
        case CoordinateFrame::WORLD:
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
    
    if (from == CoordinateFrame::CAMERA && to == CoordinateFrame::WORLD) {
        // R_armor_to_world = R_imu_to_world * R_camera_to_imu * R_armor_to_camera
        Eigen::Matrix3d R_camera_to_world = R_imu_to_world_ * R_camera_to_imu;
        return R_camera_to_world * rotation;
    }
    else if (from == CoordinateFrame::CAMERA && to == CoordinateFrame::IMU) {
        return R_camera_to_imu * rotation;
    }
    return rotation;
}



Gimbal CoordConverter::createGimbal(const Eigen::Vector3d& target_position,
                                    CoordinateFrame frame,
                                    double timestamp) const {
    Eigen::Vector3d target_world = transform(target_position, frame, CoordinateFrame::WORLD);
    return Gimbal(current_imu_angles_, target_world, timestamp);
}

YawPitch CoordConverter::getCurrentAngles() const {
    return current_imu_angles_;
}

void CoordConverter::initializeWorldFrame() {
    if (current_q_abs_.w() == 1.0 && current_q_abs_.x() == 0.0 && 
        current_q_abs_.y() == 0.0 && current_q_abs_.z() == 0.0) {
        return;
    }
    
    Eigen::Vector3d gravity_in_imu = extractGravityFromIMU(current_q_abs_);
    R_world_alignment_ = computeWorldAlignment(gravity_in_imu);
    
    Eigen::Matrix3d R_current = current_q_abs_.toRotationMatrix();
    Eigen::Matrix3d R_aligned = R_world_alignment_ * R_current;
    
    double yaw = std::atan2(R_aligned(1,0), R_aligned(0,0));
    R_yaw_init_ = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    
    is_initialized_ = true;
}

std::vector<cv::Point2f> CoordConverter::projectWorldToImage(const Eigen::Vector3d& world_point) const {
    std::vector<cv::Point2f> image_points;
    
    // 第一步：将世界坐标转换到相机坐标系
    // 这里直接复用了已有的 worldToCamera 函数
    Eigen::Vector3d point_camera = worldToCamera(world_point);
    
    // 检查点是否在相机前方（Z坐标必须为正）
    if (point_camera.z() <= 0) {
        utils::logger()->warn("目标点在相机后方，无法投影到图像平面");
        return image_points;  // 返回空vector
    }
    
    // 第二步：使用相机内参矩阵进行透视投影
    // 投影公式： u = fx * X/Z + cx,  v = fy * Y/Z + cy
    double fx = camera_matrix_.at<double>(0, 0);  // 焦距 x
    double fy = camera_matrix_.at<double>(1, 1);  // 焦距 y  
    double cx = camera_matrix_.at<double>(0, 2);  // 主点 x
    double cy = camera_matrix_.at<double>(1, 2);  // 主点 y
    
    // 归一化平面坐标
    double x_normalized = point_camera.x() / point_camera.z();
    double y_normalized = point_camera.y() / point_camera.z();
    
    // 投影到像素坐标
    double u = fx * x_normalized + cx;
    double v = fy * y_normalized + cy;

    image_points.push_back(cv::Point2f(u, v));
    
    return image_points;
}


Eigen::Vector3d CoordConverter::cameraToIMU(const Eigen::Vector3d& point_camera) const {
    return R_camera_to_imu.transpose() * point_camera + t_camera_to_imu;
}

Eigen::Vector3d CoordConverter::imuToWorld(const Eigen::Vector3d& point_imu) const {
    if (!is_initialized_) {
        return point_imu;
    }
    return R_imu_to_world_ * point_imu;
}

Eigen::Vector3d CoordConverter::cameraToWorld(const Eigen::Vector3d& point_camera) const {
    Eigen::Vector3d point_imu = cameraToIMU(point_camera);
    return imuToWorld(point_imu);
}

Eigen::Vector3d CoordConverter::worldToIMU(const Eigen::Vector3d& point_world) const {
    if (!is_initialized_) {
        return point_world;
    }
    return R_imu_to_world_.transpose() * point_world;
}

Eigen::Vector3d CoordConverter::imuToCamera(const Eigen::Vector3d& point_imu) const {
    return R_camera_to_imu * (point_imu - t_camera_to_imu);
}

Eigen::Vector3d CoordConverter::worldToCamera(const Eigen::Vector3d& point_world) const {
    Eigen::Vector3d point_imu = worldToIMU(point_world);
    return imuToCamera(point_imu);
}

Eigen::Vector3d CoordConverter::extractGravityFromIMU(const Eigen::Quaterniond& q_imu) const {
    Eigen::Vector3d gravity_world(0, 0, -9.8);
    Eigen::Matrix3d R_imu = q_imu.toRotationMatrix();
    return R_imu.transpose() * gravity_world;
}

Eigen::Matrix3d CoordConverter::computeWorldAlignment(const Eigen::Vector3d& gravity_in_imu) const {
    Eigen::Vector3d target_gravity(0, 0, -9.8);
    Eigen::Vector3d g_normalized = gravity_in_imu.normalized();
    Eigen::Vector3d t_normalized = target_gravity.normalized();
    
    Eigen::Vector3d axis = g_normalized.cross(t_normalized);
    
    if (axis.norm() < 1e-6) {
        if (g_normalized.dot(t_normalized) > 0) {
            return Eigen::Matrix3d::Identity();
        } else {
            if (std::abs(g_normalized.x()) < 0.9) {
                axis = Eigen::Vector3d::UnitX();
            } else {
                axis = Eigen::Vector3d::UnitY();
            }
            return Eigen::AngleAxisd(M_PI, axis).toRotationMatrix();
        }
    }
    
    axis.normalize();
    double angle = std::acos(g_normalized.dot(t_normalized));
    
    Eigen::Matrix3d K;
    K << 0, -axis.z(), axis.y(),
         axis.z(), 0, -axis.x(),
         -axis.y(), axis.x(), 0;
    
    return Eigen::Matrix3d::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
}

bool CoordConverter::loadCalibrationFromYAML(const std::string& yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        
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
        
        if (config["CalibParam"]["EXTRI"]) {
            auto extri = config["CalibParam"]["EXTRI"];
            
            if (extri["SO3_CmToBr"]) {
                auto so3_cm = extri["SO3_CmToBr"][0]["value"];
                double qx = so3_cm["qx"].as<double>();
                double qy = so3_cm["qy"].as<double>();
                double qz = so3_cm["qz"].as<double>();
                double qw = so3_cm["qw"].as<double>();
                
                Eigen::Quaterniond q_CmToBr(qw, qx, qy, qz);
                Eigen::Matrix3d R_CmToBr = q_CmToBr.toRotationMatrix();
                R_camera_to_imu = R_CmToBr.transpose();
            }
            
            if (extri["POS_CmInBr"]) {
                auto pos_cm = extri["POS_CmInBr"][0]["value"];
                double tx = pos_cm["r0c0"].as<double>();
                double ty = pos_cm["r1c0"].as<double>();
                double tz = pos_cm["r2c0"].as<double>();
                t_camera_to_imu = Eigen::Vector3d(tx, ty, tz);
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

std::vector<cv::Point2f> CoordConverter::reproject_armor(
    const Eigen::Vector3d & xyz_in_camera, double yaw, ArmorType type, ArmorName name) const
{

    auto pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;

    double roll = 0.0;

    // clang-format off
    Eigen::Matrix3d R = utils::rotation_matrix_yxz(yaw, pitch, roll);
    // clang-format on

    // get R_armor2camera t_armor2camera
    Eigen::Vector3d t_armor2camera = xyz_in_camera;

    // get rvec tvec
    cv::Vec3d rvec;
    cv::Mat R_armor2camera_cv;
    cv::eigen2cv(R, R_armor2camera_cv);
    cv::Rodrigues(R_armor2camera_cv, rvec);
    cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

    // reproject
    std::vector<cv::Point2f> image_points;
    
    // 需要确保 PW_BIG 和 PW_SMALL 已定义
    const auto & object_points = (type == ArmorType::big) ? PW_BIG : PW_SMALL;
    
    // 修正变量名：dist_coeffs_ 而不是 distort_coeffs_
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, dist_coeffs_, image_points);
    return image_points;
}

}