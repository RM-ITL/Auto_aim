#include "module/pnp_solver.hpp"
#include <opencv2/core/eigen.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <cmath>

namespace solver {

PnPSolver::PnPSolver(const std::string& yaml_config_path) {
    camera_matrix_ = cv::Mat();
    dist_coeffs_ = cv::Mat();
    
    if (!loadCameraParamsFromYAML(yaml_config_path)) {
        throw std::runtime_error("PnPSolver: 无法从配置文件加载相机参数: " + yaml_config_path);
    }
    
    RCLCPP_INFO(rclcpp::get_logger("PnPSolver"), 
                "成功从配置文件初始化PnP求解器: %s", 
                yaml_config_path.c_str());
}

PnPSolver::PnPSolver() {
    camera_matrix_ = cv::Mat();
    dist_coeffs_ = cv::Mat();
}

void PnPSolver::setCameraMatrix(const cv::Mat& camera_matrix) {
    camera_matrix_ = camera_matrix.clone();
    
    if (camera_matrix_.type() != CV_64F) {
        camera_matrix_.convertTo(camera_matrix_, CV_64F);
    }
}

void PnPSolver::setDistortionCoeffs(const cv::Mat& dist_coeffs) {
    dist_coeffs_ = dist_coeffs.clone();
    
    if (dist_coeffs_.type() != CV_64F) {
        dist_coeffs_.convertTo(dist_coeffs_, CV_64F);
    }
}

bool PnPSolver::loadCameraParamsFromYAML(const std::string& yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        
        if (!config["CalibParam"]["INTRI"]["Camera"]) {
            RCLCPP_ERROR(rclcpp::get_logger("PnPSolver"), 
                        "YAML文件缺少Camera节点");
            return false;
        }
        
        auto camera_node = config["CalibParam"]["INTRI"]["Camera"][0]["value"];
        
        YAML::Node camera_params;
        if (camera_node["ptr_wrapper"] && camera_node["ptr_wrapper"]["data"]) {
            camera_params = camera_node["ptr_wrapper"]["data"];
        } else {
            camera_params = camera_node;
        }
        
        std::vector<double> focal_length = camera_params["focal_length"].as<std::vector<double>>();
        std::vector<double> principal_point = camera_params["principal_point"].as<std::vector<double>>();
        
        if (focal_length.size() < 2 || principal_point.size() < 2) {
            return false;
        }
        
        // 构建内参矩阵
        camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        camera_matrix_.at<double>(0, 0) = focal_length[0];
        camera_matrix_.at<double>(1, 1) = focal_length[1];
        camera_matrix_.at<double>(0, 2) = principal_point[0];
        camera_matrix_.at<double>(1, 2) = principal_point[1];
        camera_matrix_.at<double>(2, 2) = 1.0;
        
        // 加载畸变系数
        if (camera_params["disto_param"]) {
            std::vector<double> disto_param = camera_params["disto_param"].as<std::vector<double>>();
            
            if (!disto_param.empty()) {
                int coeff_count = std::min(5, static_cast<int>(disto_param.size()));
                dist_coeffs_ = cv::Mat::zeros(1, coeff_count, CV_64F);
                
                for (int i = 0; i < coeff_count; i++) {
                    dist_coeffs_.at<double>(0, i) = disto_param[i];
                }
            }
        } else {
            dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
        }
        
        RCLCPP_INFO(rclcpp::get_logger("PnPSolver"), 
                   "相机参数加载成功 - 焦距:[%.1f, %.1f]",
                   focal_length[0], focal_length[1]);
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("PnPSolver"), 
                    "加载相机参数失败: %s", e.what());
        return false;
    }
}

solver::PnPResult PnPSolver::solvePnP(const std::vector<cv::Point2f>& corners, 
                              ArmorType armor_type,
                              double timestamp) {
    solver::PnPResult result;
    result.success = false;
    
    try {
        if (corners.size() != 4 || camera_matrix_.empty()) {
            return result;
        }
        
        // 去畸变
        std::vector<cv::Point2f> undistorted_corners = undistortPoints(corners);
        
        // 选择世界坐标点
        const auto& world_points_float = (armor_type == ArmorType::big) ? PW_BIG : PW_SMALL;
        
        cv::Mat camera_matrix_f;
        if (camera_matrix_.type() != CV_32F) {
            camera_matrix_.convertTo(camera_matrix_f, CV_32F);
        } else {
            camera_matrix_f = camera_matrix_;
        }
        
        // PnP求解
        cv::Mat rvec, tvec;
        bool pnp_success = cv::solvePnP(
            world_points_float,
            undistorted_corners,
            camera_matrix_f,
            cv::Mat(),
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE
        );
        
        if (!pnp_success) {
            return result;
        }
        
        // 转换结果
        cv::Mat rvec_d, tvec_d;
        if (rvec.type() != CV_64F) {
            rvec.convertTo(rvec_d, CV_64F);
        } else {
            rvec_d = rvec;
        }
        if (tvec.type() != CV_64F) {
            tvec.convertTo(tvec_d, CV_64F);
        } else {
            tvec_d = tvec;
        }
        
        cv::cv2eigen(tvec_d, result.camera_position);
        
        cv::Mat rmat;
        cv::Rodrigues(rvec_d, rmat);
        cv::cv2eigen(rmat, result.R_armor_to_camera);

        // utils::logger()->debug("PnP求解成功");
        // utils::logger()->debug("旋转向量 - x:{:.3f}, y:{:.3f}, z:{:.3f}", 
        //     rvec_d.at<double>(0), rvec_d.at<double>(1), rvec_d.at<double>(2));
            
        // utils::logger()->debug("旋转矩阵：");
        // for (int i = 0; i < 3; ++i) {
        //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
        //         result.R_armor_to_camera.at<double>(i,0), result.R_armor_to_camera.at<double>(i,1), result.R_armor_to_camera.at<double>(i,2));
        // }
        // utils::logger()->debug("旋转矩阵：");
        // for (int i = 0; i < 3; ++i) {
        //     // 使用 Eigen 的括号运算符来访问矩阵元素
        //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
        //         result.R_armor_to_camera(i,0), result.R_armor_to_camera(i,1), result.R_armor_to_camera(i,2));
        // }

    
        // 计算球坐标
        double distance = result.camera_position.norm();
        if (distance > 0) {
            Eigen::Vector3d ypd = utils::xyz2ypd(result.camera_position);
            result.camera_spherical.yaw = ypd[0];
            result.camera_spherical.pitch = ypd[1];
            result.camera_spherical.distance = ypd[2];
            result.camera_spherical.timestamp = timestamp;
        }
        
        result.reprojection_error = calculateReprojectionError(
            world_points_float, undistorted_corners, rvec, tvec);
        
        // utils::logger()->debug(
        //     "重投影误差是:{:.2f}",
        //     result.reprojection_error
        // );
        
        result.success = validatePnPResult(result);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("PnPSolver"), 
                    "PnP求解异常: %s", e.what());
    }
    
    return result;
}

std::vector<cv::Point2f> PnPSolver::undistortPoints(const std::vector<cv::Point2f>& distorted_points) {
    std::vector<cv::Point2f> undistorted_points;
    
    if (dist_coeffs_.empty() || cv::countNonZero(dist_coeffs_) == 0) {
        return distorted_points;
    }
    
    cv::undistortPoints(distorted_points, undistorted_points,
                        camera_matrix_, dist_coeffs_,
                        cv::Mat(), camera_matrix_);
    
    return undistorted_points;
}

double PnPSolver::calculateReprojectionError(const std::vector<cv::Point3f>& world_points,
                                            const std::vector<cv::Point2f>& image_points,
                                            const cv::Mat& rvec,
                                            const cv::Mat& tvec) const {
    std::vector<cv::Point2f> reprojected_points;
    
    cv::Mat camera_matrix_f;
    if (camera_matrix_.type() != CV_32F) {
        camera_matrix_.convertTo(camera_matrix_f, CV_32F);
    } else {
        camera_matrix_f = camera_matrix_;
    }
    
    cv::projectPoints(world_points, rvec, tvec, 
                     camera_matrix_f, cv::Mat(), reprojected_points);
    
    double total_error = 0;
    for (size_t i = 0; i < image_points.size(); i++) {
        double dx = reprojected_points[i].x - image_points[i].x;
        double dy = reprojected_points[i].y - image_points[i].y;
        double error = std::sqrt(dx * dx + dy * dy);
        total_error += error;
    }
    
    return total_error / image_points.size();
}

bool PnPSolver::validatePnPResult(const solver::PnPResult& result) const {
    if (result.camera_spherical.distance < min_distance_ || 
        result.camera_spherical.distance > max_distance_) {
        return false;
    }
    
    if (result.reprojection_error > max_reprojection_error_) {
        return false;
    }
    
    if (!std::isfinite(result.camera_position.x()) || 
        !std::isfinite(result.camera_position.y()) || 
        !std::isfinite(result.camera_position.z())) {
        return false;
    }
    
    double det = result.R_armor_to_camera.determinant();
    if (std::abs(det - 1.0) > 0.01) {
        return false;
    }
    
    return true;
}


} // namespace solver