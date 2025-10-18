#include "module/optimize_yaw.hpp"
#include <opencv2/core/eigen.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <limits>

namespace solver {

YawOptimizer::YawOptimizer(const std::string& yaml_config_path)
    : search_range_(70.0),
      search_step_(1.0),
      last_optimization_error_(0.0),
      last_optimized_yaw_(0.0) {
    
    camera_matrix_ = cv::Mat();
    dist_coeffs_ = cv::Mat();
    
    if (!loadCameraParamsFromYAML(yaml_config_path)) {
        throw std::runtime_error("YawOptimizer: 无法从配置文件加载相机参数: " + yaml_config_path);
    }
}

YawOptimizer::YawOptimizer(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
    : camera_matrix_(camera_matrix.clone()),
      dist_coeffs_(dist_coeffs.clone()),
      search_range_(70.0),
      search_step_(1.0),
      last_optimization_error_(0.0),
      last_optimized_yaw_(0.0) {
    
    if (camera_matrix_.type() != CV_64F) {
        camera_matrix_.convertTo(camera_matrix_, CV_64F);
    }
    if (!dist_coeffs_.empty() && dist_coeffs_.type() != CV_64F) {
        dist_coeffs_.convertTo(dist_coeffs_, CV_64F);
    }
}

bool YawOptimizer::loadCameraParamsFromYAML(const std::string& yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        
        if (!config["CalibParam"]["INTRI"]["Camera"]) {
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
        
        camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
        camera_matrix_.at<double>(0, 0) = focal_length[0];
        camera_matrix_.at<double>(1, 1) = focal_length[1];
        camera_matrix_.at<double>(0, 2) = principal_point[0];
        camera_matrix_.at<double>(1, 2) = principal_point[1];
        camera_matrix_.at<double>(2, 2) = 1.0;
        
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
        
        return true;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("YawOptimizer"), 
                    "加载相机参数失败: %s", e.what());
        return false;
    }
}

Eigen::Matrix3d YawOptimizer::optimizeWithPrior(
    const Eigen::Vector3d& camera_position,
    const Eigen::Matrix3d& prior_R_armor_to_camera,
    ArmorType armor_type,
    ArmorName armor_name,
    const std::vector<cv::Point2f>& detected_corners,
    const CoordConverter& converter) {
    
    double prior_yaw, prior_pitch, prior_roll;
    extractWorldEulerAngles(prior_R_armor_to_camera, prior_yaw, prior_pitch, prior_roll);
    
    auto pitch_to_use = (armor_name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
    double roll_to_use = 0.0;
    
    double best_yaw = prior_yaw;
    double min_error = 1e10;
    
    double search_range_rad = search_range_ * M_PI / 180.0;
    double search_step_rad = search_step_ * M_PI / 180.0;
        
    Eigen::Matrix3d constrained_prior = utils::rotation_matrix_yxz(prior_yaw, pitch_to_use, roll_to_use); // 构建旋转矩阵
    double prior_error = calculateReprojectionError(
        camera_position,
        constrained_prior,
        armor_type,
        detected_corners
    );
    
    if (prior_error < min_error) {
        min_error = prior_error;
        best_yaw = prior_yaw;
    }
    
    for (double yaw = 0 - search_range_rad;
         yaw <= 0 + search_range_rad;
         yaw += search_step_rad) {
        
        Eigen::Matrix3d test_rotation = utils::rotation_matrix_yxz(yaw, pitch_to_use, roll_to_use);
        
        double error = calculateReprojectionError(
            camera_position,
            test_rotation,
            armor_type,
            detected_corners
        );
        
        if (error < min_error) {
            min_error = error;
            best_yaw = yaw;
        }
    }
    
    last_optimization_error_ = min_error;
    last_optimized_yaw_ = best_yaw;     // 最优yaw
    
    return utils::rotation_matrix_yxz(best_yaw, pitch_to_use, roll_to_use);
}

void YawOptimizer::extractWorldEulerAngles(const Eigen::Matrix3d& R, 
                                           double& yaw, double& pitch, double& roll) const {
    Eigen::Vector3d angles = utils::eulers_yxz(R);
    
    yaw = angles[0] * 180.0 / CV_PI;       // 对应真实yaw
    pitch = angles[1] * 180.0 / CV_PI;        // 对应真实pitch
    roll = angles[2] * 180.0 / CV_PI;      // 对应真实roll

    // utils::logger()->debug(
    //     "提取得到的角度yaw:{:.2f}, pitch:{:.2f},  roll:{:.2f}",
    //     yaw,
    //     pitch,
    //     roll
    // );
    
}


double YawOptimizer::calculateReprojectionError(
    const Eigen::Vector3d& camera_position,
    const Eigen::Matrix3d& R_armor_to_camera,
    ArmorType armor_type,
    const std::vector<cv::Point2f>& detected_corners) const {
    
    cv::Mat rmat_cv;
    cv::eigen2cv(R_armor_to_camera, rmat_cv);
    
    cv::Mat rvec;
    cv::Rodrigues(rmat_cv, rvec);
    
    cv::Mat tvec;
    cv::eigen2cv(camera_position, tvec);
    
    const auto& object_points = (armor_type == ArmorType::big) ? 
                                PW_BIG : PW_SMALL;
    
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, rvec, tvec, 
                      camera_matrix_, dist_coeffs_, projected_points);
    
    double total_error = 0.0;
    for (size_t i = 0; i < 4; i++) {
        double dx = projected_points[i].x - detected_corners[i].x;
        double dy = projected_points[i].y - detected_corners[i].y;
        double error = std::sqrt(dx * dx + dy * dy);
        total_error += error;
    }
    
    return total_error / 4.0;
}


} // namespace solver