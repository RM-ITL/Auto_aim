#include "module/optimize_yaw.hpp"
#include <opencv2/core/eigen.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <limits>

namespace solver {

YawOptimizer::YawOptimizer(const std::string& yaml_config_path, CoordConverter* CoordConverter_)
    : CoordConverter_(CoordConverter_),
      camera_matrix_(),
      dist_coeffs_(),
      search_range_(70.0),
      search_step_(1.0),
      last_optimization_error_(0.0),
      last_optimized_yaw_(0.0) {
    
    if (!loadCameraParamsFromYAML(yaml_config_path)) {
        throw std::runtime_error("YawOptimizer: 无法从配置文件加载相机参数: " + yaml_config_path);
    }

}

YawOptimizer::YawOptimizer(
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    CoordConverter* CoordConverter_)
    : CoordConverter_(CoordConverter_),
      camera_matrix_(camera_matrix.clone()),
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

Eigen::Matrix3d YawOptimizer::optimize_yaw(
    const Eigen::Vector3d& world_position,
    const Eigen::Matrix3d& R_armor_to_world,
    ArmorType armor_type,
    ArmorName armor_name,
    const std::vector<cv::Point2f>& detected_corners,
    [[maybe_unused]] const CoordConverter& converter) {
    
    double prior_yaw, prior_pitch, prior_roll;
    extractWorldEulerAngles(R_armor_to_world, prior_yaw, prior_pitch, prior_roll);
    
    auto pitch_to_use = (armor_name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
    double roll_to_use = 0.0;
    
    double best_yaw = prior_yaw;
    double min_error = 1e10;
    
    double search_range_rad = search_range_ * M_PI / 180.0;
    double search_step_rad = search_step_ * M_PI / 180.0;
        
    Eigen::Matrix3d constrained_prior = utils::rotation_matrix_zyx(prior_yaw, pitch_to_use, roll_to_use); // 构建旋转矩阵
    // 这没问题

    // utils::logger()->debug("构建的装甲板到相机的旋转矩阵是");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //        constrained_prior(i,0), constrained_prior(i,1), constrained_prior(i,2));
    // }

    double prior_error = ReprojectionError(
        world_position,
        constrained_prior,
        armor_type,
        detected_corners
    );
    
    if (prior_error < min_error) {
        min_error = prior_error;
        best_yaw = prior_yaw;
    }
    
    for (double yaw = prior_yaw - search_range_rad;
         yaw <= prior_yaw + search_range_rad;
         yaw += search_step_rad) {
        
        Eigen::Matrix3d test_rotation = utils::rotation_matrix_zyx(yaw, pitch_to_use, roll_to_use);
        
        double error = ReprojectionError(
            world_position,
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
    
    return utils::rotation_matrix_zyx(best_yaw, pitch_to_use, roll_to_use);
}

void YawOptimizer::extractWorldEulerAngles(const Eigen::Matrix3d& R, 
                                           double& yaw, double& pitch, double& roll) const {
    Eigen::Vector3d angles = utils::eulers_zyx(R);
    
    yaw = angles[0];       // 对应真实yaw
    pitch = angles[1];        // 对应真实pitch
    roll = angles[2];      // 对应真实roll

    // utils::logger()->debug(
    //     "提取得到的角度yaw:{:.2f}, pitch:{:.2f},  roll:{:.2f}",
    //     yaw,
    //     pitch,
    //     roll
    // );
    
}

std::vector<cv::Point2f> YawOptimizer::reproject_armor(
    const Eigen::Vector3d & xyz_in_world, const Eigen::Matrix3d& R , ArmorType type) const
{


    // clang-format off
    Eigen::Matrix3d R_armor_to_world = R;
    Eigen::Matrix3d R_armor_to_camera = CoordConverter_->transformRotation(
        R_armor_to_world,
        CoordinateFrame::WORLD,
        CoordinateFrame::CAMERA
    );

    // utils::logger()->debug("计算的装甲板到相机的旋转矩阵是");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         R_armor_to_camera(i,0), R_armor_to_camera(i,1), R_armor_to_camera(i,2));
    // }

    // clang-format on
    Eigen::Vector3d t_armor_to_camera = CoordConverter_->transform(
        xyz_in_world,
        solver::CoordinateFrame::WORLD,
        solver::CoordinateFrame::CAMERA
    );
    // get R_armor2camera t_armor_to_camera

    // utils::logger()->debug(
    //     "解算得到的世界坐标系下目标的位置为x:{:.2f}, y:{:.2f}, z:{:.2f}",
    //     t_armor_to_camera[0],
    //     t_armor_to_camera[1],
    //     t_armor_to_camera[2]
    // );


    // get rvec tvec
    cv::Vec3d rvec;
    cv::Mat R_armor_to_camera_cv;
    cv::eigen2cv(R_armor_to_camera, R_armor_to_camera_cv);
    cv::Rodrigues(R_armor_to_camera_cv, rvec);
    cv::Vec3d tvec(
        t_armor_to_camera[0] * 1000.0,
        t_armor_to_camera[1] * 1000.0,
        t_armor_to_camera[2] * 1000.0
    );

    // reproject
    std::vector<cv::Point2f> image_points;
    
    // utils::logger()->debug("旋转向量 rvec: [{:.6f}, {:.6f}, {:.6f}]", 
    //                   rvec[0], rvec[1], rvec[2]);
    // utils::logger()->debug("平移向量 tvec: [{:.6f}, {:.6f}, {:.6f}]", 
    //                     tvec[0], tvec[1], tvec[2]);

    // 需要确保 PW_BIG 和 PW_SMALL 已定义
    const auto & object_points = (type == ArmorType::big) ? PW_BIG : PW_SMALL;

    
    // 修正变量名：dist_coeffs_ 而不是 distort_coeffs_
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, dist_coeffs_, image_points);
    return image_points;
}


double YawOptimizer::ReprojectionError(
    const Eigen::Vector3d& world_position,
    const Eigen::Matrix3d& R_armor_to_world,
    ArmorType armor_type,
    const std::vector<cv::Point2f>& detected_corners) const {

    // utils::logger()->debug("构建的装甲板到相机的旋转矩阵是");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         R_armor_to_world(i,0), R_armor_to_world(i,1), R_armor_to_world(i,2));
    // }
    
    std::vector<cv::Point2f> image_points = reproject_armor(world_position, R_armor_to_world, armor_type); 

    // utils::logger()->debug(
    //     "图像点的数据是：[({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f}), ({:.2f}, {:.2f})]",
    //     image_points[0].x, image_points[0].y,
    //     image_points[1].x, image_points[1].y,
    //     image_points[2].x, image_points[2].y,
    //     image_points[3].x, image_points[3].y
    // );
    
    double total_error = 0.0;
    for (size_t i = 0; i < 4; i++) {
        double dx = image_points[i].x - detected_corners[i].x;
        double dy = image_points[i].y - detected_corners[i].y;
        double error = std::sqrt(dx * dx + dy * dy);
        total_error += error;
    }
    
    return total_error / 4.0;
}

std::vector<cv::Point2f> YawOptimizer::reproject_armor_out(
    const Eigen::Vector3d & xyz_in_world, double yaw , ArmorType type, ArmorName name) const
{

    auto pitch_to_use = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
    double roll_to_use = 0.0;

    Eigen::Matrix3d R_armor_to_world = utils::rotation_matrix_zyx(yaw, pitch_to_use, roll_to_use); // 构建旋转矩阵
    // clang-format off
    Eigen::Matrix3d R_armor_to_camera = CoordConverter_->transformRotation(
        R_armor_to_world,
        CoordinateFrame::WORLD,
        CoordinateFrame::CAMERA
    );

    // utils::logger()->debug("计算的装甲板到相机的旋转矩阵是");

    // clang-format on
    Eigen::Vector3d t_armor_to_camera = CoordConverter_->transform(
        xyz_in_world,
        solver::CoordinateFrame::WORLD,
        solver::CoordinateFrame::CAMERA
    );
    // get R_armor2camera t_armor_to_camera

    // get rvec tvec
    cv::Vec3d rvec;
    cv::Mat R_armor_to_camera_cv;
    cv::eigen2cv(R_armor_to_camera, R_armor_to_camera_cv);
    cv::Rodrigues(R_armor_to_camera_cv, rvec);
    cv::Vec3d tvec(
        t_armor_to_camera[0] * 1000.0,
        t_armor_to_camera[1] * 1000.0,
        t_armor_to_camera[2] * 1000.0
    );

    // reproject
    std::vector<cv::Point2f> image_points;


    // 需要确保 PW_BIG 和 PW_SMALL 已定义
    const auto & object_points = (type == ArmorType::big) ? PW_BIG : PW_SMALL;

    
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, dist_coeffs_, image_points);
    return image_points;
}


} // namespace solver
