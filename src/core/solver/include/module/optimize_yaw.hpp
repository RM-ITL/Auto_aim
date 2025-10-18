#ifndef YAW_OPTIMIZER_HPP
#define YAW_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "solver.hpp"
#include "coord_converter.hpp"
#include "common/armor.hpp"
#include "math_tools.hpp"
#include "logger.hpp"


namespace solver {

using ArmorName = armor_auto_aim::ArmorName;
using ArmorType = armor_auto_aim::ArmorType;

class YawOptimizer {
public:
    explicit YawOptimizer(const std::string& yaml_config_path);
    YawOptimizer(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);
    
    Eigen::Matrix3d optimizeWithPrior(
        const Eigen::Vector3d& camera_position,
        const Eigen::Matrix3d& prior_R_armor_to_camera,
        ArmorType armor_type,
        ArmorName armor_name,
        const std::vector<cv::Point2f>& detected_corners,
        const CoordConverter& converter);
    
    void setSearchRange(double range_degrees) { search_range_ = range_degrees; }
    void setSearchStep(double step_degrees) { search_step_ = step_degrees; }
    
    double getLastOptimizationError() const { return last_optimization_error_; }
    double getLastOptimizedYaw() const { return last_optimized_yaw_; }
    
    bool isInitialized() const { 
        return !camera_matrix_.empty() && !dist_coeffs_.empty(); 
    }
     
private:
    bool loadCameraParamsFromYAML(const std::string& yaml_path);
    
    void extractWorldEulerAngles(const Eigen::Matrix3d& R_world_to_armor, 
                                 double& yaw, double& pitch, double& roll) const; 
    
    double calculateReprojectionError(
        const Eigen::Vector3d& camera_position,
        const Eigen::Matrix3d& R_armor_to_camera,
        ArmorType armor_type,
        const std::vector<cv::Point2f>& detected_corners) const;
    
    
private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double search_range_;
    double search_step_;
    mutable double last_optimization_error_;
    mutable double last_optimized_yaw_;
};

} // namespace solver

#endif