#ifndef PNP_SOLVER_HPP
#define PNP_SOLVER_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include "solver.hpp"
#include "math_tools.hpp"
#include "logger.hpp"

namespace solver {

using ArmorType = armor_auto_aim::ArmorType;

class PnPSolver {
public:
    explicit PnPSolver(const std::string& yaml_config_path);
    PnPSolver();
    
    void setCameraMatrix(const cv::Mat& camera_matrix);
    void setDistortionCoeffs(const cv::Mat& dist_coeffs);
    
    solver::PnPResult solvePnP(const std::vector<cv::Point2f>& corners, 
                       ArmorType armor_type,
                       double timestamp = 0.0);
    
    bool isInitialized() const { 
        return !camera_matrix_.empty() && !dist_coeffs_.empty(); 
    }
    
private:
    bool loadCameraParamsFromYAML(const std::string& yaml_path);
    std::vector<cv::Point2f> undistortPoints(const std::vector<cv::Point2f>& distorted_points);
    bool validatePnPResult(const solver::PnPResult& result) const;
    double calculateReprojectionError(const std::vector<cv::Point3f>& world_points,
                                     const std::vector<cv::Point2f>& image_points,
                                     const cv::Mat& rvec,
                                     const cv::Mat& tvec) const;

private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    
    const double max_distance_ = 10000.0;      // mm
    const double min_distance_ = 100.0;        // mm  
    const double max_reprojection_error_ = 5.0; // pixels
};

} // namespace solver

#endif