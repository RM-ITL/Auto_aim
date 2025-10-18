#ifndef GEOMETRIC_COMPENSATOR_HPP
#define GEOMETRIC_COMPENSATOR_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <yaml-cpp/yaml.h>
#include "module/coord_converter.hpp"  // 新增：引入坐标转换器

namespace aimer {

// 装甲板姿态估计结果（保持不变）
struct ArmorPoseEstimation {
    double z_to_v;                
    double zn_to_v;              
    double pitch;                 
    double roll;                  
    double reprojection_error;    
    bool success;                 
    int iterations;               
    double search_time_ms;        
    Eigen::Matrix3d R_armor_to_camera;
};

// 弹道补偿结果
struct BallisticCompensation {
    double shoot_angle;           
    double compensated_yaw;       
    double compensated_pitch;     
    double time_of_flight;        
    bool success;                 
    
    // 调试信息
    double target_distance;       
    double height_diff;          
    Eigen::Vector3d aim_point;   
    
    // 新增：补偿量（用于调试）
    double delta_yaw;            // 补偿的yaw增量
    double delta_pitch;          // 补偿的pitch增量
};

// 弹道计算函数对象（保持不变）
class ResistanceFuncLinear {
private:
    const double g = 9.8;
    const double w, h, v0;
    const double k;

public:
    ResistanceFuncLinear(double w, double h, double v0, double k) 
        : w(w), h(h), v0(v0), k(k) {}

    template<typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = (k * this->v0 * ceres::sin(x[0]) + this->g) * k * this->w
                    / (k * k * this->v0 * ceres::cos(x[0]))
                    + this->g * ceres::log(T(1.0) - (k * this->w) / (this->v0 * ceres::cos(x[0]))) / k / k
                    - this->h;
        return true;
    }
};

class GeometricCompensator {
public:
    GeometricCompensator() = default;
    
    // 从YAML文件加载参数
    bool loadParametersFromYAML(const std::string& yaml_path);
    
    // 设置相机内参
    void setCameraMatrix(const cv::Mat& K) { 
        K_ = K.clone(); 
        if (K_.type() != CV_64F) {
            K_.convertTo(K_, CV_64F);
        }
    }
    
    // 新增：设置坐标转换器（共享指针，避免生命周期问题）
    void setCoordConverter(std::shared_ptr<CoordConverter> coord_converter) {
        coord_converter_ = coord_converter;
    }
    
    // ========== 核心功能1：装甲板朝向估计（保持不变） ==========
    ArmorPoseEstimation estimateArmorOrientation(
        const Eigen::Vector3d& tvec_camera,      
        const std::vector<cv::Point2f>& corners, 
        bool is_big_armor,                       
        double initial_guess = 0.0               
    );
    
    // ========== 核心功能2：弹道补偿计算 ==========
    
    // 原有功能：相机坐标系下的弹道补偿（保持向后兼容）
    BallisticCompensation calculateBallisticCompensation(
        const Eigen::Vector3d& target_position_camera,
        double current_yaw,
        double current_pitch
    );
    
    // 新增功能：世界坐标系下的弹道补偿（简化版，利用coord_converter）
    BallisticCompensation calculateBallisticCompensationWorld(
        const Eigen::Vector3d& target_position_world,    // 世界坐标系位置（毫米）
        double current_world_yaw,                         // 当前世界坐标系yaw（弧度）
        double current_world_pitch                        // 当前世界坐标系pitch（弧度）
    );
    
    // ========== 辅助函数 ==========
    
    // 相机坐标到枪管坐标转换（考虑相机到枪管的固定偏移）
    Eigen::Vector3d cameraToBarrel(const Eigen::Vector3d& point_camera) const;
    
    // 枪管坐标到相机坐标转换
    Eigen::Vector3d barrelToCamera(const Eigen::Vector3d& point_barrel) const;
    
    // 获取相机z轴在xy平面的投影方向
    Eigen::Vector2d getCameraZ2D() const {
        return Eigen::Vector2d(0.0, 1.0);
    }

private:
    cv::Mat K_;  // 相机内参
    
    // 新增：坐标转换器指针
    std::shared_ptr<CoordConverter> coord_converter_;
    
    // ========== 装甲板参数 ==========
    double armor_small_width_ = 135.0;    
    double armor_small_height_ = 125.0;   
    double armor_big_width_ = 230.0;      
    double armor_big_height_ = 55.0;      
    
    // 装甲板姿态约束
    double armor_fixed_pitch_ = 0.0;      
    double armor_fixed_roll_ = 0.0;       
    bool use_fixed_pitch_ = false;        
    
    // ========== 搜索参数 ==========
    double search_range_ = M_PI;          
    double search_epsilon_ = 0.001;       
    int max_iterations_ = 30;             
    
    // ========== 优化参数 ==========
    double max_reprojection_error_ = 100.0;  
    double pixel_error_weight_ = 1.0;        
    double edge_error_weight_ = 2.0;         
    double angle_error_weight_ = 1.5;        
    
    // ========== 弹道参数 ==========
    double bullet_speed_ = 16.0;             
    double resistance_k_ = 0.022928514188;   
    double initial_shoot_angle_deg_ = 30.0;  
    
    // ========== 相机到枪管的偏移量（毫米）==========
    double camera_to_barrel_x_ = 0.0;
    double camera_to_barrel_y_ = 100.0;  
    double camera_to_barrel_z_ = 0.0;
    
    // ========== 私有辅助函数（保持不变）==========
    double computeReprojectionError(
        const Eigen::Vector3d& tvec_camera,
        const std::vector<cv::Point2f>& detected_corners,
        double z_to_v,
        double pitch,
        bool is_big_armor
    ) const;
    
    std::vector<Eigen::Vector3d> generateArmorCorners(bool is_big_armor) const;
    
    std::vector<Eigen::Vector3d> transformCornersToCameraFrame(
        const std::vector<Eigen::Vector3d>& local_corners,
        const Eigen::Vector3d& tvec_camera,
        double z_to_v,
        double pitch,
        double roll
    ) const;
    
    struct DetailedError {
        double pixel_error;
        double edge_error;
        double angle_error;
        double total_weighted;
    };
    
    DetailedError computeDetailedReprojectionError(
        const std::vector<cv::Point2f>& projected_corners,
        const std::vector<cv::Point2f>& detected_corners
    ) const;
    
    std::pair<double, double> getArmorDimensions(bool is_big_armor) const {
        if (is_big_armor) {
            return {armor_big_width_, armor_big_height_};
        } else {
            return {armor_small_width_, armor_small_height_};
        }
    }
    
    double solveShootAngle(double horizontal_distance, double height_diff, double bullet_speed);
    
    double normalizeAngle(double angle) const {
        while (angle > M_PI) angle -= 2 * M_PI;
        while (angle < -M_PI) angle += 2 * M_PI;
        return angle;
    }
};

} // namespace aimer

#endif