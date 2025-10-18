#include "module/geometric_compensator.hpp"
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <iostream>
#include <chrono>

namespace aimer {

bool GeometricCompensator::loadParametersFromYAML(const std::string& yaml_path) {
    try {
        YAML::Node config = YAML::LoadFile(yaml_path);
        
        if (config["geometric_compensator"]) {
            YAML::Node gc_config = config["geometric_compensator"];
            
            // 加载装甲板尺寸参数
            if (gc_config["armor_dimensions"]) {
                if (gc_config["armor_dimensions"]["small_armor"]) {
                    armor_small_width_ = gc_config["armor_dimensions"]["small_armor"]["width"].as<double>();
                    armor_small_height_ = gc_config["armor_dimensions"]["small_armor"]["height"].as<double>();
                }
                if (gc_config["armor_dimensions"]["big_armor"]) {
                    armor_big_width_ = gc_config["armor_dimensions"]["big_armor"]["width"].as<double>();
                    armor_big_height_ = gc_config["armor_dimensions"]["big_armor"]["height"].as<double>();
                }
            }
            
            // 加载装甲板姿态参数
            if (gc_config["armor_pose"]) {
                YAML::Node ap = gc_config["armor_pose"];
                if (ap["use_fixed_pitch"]) use_fixed_pitch_ = ap["use_fixed_pitch"].as<bool>();
                if (ap["fixed_pitch"]) armor_fixed_pitch_ = ap["fixed_pitch"].as<double>() * M_PI / 180.0;
                if (ap["fixed_roll"]) armor_fixed_roll_ = ap["fixed_roll"].as<double>() * M_PI / 180.0;
                if (ap["search_range"]) search_range_ = ap["search_range"].as<double>() * M_PI / 180.0;
                if (ap["search_epsilon"]) search_epsilon_ = ap["search_epsilon"].as<double>() * M_PI / 180.0;
            }
            
            // 加载优化参数
            if (gc_config["optimization"]) {
                YAML::Node opt = gc_config["optimization"];
                if (opt["max_reprojection_error"]) max_reprojection_error_ = opt["max_reprojection_error"].as<double>();
                if (opt["max_iterations"]) max_iterations_ = opt["max_iterations"].as<int>();
                if (opt["pixel_error_weight"]) pixel_error_weight_ = opt["pixel_error_weight"].as<double>();
                if (opt["edge_error_weight"]) edge_error_weight_ = opt["edge_error_weight"].as<double>();
                if (opt["angle_error_weight"]) angle_error_weight_ = opt["angle_error_weight"].as<double>();
            }
            
            // 加载弹道参数
            if (gc_config["ballistic_params"]) {
                YAML::Node bp = gc_config["ballistic_params"];
                if (bp["bullet_speed"]) bullet_speed_ = bp["bullet_speed"].as<double>();
                if (bp["resistance_k"]) resistance_k_ = bp["resistance_k"].as<double>();
                if (bp["initial_shoot_angle"]) initial_shoot_angle_deg_ = bp["initial_shoot_angle"].as<double>();
            }
            
            // 加载相机到枪管偏移
            if (gc_config["camera_to_barrel_offset"]) {
                YAML::Node offset = gc_config["camera_to_barrel_offset"];
                if (offset["x"]) camera_to_barrel_x_ = offset["x"].as<double>();
                if (offset["y"]) camera_to_barrel_y_ = offset["y"].as<double>();
                if (offset["z"]) camera_to_barrel_z_ = offset["z"].as<double>();
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "加载配置文件失败: " << e.what() << std::endl;
        return false;
    }
}

// ==================== 装甲板朝向估计相关函数 ====================

ArmorPoseEstimation GeometricCompensator::estimateArmorOrientation(
    const Eigen::Vector3d& tvec_camera,
    const std::vector<cv::Point2f>& corners,
    bool is_big_armor,
    double initial_guess) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ArmorPoseEstimation result;
    result.success = false;
    result.iterations = 0;
    
    // 输入验证
    if (corners.size() != 4 || K_.empty()) {
        std::cerr << "输入数据无效：需要4个角点和有效的相机内参" << std::endl;
        return result;
    }
    
    // 确定pitch角（如果使用固定pitch）
    double pitch = use_fixed_pitch_ ? armor_fixed_pitch_ : 0.0;
    double roll = armor_fixed_roll_;
    
    // 定义代价函数：计算给定朝向角下的重投影误差
    auto costFunc = [&](double z_to_v) -> double {
        return computeReprojectionError(tvec_camera, corners, z_to_v, pitch, is_big_armor);
    };
    
    // 使用三分搜索找到最优的朝向角
    double search_center = M_PI + initial_guess;
    double search_min = search_center - search_range_;
    double search_max = search_center + search_range_;
    
    // 执行三分搜索
    int iterations = 0;
    double left = search_min;
    double right = search_max;
    
    while (right - left > search_epsilon_ && iterations < max_iterations_) {
        double mid1 = left + (right - left) / 3.0;
        double mid2 = right - (right - left) / 3.0;
        
        double cost1 = costFunc(mid1);
        double cost2 = costFunc(mid2);
        
        if (cost1 < cost2) {
            right = mid2;
        } else {
            left = mid1;
        }
        iterations++;
    }
    
    double best_z_to_v = (left + right) / 2.0;
    double min_error = costFunc(best_z_to_v);
    
    result.iterations = iterations;
    
    // 验证结果
    if (min_error < max_reprojection_error_) {
        result.success = true;
        result.z_to_v = best_z_to_v;
        result.zn_to_v = std::remainder(best_z_to_v - M_PI, 2 * M_PI);
        result.pitch = pitch;
        result.roll = roll;
        result.reprojection_error = min_error;
        
        // 构建装甲板到相机的旋转矩阵
        Eigen::Matrix3d R_z = Eigen::Matrix3d::Identity();
        R_z(0,0) = cos(best_z_to_v);
        R_z(0,1) = -sin(best_z_to_v);
        R_z(1,0) = sin(best_z_to_v);
        R_z(1,1) = cos(best_z_to_v);
        
        Eigen::Matrix3d R_pitch = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX()).matrix();
        Eigen::Matrix3d R_roll = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitY()).matrix();
        
        result.R_armor_to_camera = R_z * R_pitch * R_roll;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.search_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // std::cout << "\n=== 装甲板朝向角估计结果 ===" << std::endl;
        // std::cout << "原始朝向角(z_to_v): " << result.z_to_v * 180 / M_PI << " 度" << std::endl;
        // std::cout << "处理后朝向角(zn_to_v): " << result.zn_to_v * 180 / M_PI << " 度" << std::endl;
        // std::cout << "装甲板是否正对相机: " << (std::abs(result.zn_to_v) < M_PI/6 ? "是" : "否") << std::endl;
        // std::cout << "重投影误差: " << result.reprojection_error << " 像素" << std::endl;
        // std::cout << "迭代次数: " << result.iterations << std::endl;
        // std::cout << "搜索耗时: " << result.search_time_ms << " ms" << std::endl;
    } else {
        std::cerr << "朝向角估计失败：重投影误差过大 (" << min_error << " 像素)" << std::endl;
    }
    
    return result;
}

// ==================== 坐标转换函数 ====================

Eigen::Vector3d GeometricCompensator::cameraToBarrel(const Eigen::Vector3d& point_camera) const {
    // 简单的平移变换（枪管相对于相机的偏移）
    Eigen::Vector3d point_barrel;
    point_barrel(0) = point_camera(0) - camera_to_barrel_x_;
    point_barrel(1) = point_camera(1) - camera_to_barrel_y_;
    point_barrel(2) = point_camera(2) - camera_to_barrel_z_;
    return point_barrel;
}

Eigen::Vector3d GeometricCompensator::barrelToCamera(const Eigen::Vector3d& point_barrel) const {
    Eigen::Vector3d point_camera;
    point_camera(0) = point_barrel(0) + camera_to_barrel_x_;
    point_camera(1) = point_barrel(1) + camera_to_barrel_y_;
    point_camera(2) = point_barrel(2) + camera_to_barrel_z_;
    return point_camera;
}

// ==================== 弹道补偿相关函数 ====================

// 原有的相机坐标系弹道补偿（保持向后兼容）
BallisticCompensation GeometricCompensator::calculateBallisticCompensation(
    const Eigen::Vector3d& target_position_camera,
    double current_yaw,
    double current_pitch) {
    
    BallisticCompensation result;
    result.success = false;
    
    // 转换到枪管坐标系
    Eigen::Vector3d target_position_barrel = cameraToBarrel(target_position_camera);
    
    // 转换为米
    double target_x_m = target_position_barrel(0) / 1000.0;
    double target_y_m = target_position_barrel(1) / 1000.0;
    double target_z_m = target_position_barrel(2) / 1000.0;
    
    double horizontal_distance = std::sqrt(target_x_m * target_x_m + target_z_m * target_z_m);
    double height_diff = -target_y_m;  // 注意相机坐标系Y轴向下
    
    // 保存调试信息
    result.target_distance = horizontal_distance;
    result.height_diff = height_diff;
    
    // 检查有效范围
    if (horizontal_distance < 0.05 || horizontal_distance > 30.0 || bullet_speed_ < 10.0) {
        std::cerr << "弹道补偿失败：距离超出有效范围" << std::endl;
        return result;
    }
    
    try {
        // 求解发射角
        result.shoot_angle = solveShootAngle(horizontal_distance, height_diff, bullet_speed_);
        
        // 计算瞄准点
        double aim_height = horizontal_distance * std::tan(result.shoot_angle);
        
        Eigen::Vector3d aim_point_barrel;
        aim_point_barrel(0) = target_x_m;
        aim_point_barrel(1) = -aim_height;
        aim_point_barrel(2) = target_z_m;
        
        result.aim_point = aim_point_barrel * 1000.0; // 转回毫米
        
        double aim_distance = aim_point_barrel.norm();
        result.compensated_yaw = std::atan2(aim_point_barrel(0), aim_point_barrel(2));
        result.compensated_pitch = std::asin(-aim_point_barrel(1) / aim_distance);
        
        // 计算飞行时间
        result.time_of_flight = horizontal_distance / (bullet_speed_ * std::cos(result.shoot_angle));
        
        result.success = true;
        
    } catch (const std::exception& e) {
        std::cerr << "弹道补偿计算异常: " << e.what() << std::endl;
        result.success = false;
    }
    
    return result;
}

// 改进后的世界坐标系弹道补偿
BallisticCompensation GeometricCompensator::calculateBallisticCompensationWorld(
    const Eigen::Vector3d& target_position_world,
    double current_world_yaw,
    double current_world_pitch) {
    
    BallisticCompensation result;
    result.success = false;
    
    // 检查坐标转换器是否已设置
    if (!coord_converter_) {
        std::cerr << "错误：坐标转换器未设置！" << std::endl;
        return result;
    }
    
    try {
        // ========== 步骤1：坐标系转换 ==========
        // 将目标从世界坐标系转换到相机坐标系，再到枪管坐标系
        Eigen::Vector3d target_camera = coord_converter_->worldToCamera(target_position_world);
        Eigen::Vector3d target_barrel = cameraToBarrel(target_camera);
        
        // 转换为米单位进行弹道计算
        double x_m = target_barrel(0) / 1000.0;
        double y_m = target_barrel(1) / 1000.0;  
        double z_m = target_barrel(2) / 1000.0;
        
        // ========== 步骤2：弹道参数计算 ==========
        double horizontal_distance = std::sqrt(x_m * x_m + z_m * z_m);
        double height_diff = -y_m;  // 世界坐标系Z轴向上，所以高度差需要取反
        
        result.target_distance = horizontal_distance;
        result.height_diff = height_diff;
        
        // 检查有效范围
        if (horizontal_distance < 0.05 || horizontal_distance > 30.0) {
            std::cerr << "目标距离超出有效范围: " << horizontal_distance << "m" << std::endl;
            return result;
        }
        
        // ========== 步骤3：求解发射角 ==========
        result.shoot_angle = solveShootAngle(horizontal_distance, height_diff, bullet_speed_);
        
        // ========== 步骤4：计算瞄准点 ==========
        // 在枪管坐标系中计算瞄准点
        double aim_height = horizontal_distance * std::tan(result.shoot_angle);
        Eigen::Vector3d aim_point_barrel(x_m * 1000.0, -aim_height * 1000.0, z_m * 1000.0);
        result.aim_point = aim_point_barrel;
        
        // 将瞄准点转换回世界坐标系
        Eigen::Vector3d aim_point_camera = barrelToCamera(aim_point_barrel);
        Eigen::Vector3d aim_point_world = coord_converter_->cameraToWorld(aim_point_camera);
        
        // ========== 步骤5：使用坐标转换器的控制角度计算 ==========
        // 这里直接利用我们改进的坐标转换器来计算控制角度
        // 这是关键改进：使用"一轴两用"的投影方法
        GimbalControlAngles control = coord_converter_->calculateGimbalControl(aim_point_world);
        
        if (control.valid) {
            // 从控制结果中提取补偿后的角度
            result.compensated_yaw = current_world_yaw + control.yaw_adjustment * M_PI / 180.0;
            result.compensated_pitch = current_world_pitch + control.pitch_adjustment * M_PI / 180.0;
            
            // 计算补偿增量（用于调试）
            result.delta_yaw = control.yaw_adjustment * M_PI / 180.0;
            result.delta_pitch = control.pitch_adjustment * M_PI / 180.0;
            
            // 计算飞行时间
            result.time_of_flight = horizontal_distance / (bullet_speed_ * std::cos(result.shoot_angle));
            
            result.success = true;
            
            // 调试输出
            std::cout << "\n=== 弹道补偿计算（世界坐标系）===" << std::endl;
            std::cout << "目标位置(世界): " << target_position_world.transpose() / 1000.0 << " m" << std::endl;
            std::cout << "瞄准位置(世界): " << aim_point_world.transpose() / 1000.0 << " m" << std::endl;
            std::cout << "水平距离: " << horizontal_distance << " m" << std::endl;
            std::cout << "高度差: " << height_diff << " m" << std::endl;
            std::cout << "发射角: " << result.shoot_angle * 180.0 / M_PI << "°" << std::endl;
            std::cout << "Yaw调整: " << result.delta_yaw * 180.0 / M_PI << "°" << std::endl;
            std::cout << "Pitch调整: " << result.delta_pitch * 180.0 / M_PI << "°" << std::endl;
            std::cout << "飞行时间: " << result.time_of_flight << " s" << std::endl;
        } else {
            std::cerr << "云台控制角度计算失败" << std::endl;
            result.success = false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "弹道补偿计算异常: " << e.what() << std::endl;
        result.success = false;
    }
    
    return result;
}

// 计算发射角度（核心弹道算法）
double GeometricCompensator::solveShootAngle(double horizontal_distance, double height_diff, double bullet_speed) {
    double shoot_angle = initial_shoot_angle_deg_ / 180.0 * M_PI;
    
    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ResistanceFuncLinear, 1, 1>(
            new ResistanceFuncLinear(horizontal_distance, height_diff, bullet_speed, resistance_k_)
        ),
        nullptr,
        &shoot_angle
    );
    
    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    return shoot_angle;
}
// ==================== 辅助函数实现 ====================

double GeometricCompensator::computeReprojectionError(
    const Eigen::Vector3d& tvec_camera,
    const std::vector<cv::Point2f>& detected_corners,
    double z_to_v,
    double pitch,
    bool is_big_armor) const {
    
    // 生成装甲板局部坐标系下的角点
    std::vector<Eigen::Vector3d> local_corners = generateArmorCorners(is_big_armor);
    
    // 转换到相机坐标系
    std::vector<Eigen::Vector3d> camera_corners = transformCornersToCameraFrame(
        local_corners, tvec_camera, z_to_v, pitch, armor_fixed_roll_
    );
    
    // 投影到图像平面
    std::vector<cv::Point2f> projected_corners;
    for (const auto& corner_3d : camera_corners) {
        double fx = K_.at<double>(0, 0);
        double fy = K_.at<double>(1, 1);
        double cx = K_.at<double>(0, 2);
        double cy = K_.at<double>(1, 2);
        
        double x = corner_3d(0) / corner_3d(2);
        double y = corner_3d(1) / corner_3d(2);
        
        float px = static_cast<float>(fx * x + cx);
        float py = static_cast<float>(fy * y + cy);
        
        projected_corners.push_back(cv::Point2f(px, py));
    }
    
    // 计算详细的重投影误差
    DetailedError error = computeDetailedReprojectionError(projected_corners, detected_corners);
    
    return error.total_weighted;
}

std::vector<Eigen::Vector3d> GeometricCompensator::generateArmorCorners(bool is_big_armor) const {
    auto [width, height] = getArmorDimensions(is_big_armor);
    double half_width = width / 2.0;
    double half_height = height / 2.0;
    
    // 按照OpenCV的顺序：左下、右下、右上、左上
    std::vector<Eigen::Vector3d> corners = {
        {-half_width, -half_height, 0},
        { half_width, -half_height, 0},
        { half_width,  half_height, 0},
        {-half_width,  half_height, 0}
    };
    
    return corners;
}

std::vector<Eigen::Vector3d> GeometricCompensator::transformCornersToCameraFrame(
    const std::vector<Eigen::Vector3d>& local_corners,
    const Eigen::Vector3d& tvec_camera,
    double z_to_v,
    double pitch,
    double roll) const {
    
    // 构建从装甲板坐标系到相机坐标系的旋转矩阵
    Eigen::Matrix3d R_yaw = Eigen::Matrix3d::Identity();
    double actual_yaw = z_to_v - M_PI;
    R_yaw(0,0) = cos(actual_yaw);
    R_yaw(0,2) = -sin(actual_yaw);
    R_yaw(2,0) = sin(actual_yaw);
    R_yaw(2,2) = cos(actual_yaw);
    
    Eigen::Matrix3d R_pitch = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX()).matrix();
    Eigen::Matrix3d R_roll = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ()).matrix();
    
    Eigen::Matrix3d R_total = R_yaw * R_pitch * R_roll;
    
    // 转换每个角点
    std::vector<Eigen::Vector3d> camera_corners;
    for (const auto& local_corner : local_corners) {
        Eigen::Vector3d camera_corner = R_total * local_corner + tvec_camera;
        camera_corners.push_back(camera_corner);
    }
    
    return camera_corners;
}

GeometricCompensator::DetailedError GeometricCompensator::computeDetailedReprojectionError(
    const std::vector<cv::Point2f>& projected_corners,
    const std::vector<cv::Point2f>& detected_corners) const {
    
    DetailedError error;
    error.pixel_error = 0.0;
    error.edge_error = 0.0;
    error.angle_error = 0.0;
    
    // 1. 计算像素距离误差
    for (size_t i = 0; i < 4; i++) {
        double dx = projected_corners[i].x - detected_corners[i].x;
        double dy = projected_corners[i].y - detected_corners[i].y;
        error.pixel_error += std::sqrt(dx*dx + dy*dy);
    }
    error.pixel_error /= 4.0;
    
    // 2. 计算边长误差
    for (size_t i = 0; i < 4; i++) {
        size_t j = (i + 1) % 4;
        
        cv::Point2f detected_edge = detected_corners[j] - detected_corners[i];
        double detected_length = cv::norm(detected_edge);
        
        cv::Point2f projected_edge = projected_corners[j] - projected_corners[i];
        double projected_length = cv::norm(projected_edge);
        
        if (detected_length > 0) {
            error.edge_error += std::abs(projected_length - detected_length) / detected_length;
        }
    }
    error.edge_error /= 4.0;
    
    // 3. 计算角度误差
    for (size_t i = 0; i < 4; i++) {
        size_t prev = (i + 3) % 4;
        size_t next = (i + 1) % 4;
        
        cv::Point2f v1 = detected_corners[prev] - detected_corners[i];
        cv::Point2f v2 = detected_corners[next] - detected_corners[i];
        double detected_angle = std::atan2(v1.cross(v2), v1.dot(v2));
        
        cv::Point2f pv1 = projected_corners[prev] - projected_corners[i];
        cv::Point2f pv2 = projected_corners[next] - projected_corners[i];
        double projected_angle = std::atan2(pv1.cross(pv2), pv1.dot(pv2));
        
        error.angle_error += std::abs(detected_angle - projected_angle);
    }
    error.angle_error /= 4.0;
    
    // 计算加权总误差
    error.total_weighted = pixel_error_weight_ * error.pixel_error +
                          edge_error_weight_ * error.edge_error * 100.0 +
                          angle_error_weight_ * error.angle_error * 180.0 / M_PI;
    
    return error;
}

} // namespace aimer