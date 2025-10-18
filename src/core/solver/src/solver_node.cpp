#include "solver_node.hpp"
#include <rclcpp/rclcpp.hpp>

namespace solver {

Solver::Solver(const std::string& yaml_config_path) {
    // 所有模块从同一个YAML文件初始化
    pnp_solver_ = std::make_unique<PnPSolver>(yaml_config_path);
    coord_converter_ = std::make_unique<CoordConverter>(yaml_config_path);
    yaw_optimizer_ = std::make_unique<YawOptimizer>(yaml_config_path);
    
    RCLCPP_INFO(rclcpp::get_logger("Solver"), 
                "Solver系统初始化完成，配置文件: %s", 
                yaml_config_path.c_str());
}

void Solver::updateIMU(const Eigen::Quaterniond& q_absolute, double timestamp) {
    coord_converter_->updateIMU(q_absolute, timestamp);
}

void Solver::updateIMU(double yaw, double pitch, double timestamp) {
    coord_converter_->updateIMU(yaw, pitch, timestamp);
}

Armor_pose Solver::processArmor(const autoaim_msgs::msg::Armor& armor_msg, 
                               double timestamp) {
    Armor_pose armor_pose;
    armor_pose.timestamp = timestamp;
    
    armor_pose.id = parseArmorNumber(armor_msg.number);
    armor_pose.type = parseArmorType(armor_msg.type);
    
    std::vector<cv::Point2f> corners = extractCorners(armor_msg);
    
    
    last_pnp_result_ = pnp_solver_->solvePnP(corners, armor_pose.type, timestamp);

    
    if (!last_pnp_result_.success) {
        RCLCPP_WARN(rclcpp::get_logger("Solver"), 
                   "PnP求解失败，装甲板编号: %d", armor_msg.number);
        return armor_pose;
    }
    
    // 判断是否需要优化
    Eigen::Matrix3d optimized_R_armor_to_camera = last_pnp_result_.R_armor_to_camera;   // pnp出来的旋转矩阵
    
    bool need_optimization = true;
    // 大装甲板的3、4、5号不需要优化（通常是步兵）
    if (armor_pose.type == armor_auto_aim::ArmorType::big &&
        (armor_pose.id == armor_auto_aim::ArmorName::three ||
         armor_pose.id == armor_auto_aim::ArmorName::four ||
         armor_pose.id == armor_auto_aim::ArmorName::five)) {
        need_optimization = false;
    }

    // 进yaw优化
    
    if (need_optimization) {
        optimized_R_armor_to_camera = yaw_optimizer_->optimizeWithPrior(
            last_pnp_result_.camera_position,
            last_pnp_result_.R_armor_to_camera,
            armor_pose.type,
            armor_pose.id,
            corners,
            *coord_converter_
        );
    }

    // utils::logger()->debug("旋转矩阵：");
    // for (int i = 0; i < 3; ++i) {
    //     // 使用 Eigen 的括号运算符来访问矩阵元素
    //     utils::logger()->debug("  [{:.3f}, {:.3f}, {:.3f}]", 
    //         optimized_R_armor_to_camera(i,0), optimized_R_armor_to_camera(i,1), optimized_R_armor_to_camera(i,2));
    // }
    
    // 保存相机坐标系结果
    armor_pose.camera_position = last_pnp_result_.camera_position / 1000.0;
    // armor_pose.camera_orientation = coord_converter_->extractOrientation(     // 传入参数旋转矩阵，进行欧拉角的提取，得到ypr
    //     
    // );
    armor_pose.camera_orientation = Orientation(optimized_R_armor_to_camera, timestamp);
    // utils::logger()->debug(
    //     "processArmor - 相机姿态提取完成: yaw={:.2f}°, pitch={:.2f}°, roll={:.2f}°",
    //     armor_pose.camera_orientation.yaw,
    //     armor_pose.camera_orientation.pitch,
    //     armor_pose.camera_orientation.roll
    // );
    
    // 转换到世界坐标系
    armor_pose.world_position = coord_converter_->transform(
        last_pnp_result_.camera_position / 1000.0,
        CoordinateFrame::CAMERA,
        CoordinateFrame::WORLD
    );
    
    // Eigen::Matrix3d R_armor_to_world = coord_converter_->transformRotation(
    //     optimized_R_armor_to_camera,
    //     CoordinateFrame::CAMERA,
    //     CoordinateFrame::WORLD
    // );
    
    // armor_pose.world_orientation = Orientation(
    //     R_armor_to_world, timestamp
    // );

    armor_pose.world_orientation =  armor_pose.camera_orientation;
    
    armor_pose.world_spherical = SphericalCoord(armor_pose.world_position, timestamp);
    
    last_armor_pose_ = armor_pose;

    // 这里armor_pose的世界球坐标和姿态都需要是弧度值
    
    
    return armor_pose;
}

Gimbal Solver::getCurrentGimbal() const {
    YawPitch current = coord_converter_->getCurrentAngles();
    
    if (last_armor_pose_.world_position.norm() > 1e-6) {
        return Gimbal(current, last_armor_pose_.world_position, current.timestamp);
    }
    
    return Gimbal(current, current, current.timestamp);
}

ArmorName Solver::parseArmorNumber(int number) const {
    switch(number) {
        case 1: return armor_auto_aim::ArmorName::one;
        case 2: return armor_auto_aim::ArmorName::two;
        case 3: return armor_auto_aim::ArmorName::three;
        case 4: return armor_auto_aim::ArmorName::four;
        case 5: return armor_auto_aim::ArmorName::five;
        case 6: return armor_auto_aim::ArmorName::sentry;
        case 7: return armor_auto_aim::ArmorName::outpost;
        case 8: return armor_auto_aim::ArmorName::base;
        default: return armor_auto_aim::ArmorName::not_armor;
    }
}

ArmorType Solver::parseArmorType(const std::string& type_str) const {
    if (type_str == "big" || type_str == "large") {
        return armor_auto_aim::ArmorType::big;
    }
    return armor_auto_aim::ArmorType::small;
}

std::vector<cv::Point2f> Solver::extractCorners(const autoaim_msgs::msg::Armor& armor_msg) const {
    std::vector<cv::Point2f> corners;
    corners.reserve(4);
    
    for (const auto& point : armor_msg.corners) {
        corners.emplace_back(point.x, point.y);
    }
    
    return corners;
}

}